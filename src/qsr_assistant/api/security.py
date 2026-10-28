"""API hardening: client IP resolution, rate limiting, API key, per-user tokens, prod config check.

Threat model (what each control is for)
  * API_KEY            - every call except /health must carry X-API-Key. Stops anonymous use of the
                         LLM-backed endpoints (cost abuse) and of the data endpoints.
  * MANAGER_API_KEY    - mode=manager and /metrics, /traces, /forecast, /inventory need X-Manager-Key.
  * USER_TOKEN_SECRET  - a request that names a user_id must carry X-User-Token, an HMAC of that id
                         minted by YOUR backend after it authenticated the user. Without it any caller
                         could read another user's personalised results (IDOR). In a real deployment
                         replace this with the JWT your auth layer already issues; the point is that
                         `user_id` is never trusted from the request body alone.
  * Rate limits        - on by default; the LLM-backed paths get a tighter, separate bucket.
  * APP_ENV=production - refuses to start unless all three secrets are configured.

With none of these set the service runs in open demo mode and /health says so.
"""
from __future__ import annotations

import hashlib
import hmac
import threading
import time

from ..config import env

# Paths that spend LLM tokens (or call a vision model): limited more tightly than the rest.
LLM_PATHS = frozenset({"/chat", "/chat/stream", "/chat/image", "/extract/ticket"})
PUBLIC_PATHS = frozenset({"/health"})

DEFAULT_RATE_PER_MIN = 60
DEFAULT_LLM_RATE_PER_MIN = 20


class ConfigError(RuntimeError):
    pass


# ------------------------------------------------------------------------------ client identity
def client_ip(headers, peer: str | None) -> str:
    """Return the address to rate-limit on.

    Behind a reverse proxy every request has the proxy as its TCP peer, so a naive limiter puts all
    users in one bucket. With TRUSTED_PROXY_HOPS=N we take the entry N from the right of
    X-Forwarded-For (the address our outermost trusted proxy saw). Entries further left are
    client-supplied and spoofable, so they are never used. Default 0 = ignore the header entirely.
    """
    hops = int(env("TRUSTED_PROXY_HOPS", "0") or 0)
    if hops > 0:
        xff = [p.strip() for p in headers.get("x-forwarded-for", "").split(",") if p.strip()]
        if len(xff) >= hops:
            return xff[-hops]
    return peer or "unknown"


# ------------------------------------------------------------------------------ rate limiting
class RateLimiter:
    """Per-key token bucket (in-process). Use Redis or an API gateway for multi-replica setups."""

    MAX_KEYS = 10_000

    def __init__(self):
        self.buckets: dict[str, tuple[float, float]] = {}
        self.lock = threading.Lock()

    def allow(self, key: str, per_min: int) -> bool:
        now = time.monotonic()
        with self.lock:
            tokens, last = self.buckets.get(key, (float(per_min), now))
            tokens = min(float(per_min), tokens + (now - last) * per_min / 60.0)
            ok = tokens >= 1.0
            self.buckets[key] = (tokens - 1.0 if ok else tokens, now)
            if len(self.buckets) > self.MAX_KEYS:
                self._evict(now, per_min)
            return ok

    def _evict(self, now: float, per_min: int) -> None:
        # Forget the clients whose bucket is (nearly) full first: forgetting them costs nothing
        # because they would get a full bucket back anyway. Throttled clients (few tokens left) are
        # kept, so flooding with throw-away keys cannot reset an attacker's own bucket. Evict down
        # to 90% so this O(n log n) pass does not run on every request once the table is full.
        def tokens(k: str) -> float:
            t, last = self.buckets[k]
            return min(float(per_min), t + (now - last) * per_min / 60.0)

        target = int(self.MAX_KEYS * 0.9)
        for k in sorted(self.buckets, key=tokens, reverse=True)[: max(0, len(self.buckets) - target)]:
            del self.buckets[k]


def rate_limit_for(path: str) -> int:
    """Requests/min allowed for this path; 0 disables limiting."""
    if path in LLM_PATHS:
        return int(env("RATE_LIMIT_LLM_PER_MIN", str(DEFAULT_LLM_RATE_PER_MIN)) or 0)
    return int(env("RATE_LIMIT_PER_MIN", str(DEFAULT_RATE_PER_MIN)) or 0)


def bucket_key(ip: str, path: str) -> str:
    return f"{ip}|{'llm' if path in LLM_PATHS else 'std'}"


# ------------------------------------------------------------------------------ credentials
def _eq(a: str | None, b: str) -> bool:
    return hmac.compare_digest((a or "").encode(), b.encode())


def api_key_ok(provided: str | None) -> bool:
    required = env("API_KEY")
    return True if not required else _eq(provided, required)


def manager_key_ok(provided: str | None) -> bool:
    required = env("MANAGER_API_KEY")
    return True if not required else _eq(provided, required)


def mint_user_token(user_id: int, secret: str | None = None) -> str:
    """Token your auth backend gives the client for `user_id` (HMAC-SHA256, hex)."""
    key = (secret if secret is not None else env("USER_TOKEN_SECRET")).encode()
    return hmac.new(key, f"user:{int(user_id)}".encode(), hashlib.sha256).hexdigest()


def user_token_ok(user_id: int | None, token: str | None, mode: str,
                  manager_key: str | None) -> bool:
    """May this caller act as `user_id`?

    * no user_id (anonymous)                -> yes
    * USER_TOKEN_SECRET unset (demo)        -> yes
    * a valid manager key (mode=manager)    -> yes, managers can inspect any user
    * otherwise the token must match this exact user_id
    """
    if user_id is None or not env("USER_TOKEN_SECRET"):
        return True
    if mode == "manager" and env("MANAGER_API_KEY") and _eq(manager_key, env("MANAGER_API_KEY")):
        return True
    return _eq(token, mint_user_token(user_id))


# ------------------------------------------------------------------------------ startup check
def check_production_config() -> list[str]:
    """Names of missing secrets when APP_ENV=production (empty list = fine / not production)."""
    if env("APP_ENV").lower() != "production":
        return []
    return [n for n in ("API_KEY", "MANAGER_API_KEY", "USER_TOKEN_SECRET") if not env(n)]


def enforce_production_config() -> None:
    missing = check_production_config()
    if missing:
        raise ConfigError("APP_ENV=production but these secrets are not set: " + ", ".join(missing))
