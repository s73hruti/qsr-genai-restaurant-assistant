"""API hardening: IDOR on user_id, client API key, rate limiting, proxy-aware IPs, prod config."""
import pytest
from fastapi.testclient import TestClient

from qsr_assistant.api import main
from qsr_assistant.api import security as sec


@pytest.fixture()
def client(agent, monkeypatch):
    monkeypatch.setattr(main, "get_agent", lambda: agent)
    for v in ("MANAGER_API_KEY", "API_KEY", "USER_TOKEN_SECRET", "APP_ENV", "TRUSTED_PROXY_HOPS"):
        monkeypatch.delenv(v, raising=False)
    main._limiter.buckets.clear()
    return TestClient(main.app)


# ----------------------------------------------------------------------------- user identity
def test_recommend_requires_matching_user_token(client, monkeypatch):
    monkeypatch.setenv("USER_TOKEN_SECRET", "u-secret")
    assert client.get("/recommend/7").status_code == 403                       # no token
    assert client.get("/recommend/7", headers={"X-User-Token": "nope"}).status_code == 403
    mine = {"X-User-Token": sec.mint_user_token(7)}
    assert client.get("/recommend/7", headers=mine).status_code == 200
    assert client.get("/recommend/8", headers=mine).status_code == 403          # someone else's id


def test_chat_cannot_impersonate_another_user(client, monkeypatch):
    monkeypatch.setenv("USER_TOKEN_SECRET", "u-secret")
    body = {"message": "what should I order?", "user_id": 5}
    assert client.post("/chat", json=body).status_code == 403
    ok = client.post("/chat", json=body, headers={"X-User-Token": sec.mint_user_token(5)})
    assert ok.status_code == 200
    other = client.post("/chat", json={**body, "user_id": 6},
                        headers={"X-User-Token": sec.mint_user_token(5)})
    assert other.status_code == 403
    # anonymous requests (no user_id) stay allowed
    assert client.post("/chat", json={"message": "hi"}).status_code == 200


def test_session_delete_is_user_scoped(client, monkeypatch):
    monkeypatch.setenv("USER_TOKEN_SECRET", "u-secret")
    r = client.delete("/sessions/abcdefgh?user_id=3")
    assert r.status_code == 403
    r = client.delete("/sessions/abcdefgh?user_id=3", headers={"X-User-Token": sec.mint_user_token(3)})
    assert r.status_code == 200


def test_manager_key_lets_a_manager_inspect_any_user(client, monkeypatch):
    monkeypatch.setenv("USER_TOKEN_SECRET", "u-secret")
    monkeypatch.setenv("MANAGER_API_KEY", "mgr")
    assert client.get("/recommend/9", headers={"X-Manager-Key": "mgr"}).status_code == 200
    assert client.get("/recommend/9", headers={"X-Manager-Key": "wrong"}).status_code == 403


def test_open_demo_mode_unchanged_and_reported(client):
    assert client.get("/recommend/7").status_code == 200
    h = client.get("/health").json()
    assert h["client_auth"] is False and h["user_auth"] is False and h["manager_auth"] is False


# ----------------------------------------------------------------------------- client API key
def test_api_key_protects_llm_and_data_endpoints_but_not_health(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "k-123")
    assert client.get("/health").status_code == 200
    assert client.get("/health").json()["client_auth"] is True
    for call in (lambda h: client.post("/extract/ticket", json={"text": "cold fries"}, headers=h),
                 lambda h: client.post("/chat", json={"message": "hi"}, headers=h),
                 lambda h: client.get("/wait-time/1", headers=h),
                 lambda h: client.get("/recommend/1", headers=h)):
        assert call({}).status_code == 401
        assert call({"X-API-Key": "wrong"}).status_code == 401
        assert call({"X-API-Key": "k-123"}).status_code == 200


# ----------------------------------------------------------------------------- rate limiting
def test_llm_paths_have_their_own_tighter_bucket(client, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "100")
    monkeypatch.setenv("RATE_LIMIT_LLM_PER_MIN", "2")
    codes = [client.post("/extract/ticket", json={"text": "cold fries"}).status_code for _ in range(4)]
    assert codes[:2] == [200, 200] and codes[2:] == [429, 429]
    assert client.get("/wait-time/1").status_code == 200       # standard bucket untouched
    assert client.get("/health").status_code == 200             # health is never limited


def test_rate_limit_is_on_by_default_in_code(monkeypatch):
    monkeypatch.delenv("RATE_LIMIT_PER_MIN", raising=False)
    monkeypatch.delenv("RATE_LIMIT_LLM_PER_MIN", raising=False)
    assert sec.rate_limit_for("/wait-time/1") == sec.DEFAULT_RATE_PER_MIN > 0
    assert sec.rate_limit_for("/extract/ticket") == sec.DEFAULT_LLM_RATE_PER_MIN > 0


def test_client_ip_ignores_xff_unless_proxy_trusted(monkeypatch):
    h = {"x-forwarded-for": "6.6.6.6, 10.0.0.9"}
    monkeypatch.delenv("TRUSTED_PROXY_HOPS", raising=False)
    assert sec.client_ip(h, "1.2.3.4") == "1.2.3.4"             # spoofable header never trusted
    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "1")
    assert sec.client_ip(h, "172.16.0.1") == "10.0.0.9"         # what our proxy actually saw
    assert sec.client_ip({"x-forwarded-for": "6.6.6.6, 5.5.5.5, 10.0.0.9"}, "p") == "10.0.0.9"
    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "2")
    assert sec.client_ip(h, "172.16.0.1") == "6.6.6.6"
    assert sec.client_ip({}, "172.16.0.1") == "172.16.0.1"      # header absent: fall back to peer


def test_spoofed_xff_cannot_dodge_the_limit_without_trusted_proxy(client, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "2")
    codes = [client.get("/wait-time/1", headers={"X-Forwarded-For": f"9.9.9.{i}"}).status_code
             for i in range(4)]
    assert codes == [200, 200, 429, 429]


def test_limiter_eviction_does_not_reset_active_clients(monkeypatch):
    lim = sec.RateLimiter()
    lim.MAX_KEYS = 50
    assert lim.allow("attacker", 2) and lim.allow("attacker", 2) and not lim.allow("attacker", 2)
    for i in range(200):                       # flood with throw-away keys to force eviction
        lim.allow(f"burner-{i}", 2)
    assert len(lim.buckets) <= 50
    assert not lim.allow("attacker", 2), "flooding must not hand back a fresh bucket"


# ----------------------------------------------------------------------------- production guard
def test_production_refuses_to_start_without_secrets(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    for v in ("API_KEY", "MANAGER_API_KEY", "USER_TOKEN_SECRET"):
        monkeypatch.delenv(v, raising=False)
    assert sec.check_production_config() == ["API_KEY", "MANAGER_API_KEY", "USER_TOKEN_SECRET"]
    with pytest.raises(sec.ConfigError, match="API_KEY"):
        sec.enforce_production_config()
    with pytest.raises(sec.ConfigError):
        with TestClient(main.app):               # the lifespan hook runs on startup
            pass
    for v in ("API_KEY", "MANAGER_API_KEY", "USER_TOKEN_SECRET"):
        monkeypatch.setenv(v, "x")
    assert sec.check_production_config() == []
    monkeypatch.setenv("APP_ENV", "dev")
    monkeypatch.delenv("API_KEY")
    assert sec.check_production_config() == []
