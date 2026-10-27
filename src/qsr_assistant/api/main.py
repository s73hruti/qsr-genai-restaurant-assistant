"""FastAPI service: /chat, /chat/stream (SSE), sessions, metrics, traces and direct model endpoints.

Manager mode: if MANAGER_API_KEY is set, requests with mode=manager must send the same value in the
`X-Manager-Key` header. In production derive the mode from your auth layer (JWT claims etc.); the
mode must never be trusted from the client alone. If MANAGER_API_KEY is unset (local demo) the
mode is open and /health reports `manager_auth: false`.
"""
from __future__ import annotations

import hmac
import json
import threading
import time
from datetime import date, datetime, timedelta
from functools import lru_cache

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..agent.agent import Agent
from ..agent.memory import SessionMemory
from ..config import env
from ..llm.structured import extract_ticket
from ..llm.vision import ImageError, image_to_query

app = FastAPI(title="QSR GenAI Assistant", version="0.3.0")


class RateLimiter:
    """Per-client token bucket (in-process). Use Redis or an API gateway for multi-replica setups."""

    def __init__(self):
        self.buckets: dict[str, tuple[float, float]] = {}
        self.lock = threading.Lock()

    def allow(self, client: str, per_min: int) -> bool:
        now = time.monotonic()
        with self.lock:
            tokens, last = self.buckets.get(client, (float(per_min), now))
            tokens = min(float(per_min), tokens + (now - last) * per_min / 60.0)
            ok = tokens >= 1.0
            self.buckets[client] = (tokens - 1.0 if ok else tokens, now)
            if len(self.buckets) > 10_000:  # bound memory
                self.buckets.clear()
            return ok


_limiter = RateLimiter()


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    per_min = int(env("RATE_LIMIT_PER_MIN", "0") or 0)
    if per_min > 0 and request.url.path != "/health":
        who = request.client.host if request.client else "unknown"
        if not _limiter.allow(who, per_min):
            return JSONResponse({"detail": "rate limit exceeded"}, status_code=429,
                                headers={"Retry-After": "10"})
    return await call_next(request)


@lru_cache(maxsize=1)
def get_agent() -> Agent:
    return Agent()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    mode: str = Field("customer", pattern="^(customer|manager)$")
    user_id: int | None = Field(None, ge=1)
    store_id: int | None = Field(None, ge=1, le=20)
    session_id: str | None = Field(None, min_length=8, max_length=64, pattern=r"^[\w\-]+$")


class ImageChatRequest(ChatRequest):
    message: str = Field(min_length=1, max_length=400)  # leaves room under the 1000-char input cap
    image_base64: str = Field(min_length=8, max_length=6_000_000)
    media_type: str = Field("image/jpeg", pattern="^image/(png|jpeg|gif|webp)$")


class TicketRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


def _authorize(mode: str, key: str | None) -> None:
    required = env("MANAGER_API_KEY")
    if mode == "manager" and required and not hmac.compare_digest(key or "", required):
        raise HTTPException(403, "manager mode requires a valid X-Manager-Key")


@app.get("/health")
def health():
    a = get_agent()
    r = a.reg.retriever
    return {"status": "ok", "llm_backend": a.llm.name, "llm_model": a.llm.model,
            "retrieval": {"mode": r.mode, "rerank": r.use_rerank, "rewrite": r.use_rewrite,
                          "embedder": r.embedder.name, "vector_store": r.store.name,
                          "documents": r.n_docs, "chunks": len(r.chunks)},
            "manager_auth": bool(env("MANAGER_API_KEY")), "output_guard_strict": a.guard_strict,
            "models": a.reg.metrics}


@app.post("/chat")
def chat(req: ChatRequest, x_manager_key: str | None = Header(None)):
    _authorize(req.mode, x_manager_key)
    return get_agent().run(req.message, req.mode, req.user_id, req.store_id,
                           req.session_id).to_dict()


@app.post("/chat/image")
def chat_image(req: ImageChatRequest, x_manager_key: str | None = Header(None)):
    """Multimodal chat: the image is described by a vision model, sanitised, then answered by the
    normal agent loop. Without a configured vision model the image is acknowledged but not analysed."""
    _authorize(req.mode, x_manager_key)
    agent = get_agent()
    try:
        query, info = image_to_query(agent.llm, req.message, req.image_base64, req.media_type)
    except ImageError as e:
        raise HTTPException(422, str(e))
    out = agent.run(query, req.mode, req.user_id, req.store_id, req.session_id).to_dict()
    out["image"] = info
    return out


@app.post("/extract/ticket")
def extract(req: TicketRequest):
    """Free-text complaint -> schema-validated support ticket (LLM with repair loop, else rules)."""
    return extract_ticket(req.text, get_agent().llm)


@app.post("/chat/stream")
def chat_stream(req: ChatRequest, x_manager_key: str | None = Header(None)):
    """Server-sent events: token | tool_call | tool_result | degraded | final.

    Tokens are streamed as generated; the `final` event carries the authoritative answer, which
    can differ from the streamed text if an output guardrail replaced it - clients should replace.
    """
    _authorize(req.mode, x_manager_key)
    agent = get_agent()

    def events():
        for ev in agent.stream(req.message, req.mode, req.user_id, req.store_id, req.session_id):
            if ev["type"] == "final":
                ev = {"type": "final", "response": ev["response"].to_dict()}
            yield f"data: {json.dumps(ev, default=str)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str, mode: str = "customer", user_id: int | None = None,
                   x_manager_key: str | None = Header(None)):
    _authorize(mode, x_manager_key)
    key = SessionMemory.key(mode, user_id, session_id)
    return {"cleared": get_agent().memory.clear(key)}


@app.get("/metrics")
def metrics(x_manager_key: str | None = Header(None)):
    _authorize("manager", x_manager_key)
    return get_agent().metrics.snapshot()


@app.get("/traces")
def traces(limit: int = 20, x_manager_key: str | None = Header(None)):
    _authorize("manager", x_manager_key)
    return get_agent().tracer.last(min(limit, 100))


@app.get("/forecast/{store_id}")
def forecast(store_id: int, days: int = 7, x_manager_key: str | None = Header(None)):
    _authorize("manager", x_manager_key)
    reg = get_agent().reg
    try:
        fc = reg.forecaster.predict(store_id, date.today() + timedelta(days=1), min(days, 28))
    except ValueError as e:
        raise HTTPException(404, str(e))
    return [{"date": str(d.date()), "orders": int(v)} for d, v in zip(fc.date, fc.forecast)]


@app.get("/inventory/{store_id}")
def inventory(store_id: int, days: int = 3, x_manager_key: str | None = Header(None)):
    _authorize("manager", x_manager_key)
    reg = get_agent().reg
    try:
        plan = reg.forecaster.inventory_plan(store_id, date.today() + timedelta(days=1), min(days, 14))
    except ValueError as e:
        raise HTTPException(404, str(e))
    return plan.to_dict("records")


@app.get("/wait-time/{store_id}")
def wait(store_id: int, hour: int | None = None, queue_len: float | None = None):
    now = datetime.now()
    reg = get_agent().reg
    try:
        return reg.waittime.predict(store_id, hour if hour is not None else now.hour,
                                    now.weekday(), queue_len)
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.get("/recommend/{user_id}")
def recommend(user_id: int, hour: int | None = None, temp_c: float = 28.0, k: int = 3):
    reg = get_agent().reg
    return reg.recommender.recommend(user_id, hour if hour is not None else datetime.now().hour,
                                     temp_c, k)
