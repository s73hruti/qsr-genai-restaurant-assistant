"""Lightweight tracing: a trace per request with timed, attributed spans.

* Keeps the last N traces in memory (served by the API's /traces endpoint).
* TRACE_FILE=path appends each finished trace as one JSON line.
* LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY (+ `pip install ".[observability]"`) mirrors traces to
  Langfuse through a best-effort adapter (written against the v3 SDK); failures never affect requests.

Trace objects are passed explicitly (no contextvars) so they are safe with streaming generators
that are consumed from different threads.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager

from ..config import env


class _LangfuseAdapter:
    def __init__(self):
        from langfuse import get_client  # v3 SDK
        self.client = get_client()

    def start(self, name, attrs, parent=None):
        try:
            owner = parent if parent is not None else self.client
            return owner.start_span(name=name, input=attrs)
        except Exception:
            return None

    def end(self, handle, attrs):
        try:
            if handle is not None:
                handle.update(output=attrs)
                handle.end()
        except Exception:
            pass


class Trace:
    def __init__(self, tracer: "Tracer", name: str, attrs: dict):
        self.tracer = tracer
        self.data = {"trace_id": uuid.uuid4().hex[:16], "name": name, "ts": time.time(),
                     "attrs": dict(attrs), "spans": []}
        self._lf = tracer.langfuse.start(name, attrs) if tracer.langfuse else None
        self._t0 = time.perf_counter()

    @property
    def id(self) -> str:
        return self.data["trace_id"]

    def set(self, **attrs) -> None:
        self.data["attrs"].update(attrs)

    @contextmanager
    def span(self, name: str, **attrs):
        s = {"name": name, "attrs": dict(attrs)}
        lf = self.tracer.langfuse
        handle = lf.start(name, attrs, self._lf) if lf else None
        t0 = time.perf_counter()
        try:
            yield s
        finally:
            s["ms"] = round((time.perf_counter() - t0) * 1000, 1)
            if lf:
                lf.end(handle, s["attrs"])
            self.data["spans"].append(s)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.data["duration_ms"] = round((time.perf_counter() - self._t0) * 1000, 1)
        if self.tracer.langfuse:
            self.tracer.langfuse.end(self._lf, self.data["attrs"])
        self.tracer._finish(self.data)
        return False


class Tracer:
    def __init__(self, keep: int = 200):
        self.recent: deque = deque(maxlen=keep)
        self.file = env("TRACE_FILE")
        self._lock = threading.Lock()
        self.langfuse = None
        if env("LANGFUSE_PUBLIC_KEY") and env("LANGFUSE_SECRET_KEY"):
            try:
                self.langfuse = _LangfuseAdapter()
            except Exception:
                self.langfuse = None

    def trace(self, name: str, **attrs) -> Trace:
        return Trace(self, name, attrs)

    def _finish(self, data: dict) -> None:
        with self._lock:
            self.recent.append(data)
            if self.file:
                with open(self.file, "a") as f:
                    f.write(json.dumps(data, default=str) + "\n")

    def last(self, n: int = 20) -> list[dict]:
        return list(self.recent)[-n:]
