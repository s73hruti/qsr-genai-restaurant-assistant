"""Per-session conversation memory: a sliding window plus a short extractive summary.

Sessions are namespaced by (mode, user_id, session_id) so a conversation held in manager mode can
never be replayed into a customer-mode request, even with the same session id.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict

from ..llm.types import Message


class SessionMemory:
    def __init__(self, max_turns: int = 6, ttl_s: int = 3600, max_sessions: int = 1000):
        self.max_turns, self.ttl_s, self.max_sessions = max_turns, ttl_s, max_sessions
        self._data: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def key(mode: str, user_id: int | None, session_id: str | None) -> str | None:
        return f"{mode}:{user_id}:{session_id}" if session_id else None

    def _evict(self) -> None:
        now = time.time()
        for k in [k for k, v in self._data.items() if now - v["ts"] > self.ttl_s]:
            del self._data[k]
        while len(self._data) > self.max_sessions:
            self._data.popitem(last=False)

    def get(self, key: str | None) -> tuple[str, list[Message]]:
        if key is None:
            return "", []
        with self._lock:
            self._evict()
            s = self._data.get(key)
            if not s:
                return "", []
            self._data.move_to_end(key)
            msgs = [Message(r, c) for turn in s["turns"] for r, c in turn]
            return s["summary"], msgs

    def add(self, key: str | None, user_text: str, assistant_text: str) -> None:
        if key is None:
            return
        with self._lock:
            s = self._data.setdefault(key, {"turns": [], "summary": "", "ts": time.time()})
            s["turns"].append([("user", user_text), ("assistant", assistant_text)])
            s["ts"] = time.time()
            while len(s["turns"]) > self.max_turns:
                old = s["turns"].pop(0)
                s["summary"] = (s["summary"] + f" User asked: {old[0][1][:80]}.").strip()[-500:]
            self._data.move_to_end(key)
            self._evict()

    def clear(self, key: str | None) -> bool:
        with self._lock:
            return self._data.pop(key, None) is not None if key else False

    def user_turns(self, key: str | None) -> list[str]:
        return [m.content for m in self.get(key)[1] if m.role == "user"]
