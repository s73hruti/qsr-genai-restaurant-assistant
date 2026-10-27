"""In-process counters and latency percentiles for the /metrics endpoint."""
from __future__ import annotations

import threading
from collections import Counter, deque

import numpy as np


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.counts: Counter = Counter()
        self.latencies: deque = deque(maxlen=2000)
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0

    def record(self, *, latency_ms: float, usage, cost: float, cached: bool, blocked: bool,
               tools: list[str], guard_flags: list[str], backend: str) -> None:
        with self._lock:
            self.counts["requests"] += 1
            self.counts["cache_hits"] += int(cached)
            self.counts["input_blocked"] += int(blocked)
            self.counts[f"backend:{backend}"] += 1
            for t in tools:
                self.counts[f"tool:{t}"] += 1
            for f in guard_flags:
                self.counts[f"guard:{f}"] += 1
            self.latencies.append(latency_ms)
            self.input_tokens += usage.input_tokens
            self.output_tokens += usage.output_tokens
            self.cost_usd += cost

    def snapshot(self) -> dict:
        with self._lock:
            lat = list(self.latencies)
            return {"counts": dict(self.counts),
                    "latency_ms": ({"p50": round(float(np.percentile(lat, 50)), 1),
                                    "p95": round(float(np.percentile(lat, 95)), 1)} if lat else {}),
                    "tokens": {"input": self.input_tokens, "output": self.output_tokens},
                    "estimated_cost_usd": round(self.cost_usd, 6)}
