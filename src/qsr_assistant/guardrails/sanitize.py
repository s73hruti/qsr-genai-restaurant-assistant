"""Neutralise instructions hidden inside retrieved documents (indirect prompt injection)."""
from __future__ import annotations

import re

from .patterns import COMPILED

NEUTRAL = "[removed: instruction-like text in document]"


def sanitize_context(text: str) -> str:
    out = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        if any(rx.search(sentence) for _, rx in COMPILED):
            out.append(NEUTRAL)
        else:
            out.append(sentence)
    return " ".join(s for s in out if s)
