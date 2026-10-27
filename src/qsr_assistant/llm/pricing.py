"""Rough cost estimation. Prices are USD per 1M tokens (input, output) and are APPROXIMATE.

Check your provider's pricing page and override with LLM_PRICE_IN / LLM_PRICE_OUT (USD per 1M).
"""
from __future__ import annotations

from ..config import env
from .types import Usage

PRICES = {  # prefix -> (input, output) USD per 1M tokens; estimates, verify before relying on them
    "claude-haiku": (1.0, 5.0),
    "claude-sonnet": (3.0, 15.0),
    "claude-opus": (15.0, 75.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.5, 10.0),
}


def estimate_cost(model: str, usage: Usage) -> float:
    if env("LLM_PRICE_IN") and env("LLM_PRICE_OUT"):
        p_in, p_out = float(env("LLM_PRICE_IN")), float(env("LLM_PRICE_OUT"))
    else:
        match = next((v for k, v in PRICES.items() if model.startswith(k)), None)
        if match is None:
            return 0.0
        p_in, p_out = match
    return (usage.input_tokens * p_in + usage.output_tokens * p_out) / 1_000_000
