"""Ranking metrics, text-match helpers and bootstrap confidence intervals."""
from __future__ import annotations

import math
import re

import numpy as np

from ..guardrails.output import numbers
from ..rag.embeddings import unigram_analyzer


def dedupe(ids: list[str]) -> list[str]:
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def hit_at_k(ranked: list[str], gold: str, k: int) -> float:
    return float(gold in ranked[:k])


def mrr_at_k(ranked: list[str], gold: str, k: int) -> float:
    for r, d in enumerate(ranked[:k], 1):
        if d == gold:
            return 1.0 / r
    return 0.0


def ndcg_at_k(ranked: list[str], gold: str, k: int) -> float:
    for r, d in enumerate(ranked[:k], 1):
        if d == gold:
            return 1.0 / math.log2(r + 1)  # single relevant doc: ideal DCG = 1
    return 0.0


def bootstrap_ci(values: list[float], n: int = 1000, seed: int = 0) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[*_`]", "", text.lower())).strip()


def fact_in_text(expect: str, text: str) -> bool:
    """Exact (normalised) substring match, or - for paraphrased answers - every number in the
    expected fact is present and >=80% of its content words are."""
    e, t = norm(expect), norm(text)
    if e in t:
        return True
    exp_nums = numbers(e)
    if exp_nums and not exp_nums <= numbers(t):
        return False
    et, tt = set(unigram_analyzer(e)), set(unigram_analyzer(t))
    if exp_nums:  # numbers are the discriminating content; allow wording changes around them
        return len(et) <= 1 or bool(et & tt)
    return bool(et) and len(et & tt) / len(et) >= 0.8


ABSTAIN_RX = re.compile(
    r"(don't|do not|doesn't|does not|can't|cannot|couldn't|unable to|no) (have|find|provide|see|"
    r"access|answer|help)|not (available|something i|able)|only available to store managers|"
    r"i('m| am) (not sure|sorry)|outside (of )?(what|my)|can't help", re.I)


def is_abstention(answer: str) -> bool:
    return bool(ABSTAIN_RX.search(answer))


def pct(x: float) -> float:
    return round(100 * x, 1)
