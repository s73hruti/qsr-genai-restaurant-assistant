"""Output guardrails: numeric grounding, manager-info leak detection, system-prompt leak.

Numeric grounding is the cheap, high-value check for a restaurant assistant: every price, time,
count or percentage in the answer must appear in the tool results / retrieved text (or the user's
own question). Ungrounded numbers are the usual shape of a hallucination here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

NUM = re.compile(r"(?<![\w.])\$?\d[\d,]*(?:\.\d+)?%?")
MIN_LEAK_WORDS = 7


def numbers(text: str) -> set[str]:
    out = set()
    for m in NUM.findall(text):
        n = m.replace("$", "").replace(",", "").replace("%", "")
        if "." in n:
            n = n.rstrip("0").rstrip(".")
        out.add(n)
    return out


@dataclass
class OutputCheck:
    ok: bool = True
    flags: list[str] = field(default_factory=list)
    ungrounded: list[str] = field(default_factory=list)


def _shingles(text: str, n: int) -> set[tuple]:
    t = text.lower().split()
    return {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}


def check_output(answer: str, evidence: str, question: str, mode: str,
                 manager_shingles: set[tuple] | None = None,
                 system_prompt: str = "") -> OutputCheck:
    res = OutputCheck()
    allowed = numbers(evidence) | numbers(question)
    bad = sorted(n for n in numbers(answer) if n not in allowed)
    if bad:
        res.ungrounded = bad
        res.flags.append("ungrounded_numbers")
    if mode != "manager" and manager_shingles and _shingles(answer, MIN_LEAK_WORDS) & manager_shingles:
        res.flags.append("manager_leak")
    if system_prompt and _shingles(answer, 8) & _shingles(system_prompt, 8):
        res.flags.append("system_prompt_leak")
    res.ok = not res.flags
    return res
