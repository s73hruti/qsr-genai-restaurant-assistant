"""Output guardrails: numeric grounding, manager-info leak detection, system-prompt leak.

Numeric grounding is the cheap, high-value check for a restaurant assistant: every price, time,
count or percentage in the answer must be traceable to the tool results / retrieved text (or the
user's own question). Ungrounded numbers are the usual shape of a hallucination here.

"Traceable" is deliberately a little wider than "appears verbatim", because a literal match flags
correct answers (41.7% of legitimate paraphrases in `eval/data/guard_cases.json` with the literal
version). A number is accepted if it is
  * in the evidence or the question,
  * an evidence number rounded (4.8 -> 5),
  * a difference or quotient of two quantities on the same evidence line (417 - 390, 1197 / 3 days),
    a sum / difference of two prices from anywhere in the evidence ($5.99 + $2.49), or the product
    of an evidence number and a number the user asked about (3 coffees x $2.50). Percentages, clock
    times, store numbers, dates and years are never combined (4.8 min + 7.2 min is not a claim
    anyone makes, and allowing it let an invented "12 minutes" through),
  * a count of what the evidence lists (3 picks, 2 offers),
  * the 12h/24h form of a clock time in the evidence,
and citation tags ([doc_id]) and list numbering (1. 2. 3.) are not counted as claims. Small derived
integers (< 10) are NOT accepted, so a made-up "3 hours" is still caught. Measure it with
`python -m qsr_assistant.eval.guard_eval`. Known gaps: number words ("twenty minutes") and codes
with digits glued to letters ("SAVE20") are invisible to a digit-based check.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations

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


# ---------------------------------------------------------------------------- grounding helpers
_CITE = re.compile(r"\[[^\]]*\]")
_LIST_MARK = re.compile(r"(?m)^\s*\d+[.)]\s+")
_CLOCK_12 = re.compile(r"(?<![\w.])(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", re.I)
_CLOCK_24 = re.compile(r"(?<![\w.])([01]?\d|2[0-3]):([0-5]\d)(?!\d)")
_MAX_LINE_NUMS = 25
_MAX_PRICES = 30
_MAX_COUNT = 12


def _norm(n: str) -> str:
    """Canonical text form: no leading zeros, no trailing decimal zeros ('06' -> '6', '7.50' -> '7.5')."""
    if "." in n:
        ip, fp = n.split(".", 1)
        fp = fp.rstrip("0")
        return (ip.lstrip("0") or "0") + ("." + fp if fp else "")
    return n.lstrip("0") or "0"


def _fmt(v: float) -> str:
    return _norm(f"{v:.6f}".rstrip("0").rstrip(".") if v != int(v) else str(int(v)))


def _nums(text: str) -> set[str]:
    return {_norm(n) for n in numbers(text)}


def _values(text: str) -> list[float]:
    out: list[float] = []
    for n in sorted(_nums(text)):
        try:
            out.append(float(n))
        except ValueError:
            pass
    return out


_NON_QUANTITY = re.compile(
    r"\d[\d,.]*\s*%"                                   # percentages
    r"|\d{1,2}(?::\d{2})?\s*(?:am|pm)\b|\b\d{1,2}:\d{2}\b"   # clock times
    r"|\bstores?\s*#?\s*\d+|#\s*\d+"                    # store / ticket numbers
    r"|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*"  # dates
    r"|\b(?:19|20)\d{2}\b", re.I)                       # years


def _add_derived(into: set[str], v: float) -> None:
    """Accept a derived value exactly (2 dp); round to an integer only for large figures. Small
    derived integers are never accepted: that is where hallucinated counts hide."""
    r2 = round(v, 2)
    if not (r2 == int(r2) and 0 <= r2 < 10):
        into.add(_fmt(r2))
    if v >= 50:
        into.add(_fmt(round(v)))


def _quantities(line: str) -> list[float]:
    return _values(_NON_QUANTITY.sub(" ", line))[:_MAX_LINE_NUMS]


def grounded_numbers(evidence: str, question: str = "") -> set[str]:
    """Every number the answer may legitimately contain (see module docstring)."""
    ok = _nums(evidence) | _nums(question)
    ev_vals = _values(evidence)

    for v in ev_vals:  # rounding of evidence decimals
        if v != int(v):
            for r in (round(v), int(v), int(v) + 1, round(v, 1)):
                ok.add(_fmt(r))

    lines = [ln for ln in evidence.splitlines() if ln.strip()]
    for ln in lines:  # same-line quantities: differences and quotients ("27 higher", "399 a day")
        for a, b in combinations(_quantities(ln), 2):
            _add_derived(ok, abs(a - b))
            if a and b:
                _add_derived(ok, a / b)
                _add_derived(ok, b / a)
    prices = sorted({float(_norm(m.replace(",", ""))) for ln in lines
                     for m in re.findall(r"(?<![\w.])\$(\d[\d,]*(?:\.\d+)?)", ln)})[:_MAX_PRICES]
    for a, b in combinations(prices, 2):  # prices from different documents may be added up
        _add_derived(ok, a + b)
        _add_derived(ok, abs(a - b))

    q_qty = _quantities(question)  # "store 2" / "6pm" in a question are not quantities to multiply
    for e in [v for ln in lines for v in _quantities(ln)][:60]:  # "3 coffees at $2.50"
        for q in q_qty:
            r2 = round(e * q, 2)
            if r2 > 0:
                ok.add(_fmt(r2))

    counts = {len(lines)} | {len(re.findall(r"\[[^\]]+\]", evidence))}
    counts |= {len([x for x in ln.split(";") if x.strip()]) for ln in lines}
    ok |= {str(c) for c in counts if 1 <= c <= _MAX_COUNT}

    for m in _CLOCK_12.finditer(evidence):  # 11pm <-> 23:00
        h, mins, ap = int(m.group(1)), m.group(2), m.group(3).lower()
        ok |= {str(h % 12 + (12 if ap == "pm" else 0)), str(h), "0" if not mins else _norm(mins)}
    for m in _CLOCK_24.finditer(evidence):  # 17:00 <-> 5pm
        h = int(m.group(1))
        ok |= {str(h), str(h % 12 or 12), _norm(m.group(2))}
    return ok


def ungrounded_numbers(answer: str, evidence: str, question: str = "") -> list[str]:
    claims = _LIST_MARK.sub("", _CITE.sub(" ", answer))
    allowed = grounded_numbers(evidence, question)
    return sorted(n for n in _nums(claims) if n not in allowed)


@dataclass
class OutputCheck:
    ok: bool = True
    flags: list[str] = field(default_factory=list)
    ungrounded: list[str] = field(default_factory=list)


def _shingles(text: str, n: int) -> set[tuple]:
    t = text.lower().split()
    return {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}


def _has_unbalanced_parentheses(text: str) -> bool:
    """Detect unmatched parentheses outside square-bracketed citation tags."""
    depth = 0
    in_brackets = False

    for char in text:
        if char == "[":
            in_brackets = True
        elif char == "]":
            in_brackets = False
        elif not in_brackets and char == "(":
            depth += 1
        elif not in_brackets and char == ")":
            depth -= 1
            if depth < 0:
                return True

    return depth != 0


def check_output(answer: str, evidence: str, question: str, mode: str,
                 manager_shingles: set[tuple] | None = None,
                 system_prompt: str = "") -> OutputCheck:
    res = OutputCheck()
    if _has_unbalanced_parentheses(answer):
        res.flags.append("incomplete_formatting")
    bad = ungrounded_numbers(answer, evidence, question)
    if bad:
        res.ungrounded = bad
        res.flags.append("ungrounded_numbers")
    if mode != "manager" and manager_shingles and _shingles(answer, MIN_LEAK_WORDS) & manager_shingles:
        res.flags.append("manager_leak")
    if system_prompt and _shingles(answer, 8) & _shingles(system_prompt, 8):
        res.flags.append("system_prompt_leak")
    res.ok = not res.flags
    return res
