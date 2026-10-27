"""Input guardrails: length limit, prompt-injection screen, PII redaction."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .patterns import find_injection

MAX_CHARS = 1000
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"(?<!\d)(?:\+?\d[\s-]?){10,13}(?!\d)")
CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")

BLOCK_MESSAGE = ("I can't help with that request. I can answer questions about our menu, offers, "
                 "stores and wait times - what would you like to know?")


def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


@dataclass
class InputCheck:
    text: str
    blocked: bool = False
    reason: str = ""
    flags: list[str] = field(default_factory=list)


def redact_pii(text: str) -> tuple[str, list[str]]:
    flags = []

    def card_sub(m):
        raw = m.group()
        digits = re.sub(r"\D", "", raw)
        if 13 <= len(digits) <= 19 and _luhn(digits):
            flags.append("card")
            return "[REDACTED_CARD]" + (" " if raw.endswith((" ", "-")) else "")
        return raw

    text = CARD.sub(card_sub, text)
    if EMAIL.search(text):
        flags.append("email")
        text = EMAIL.sub("[REDACTED_EMAIL]", text)
    if PHONE.search(text):
        flags.append("phone")
        text = PHONE.sub("[REDACTED_PHONE]", text)
    return text, flags


def check_input(text: str) -> InputCheck:
    if len(text) > MAX_CHARS:
        return InputCheck(text[:MAX_CHARS], True, "too_long", ["too_long"])
    hits = find_injection(text)
    clean, pii = redact_pii(text)
    flags = [f"pii:{p}" for p in pii] + [f"injection:{h}" for h in hits]
    if hits:
        return InputCheck(clean, True, hits[0], flags)
    return InputCheck(clean, False, "", flags)
