"""Synthetic complaint -> ticket dataset (for fine-tuning and evaluating the extraction task).

Fully synthetic: complaint sentences are composed from templates with varied phrasing, and the gold
label follows from the template. It is a *format and schema* teaching set, not evidence of how a
model would do on real customer text; say so when you quote numbers from it.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from .catalog import MENU

ITEMS = [m["name"] for m in MENU]

# category -> (templates, severity, needs_human)
TEMPLATES = {
    "food_quality": (["My {item} was stone cold when it arrived{at}.",
                      "The {item} tasted stale and soggy{at}.",
                      "Honestly the {item} looked undercooked{at}, really disappointing."],
                     "medium", True),
    "wrong_or_missing_item": (["You forgot my {item}{at}, the bag was missing it.",
                               "I got the wrong order{at}: I asked for a {item} and it was not there.",
                               "Missing item again{at}, no {item} in the bag."],
                              "medium", True),
    "wait_time": (["Waited ages for a {item}{at}, the queue barely moved.",
                   "Way too slow{at}. It took forever to get my {item}.",
                   "The drive-thru wait{at} was ridiculous just for a {item}."],
                  "low", False),
    "app_or_payment": (["The app charged my card twice for a {item}{at}.",
                        "Payment failed in the app but I was still charged for the {item}{at}.",
                        "App keeps crashing when I try to pay for a {item}{at}."],
                       "medium", True),
    "allergen_safety": (["I told them about my nut allergy and the {item} still had traces{at}.",
                         "My daughter had an allergic reaction after the {item}{at}.",
                         "Nobody checked my gluten allergy when I ordered the {item}{at}."],
                        "high", True),
    "staff_behaviour": (["The staff were rude when I asked about my {item}{at}.",
                         "Employee ignored me and shouted when I complained about the {item}{at}."],
                        "low", False),
    "other": (["Just wanted to say the {item} idea for a new flavour would be great{at}.",
               "Is there any chance you could extend opening hours{at}? I love the {item}."],
              "low", False),
}


def make_example(rng: random.Random) -> dict:
    cat = rng.choice(list(TEMPLATES))
    templates, severity, human = TEMPLATES[cat]
    item = rng.choice(ITEMS)
    store = rng.choice([None, None, rng.randint(1, 20)])
    at = f" at store {store}" if store else ""
    text = rng.choice(templates).format(item=item.lower(), at=at)
    return {"text": text, "ticket": {"category": cat, "severity": severity, "store_id": store,
                                     "items": [item.lower()], "needs_human": human}}


def build_dataset(n: int = 600, seed: int = 7) -> tuple[list[dict], list[dict]]:
    """Return (train, val) with an 80/20 split; texts are de-duplicated across the split."""
    rng = random.Random(seed)
    seen, rows = set(), []
    while len(rows) < n:
        ex = make_example(rng)
        if ex["text"] not in seen:
            seen.add(ex["text"])
            rows.append(ex)
    cut = int(n * 0.8)
    return rows[:cut], rows[cut:]


def to_chat(example: dict, instruction: str) -> dict:
    """Chat-format record for SFT (system / user / assistant)."""
    return {"messages": [
        {"role": "system", "content": instruction},
        {"role": "user", "content": example["text"]},
        {"role": "assistant", "content": json.dumps(example["ticket"], ensure_ascii=False)}]}


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
