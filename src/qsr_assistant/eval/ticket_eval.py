"""Score ticket extractors: JSON/schema validity and field-level accuracy against gold labels."""
from __future__ import annotations

import json

from pydantic import ValidationError

from ..llm.structured import Ticket, parse_json_object, rule_extract_ticket


def score(preds: list[str | dict | None], gold: list[dict]) -> dict:
    """`preds` are raw model outputs (str), dicts, or None; compared field by field with gold."""
    n = len(gold)
    valid = cat = sev = store = human = items = exact = 0
    for p, g in zip(preds, gold):
        try:
            obj = p if isinstance(p, dict) else parse_json_object(p or "")
            t = Ticket.model_validate({"summary": obj.get("summary", "n/a"), **obj}).model_dump()
        except (ValueError, ValidationError, TypeError, AttributeError, json.JSONDecodeError):
            continue
        valid += 1
        ok = [t["category"] == g["category"], t["severity"] == g["severity"],
              t["store_id"] == g["store_id"], t["needs_human"] == g["needs_human"],
              set(t["items"]) == set(g["items"])]
        cat, sev, store, human, items = (cat + ok[0], sev + ok[1], store + ok[2],
                                         human + ok[3], items + ok[4])
        exact += all(ok)
    pct = lambda x: round(100 * x / n, 1)  # noqa: E731
    return {"n": n, "valid_json_%": pct(valid), "category_acc_%": pct(cat),
            "severity_acc_%": pct(sev), "store_id_acc_%": pct(store),
            "needs_human_acc_%": pct(human), "items_acc_%": pct(items), "exact_match_%": pct(exact)}


def rule_baseline(val: list[dict]) -> dict:
    preds = [rule_extract_ticket(v["text"]).model_dump() for v in val]
    return score(preds, [v["ticket"] for v in val])
