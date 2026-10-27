"""Rule-based intent routing and entity extraction.

With a real LLM the model chooses tools through native function calling and this module is not on
the critical path. It powers the OfflineClient (a deterministic stand-in for an LLM, so the full
agent loop is testable without keys) and is the baseline that `make eval-llm` compares against.
"""
from __future__ import annotations

import re
from datetime import datetime

from ..data.catalog import MENU

MANAGER_TOOLS = {"forecast_demand", "plan_inventory"}

RULES: list[tuple[str, str]] = [
    ("plan_inventory", r"\b(inventory|stock|reorder|supply|ingredients?|how much (beef|chicken|potato|"
                       r"cheese|bun|milk|coffee)|ingredient orders?)\b(?!.*\b(deadline|cutoff|count)\b)"),
    ("forecast_demand", r"\b(demand|forecast|expected orders|how busy|footfall|predicted orders|"
                        r"sales (forecast|prediction))\b"),
    ("wait_time", r"\b(wait|how long (is|will|does|until|before|to)\b.*\b(wait|order|take|ready|pickup)|queue|"
                  r"prep(aration)? time|ready in|throughput)\b"),
    ("dynamic_menu", r"\b(weather|hot day|rainy|cold day|trending|today'?s (picks|specials)|"
                     r"menu board)\b"),
    ("recommend", r"\b(recommend|suggest|what should i (eat|order|get)|upsell|cross[- ]?sell|"
                  r"pair with|goes with|add to my order|i('| a)?m hungry)\b"),
]

REFUSAL = ("That information is only available to store managers. Switch to manager mode "
           "if you have access, or ask me about the menu, offers or wait times.")


def extract_entities(query: str, now: datetime | None = None) -> dict:
    """Return only the entities that are explicitly present in the text."""
    q = query.lower()
    e: dict = {}
    if m := re.search(r"store\s*#?\s*(\d+)", q):
        e["store_id"] = int(m.group(1))
    if m := re.search(r"(\d+)\s*days?", q):
        e["days"] = max(1, min(int(m.group(1)), 14))
    elif "week" in q:
        e["days"] = 7
    elif "tomorrow" in q or "today" in q:
        e["days"] = 1
    if m := re.search(r"\b(\d{1,2})\s*(am|pm)\b", q):
        e["hour"] = int(m.group(1)) % 12 + (12 if m.group(2) == "pm" else 0)
    if m := re.search(r"(\d{2})\s*°?\s*c\b", q):
        e["temp_c"] = float(m.group(1))
    elif re.search(r"\b(hot|heatwave|sweltering)\b", q):
        e["temp_c"] = 35.0
    elif re.search(r"\b(cold|chilly|winter)\b", q):
        e["temp_c"] = 14.0
    if m := re.search(r"queue (of|is|length)?\s*(\d+)", q):
        e["queue_length"] = int(m.group(2))
    cart = [it["name"] for it in MENU
            if it["name"].lower() in q or it["name"].lower().replace(" 6pc", "") in q]
    if cart:
        e["cart_items"] = cart
    for ing in ("beef", "chicken", "potato", "cheese", "milk", "coffee", "bun"):
        if re.search(rf"\b{ing}\b", q):
            e["ingredient"] = ing
            break
    return e


def rule_route(query: str) -> list[str]:
    q = query.lower()
    tools = [name for name, pattern in RULES if re.search(pattern, q)]
    if "plan_inventory" in tools and "forecast_demand" in tools:
        tools.remove("forecast_demand")  # an inventory question already includes the forecast
    return tools[:2]
