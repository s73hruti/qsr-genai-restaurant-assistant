"""Agent tools: JSON-schema definitions, permission checks, argument validation and execution.

Security model (defence in depth):
  1. The model is only *shown* tools allowed for the caller's mode.
  2. The executor re-checks permissions, so a hallucinated or injected call is still denied.
  3. Identity (mode, user_id) and audience come from the authenticated session via `ToolContext`,
     never from model-supplied arguments. Unknown arguments are dropped.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable

from ..config import env
from ..data.catalog import ITEM_BY_ID, MENU
from ..guardrails.sanitize import sanitize_context
from ..ml.registry import Registry
from .weather import get_weather

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
# Floor for 'no relevant info'. Calibrated by `python -m qsr_assistant.eval.abstain` (dev split).
# Lexical confidence is a weak signal, so this is deliberately low: above it the LLM judges relevance.
ABSTAIN_THRESHOLD = float(env("ABSTAIN_THRESHOLD", "0.23"))
ITEM_BY_NAME = {m["name"].lower(): m["id"] for m in MENU}


@dataclass
class ToolContext:
    reg: Registry
    mode: str = "customer"
    user_id: int | None = None
    store_id: int | None = None
    history: list[str] = field(default_factory=list)  # prior user turns (for query rewriting)


@dataclass
class ToolResult:
    name: str
    summary: str
    data: dict = field(default_factory=dict)
    ok: bool = True
    denied: bool = False
    latency_ms: float = 0.0


@dataclass
class ToolSpec:
    name: str
    description: str
    properties: dict
    required: list[str]
    fn: Callable
    manager_only: bool = False

    def schema(self, ctx_mode: str, reg: Registry | None = None) -> dict:
        props = {k: dict(v) for k, v in self.properties.items()}
        if self.name == "search_knowledge" and reg is not None:
            props["doc_type"]["enum"] = reg.retriever.doc_types(ctx_mode)
        return {"name": self.name, "description": self.description,
                "input_schema": {"type": "object", "properties": props,
                                 "required": self.required}}


def _fmt_time(h: int) -> str:
    return f"{(h + 11) % 12 + 1}{'am' if h < 12 else 'pm'}"


def _store(ctx: ToolContext, a: dict) -> int:
    return int(a.get("store_id") or ctx.store_id or 1)


# ------------------------------------------------------------------ tool implementations
def search_knowledge(ctx: ToolContext, a: dict) -> ToolResult:
    ret = ctx.reg.retriever
    dt = [a["doc_type"]] if a.get("doc_type") else None
    hits = ret.search(a["query"], k=4, audience=ctx.mode, doc_types=dt, history=ctx.history)
    conf = max((h.confidence for h in hits), default=0.0)
    if not hits or conf < ABSTAIN_THRESHOLD:
        return ToolResult("search_knowledge",
                          "No sufficiently relevant information found in the knowledge base. "
                          "Tell the user you don't have that information.",
                          {"hits": [], "low_confidence": True, "confidence": round(conf, 3)})
    lines = [f"[{h.chunk.doc_id}] ({h.chunk.heading}) {sanitize_context(h.chunk.text)}"
             for h in hits]
    return ToolResult("search_knowledge", "\n".join(lines),
                      {"hits": [h.to_dict() for h in hits], "low_confidence": False,
                       "confidence": round(conf, 3)})


def forecast_demand(ctx: ToolContext, a: dict) -> ToolResult:
    sid, days = _store(ctx, a), max(1, min(int(a.get("days", 3)), 14))
    start = date.today() + timedelta(days=1)
    fc = ctx.reg.forecaster.predict(sid, start, days)
    peak = fc.loc[fc.forecast.idxmax()]
    lines = ", ".join(f"{d:%a %d %b}: {int(v)}" for d, v in zip(fc.date, fc.forecast))
    return ToolResult("forecast_demand",
                      f"Store {sid} demand forecast for the next {days} days: {lines}. "
                      f"Total {int(fc.forecast.sum())} orders; busiest day {peak.date:%A} "
                      f"(~{int(peak.forecast)}).",
                      {"store_id": sid, "forecast": [{"date": str(d.date()), "orders": int(v)}
                                                     for d, v in zip(fc.date, fc.forecast)]})


def plan_inventory(ctx: ToolContext, a: dict) -> ToolResult:
    sid, days = _store(ctx, a), int(a.get("days", 3))
    plan = ctx.reg.forecaster.inventory_plan(sid, date.today() + timedelta(days=1), days)
    ing = (a.get("ingredient") or "").lower()
    if ing:
        sub = plan[plan.ingredient.str.contains(ing, case=False)]
        plan_view = sub if not sub.empty else plan
    else:
        plan_view = plan
    top = plan_view.sort_values("recommended_order", ascending=False).head(6)
    lines = "; ".join(f"{r.ingredient} {int(r.recommended_order)} {r.unit}" for r in top.itertuples())
    return ToolResult("plan_inventory",
                      f"Store {sid} ingredient order for the next {days} day(s) "
                      f"(forecast need + 95% safety stock): {lines}.",
                      {"store_id": sid, "days": days, "plan": plan.to_dict("records")})


def wait_time(ctx: ToolContext, a: dict) -> ToolResult:
    from datetime import datetime
    now = datetime.now()
    sid = _store(ctx, a)
    hour = int(a.get("hour", now.hour))
    r = ctx.reg.waittime.predict(sid, hour, now.weekday(), a.get("queue_length"),
                                 items=int(a.get("items", 2)))
    note = "typical queue" if r["queue_estimated"] else "current queue"
    return ToolResult("wait_time",
                      f"Store {sid} at {_fmt_time(hour)} on {DAYS[now.weekday()]}: expected wait "
                      f"about {r['wait_min']} min (90% of orders under {r['p90_min']} min), based "
                      f"on a {note} of {r['queue_len']} vehicles/orders and {r['staff']} staff.", r)


def _cart_ids(a: dict) -> list[int]:
    out = []
    for name in a.get("cart_items") or []:
        iid = ITEM_BY_NAME.get(str(name).lower())
        if iid:
            out.append(iid)
    return out


def recommend(ctx: ToolContext, a: dict) -> ToolResult:
    from datetime import datetime
    w = get_weather()
    temp = float(a["temp_c"]) if a.get("temp_c") is not None else w["temp_c"]
    hour = datetime.now().hour
    cart = _cart_ids(a)
    if cart:
        recs = ctx.reg.recommender.cross_sell(cart, hour, temp, k=3)
        head = "To go with " + ", ".join(ITEM_BY_ID[i]["name"] for i in cart)
    else:
        recs = ctx.reg.recommender.recommend(ctx.user_id, hour, temp, k=3)
        head = "Picks for you" if ctx.user_id else "Popular picks right now"
    lines = "; ".join(f"{r['name']} (${r['price']:.2f}, {r['reason']})" for r in recs)
    return ToolResult("recommend", f"{head}: {lines}.", {"recommendations": recs, "temp_c": temp})


def dynamic_menu(ctx: ToolContext, a: dict) -> ToolResult:
    from datetime import datetime
    w = get_weather()
    given = a.get("temp_c") is not None
    temp = float(a["temp_c"]) if given else w["temp_c"]
    hour = int(a.get("hour", datetime.now().hour))
    recs = ctx.reg.recommender.recommend(None, hour, temp, k=4)
    lines = "; ".join(f"{r['name']} ({r['reason']})" for r in recs)
    src = "from your message" if given else w["source"]
    return ToolResult("dynamic_menu", f"Menu board for {_fmt_time(hour)} at {temp:.0f}°C ({src}): {lines}.",
                      {"recommendations": recs, "temp_c": temp})


INT = {"type": "integer"}
SPECS: dict[str, ToolSpec] = {s.name: s for s in [
    ToolSpec("search_knowledge",
             "Search the knowledge base: menu items and prices, allergens, calories, offers and "
             "terms, store hours and services, FAQs, policies and (for managers) operations "
             "manuals and training. Use for any factual question about policies, products or "
             "stores. Write a standalone search query.",
             {"query": {"type": "string", "description": "Standalone search query"},
              "doc_type": {"type": "string", "description": "Optional document type filter"}},
             ["query"], search_knowledge),
    ToolSpec("forecast_demand", "Forecast daily order volume for a store (ML model). Manager only.",
             {"store_id": {**INT, "description": "Store 1-20"},
              "days": {**INT, "description": "Days ahead, 1-14"}}, [], forecast_demand, True),
    ToolSpec("plan_inventory", "Recommend ingredient order quantities for a store using the demand "
             "forecast, recipe bill of materials and safety stock (ML model). Manager only.",
             {"store_id": {**INT, "description": "Store 1-20"},
              "days": {**INT, "description": "Days to cover, 1-14"},
              "ingredient": {"type": "string", "description": "Optional ingredient filter, e.g. beef"}},
             [], plan_inventory, True),
    ToolSpec("wait_time", "Predict the current or future wait time at a store (ML model with P90).",
             {"store_id": {**INT, "description": "Store 1-20"},
              "hour": {**INT, "description": "Hour of day 0-23"},
              "queue_length": {**INT, "description": "Vehicles/orders ahead, if known"},
              "items": {**INT, "description": "Items in the order"}}, [], wait_time),
    ToolSpec("recommend", "Recommend menu items, or cross-sell items to go with items already "
             "chosen, using the recommender (ML model).",
             {"cart_items": {"type": "array", "items": {"type": "string"},
                             "description": "Menu item names already chosen"},
              "temp_c": {"type": "number", "description": "Outside temperature in Celsius"}},
             [], recommend),
    ToolSpec("dynamic_menu", "Weather- and time-aware menu board picks.",
             {"temp_c": {"type": "number", "description": "Temperature in Celsius"},
              "hour": {**INT, "description": "Hour of day 0-23"}}, [], dynamic_menu),
]}


def tool_defs(mode: str, reg: Registry | None = None) -> list[dict]:
    """Tool schemas the model may see for this mode (manager tools are hidden from customers)."""
    return [s.schema(mode, reg) for s in SPECS.values() if mode == "manager" or not s.manager_only]


def _coerce(spec: ToolSpec, args: dict) -> dict:
    out = {}
    for key, prop in spec.properties.items():
        if key not in args or args[key] is None:
            continue
        v = args[key]
        try:
            if prop["type"] == "integer":
                v = int(float(v))
                if key == "store_id":
                    v = max(1, v)
                elif key == "days":
                    v = max(1, min(v, 14))
                elif key == "hour":
                    v = max(0, min(v, 23))
                else:
                    v = max(0, min(v, 100))
            elif prop["type"] == "number":
                v = float(v)
            elif prop["type"] == "string":
                v = str(v)[:300]
            elif prop["type"] == "array":
                v = [str(x)[:60] for x in list(v)[:10]]
        except (TypeError, ValueError):
            continue
        out[key] = v
    return out


def execute_tool(name: str, args: dict, ctx: ToolContext) -> ToolResult:
    t0 = time.perf_counter()
    spec = SPECS.get(name)
    if spec is None:
        return ToolResult(name, f"Unknown tool '{name}'.", ok=False)
    if spec.manager_only and ctx.mode != "manager":
        return ToolResult(name, "DENIED: this tool is only available to store managers.",
                          {"denied": True}, ok=False, denied=True)
    clean = _coerce(spec, args if isinstance(args, dict) else {})
    missing = [r for r in spec.required if r not in clean]
    if missing:
        return ToolResult(name, f"Missing required argument(s): {', '.join(missing)}.", ok=False)
    try:
        res = spec.fn(ctx, clean)
    except Exception as e:  # a failing tool must not take down the answer
        res = ToolResult(name, f"{name} is unavailable right now ({e}).", ok=False)
    res.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    return res
