"""OfflineClient: a deterministic stand-in for an LLM that implements the same tool-calling protocol.

Policy: pick tools with the rule router, call `search_knowledge` for everything else, then compose
the answer extractively from tool results. It deliberately behaves like a naive model in one respect:
it requests manager tools even for customers, so tests prove the executor (not the model) enforces
permissions.
"""
from __future__ import annotations

import re
import uuid

from ..agent.router import REFUSAL, extract_entities, rule_route
from .clients import BaseClient
from .types import LLMResponse, Message, ToolCall, Usage

NO_INFO = "I don't have that information in the menu, offers, store or help articles."
_ARG_NAMES = {
    "forecast_demand": ("store_id", "days"),
    "plan_inventory": ("store_id", "days", "ingredient"),
    "wait_time": ("store_id", "hour", "queue_length"),
    "recommend": ("cart_items", "temp_c"),
    "dynamic_menu": ("temp_c", "hour"),
}


def _tokens(text: str) -> int:
    return max(1, len(text) // 4)


class OfflineClient(BaseClient):
    name = "offline"
    online = False
    model = "offline-rules"

    def chat(self, system, messages, tools=None, max_tokens=700) -> LLMResponse:
        last = messages[-1]
        usage = Usage(sum(_tokens(m.content) for m in messages) + _tokens(system), 0)
        if last.role == "tool":
            tool_msgs = []
            for m in reversed(messages):
                if m.role != "tool":
                    break
                tool_msgs.append(m)
            text = self._compose(list(reversed(tool_msgs)))
            usage.output_tokens = _tokens(text)
            return LLMResponse(text, [], usage, "end_turn", self.model)
        question = last.content
        if re.fullmatch(r"\s*(hi|hello|hey|thanks|thank you|good (morning|afternoon|evening))\W*",
                        question, re.I):
            return LLMResponse("Hi! I can help with the menu, offers, store hours and wait times.",
                               [], usage, "end_turn", self.model)
        calls = []
        for name in rule_route(question):
            ent = extract_entities(question)
            args = {k: ent[k] for k in _ARG_NAMES[name] if k in ent}
            calls.append(ToolCall(f"call_{uuid.uuid4().hex[:8]}", name, args))
        if not calls:
            calls.append(ToolCall(f"call_{uuid.uuid4().hex[:8]}", "search_knowledge",
                                  {"query": question}))
        usage.output_tokens = 20 * len(calls)
        return LLMResponse("", calls, usage, "tool_use", self.model)

    def stream(self, system, messages, tools=None, max_tokens=700):
        resp = self.chat(system, messages, tools, max_tokens)
        for word in re.findall(r"\S+\s*", resp.text):
            yield "text", word
        yield "final", resp

    def complete(self, system: str, user: str, max_tokens: int = 500) -> str:
        raise RuntimeError("OfflineClient cannot do free-form completion")

    @staticmethod
    def _compose(tool_msgs: list[Message]) -> str:
        parts: list[str] = []
        for m in tool_msgs:
            d = m.data or {}
            if d.get("denied"):
                parts.append(REFUSAL)
            elif m.name == "search_knowledge":
                hits = d.get("hits", [])
                if d.get("low_confidence") or not hits:
                    parts.append(NO_INFO)
                else:
                    top = hits[0]
                    body = top["text"].replace("\n", " ")
                    parts.append(f"{body} [{top['doc_id']}]")
            else:
                parts.append(m.content)
        seen, out = set(), []
        for p in parts:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return "\n\n".join(out) or NO_INFO
