"""The same assistant as a LangGraph state machine (researcher + critic roles).

    START -> input_guard -> agent <-> tools
                              |
                              v
                           critic -> (revise -> agent) | finalize -> END

* `agent`   : LLM step with tool schemas (the "researcher").
* `tools`   : executes tool calls through the shared executor (permissions, clamping, sanitising).
* `critic`  : verifies the draft against tool evidence (numeric grounding, manager-info and prompt
              leaks). On a violation it sends the draft back ONCE with feedback so the model can fix
              it; if it still fails, `finalize` falls back to an answer composed from tool results.
* Everything else (guardrails, tools, memory, pricing) is reused from the hand-rolled agent, so
  `tests/test_graph.py` can check parity between the two implementations.

Why keep both? The hand-rolled loop in agent.py has streaming, caching and tracing; this version
shows the same logic in the orchestration framework most teams use, and makes the critic loop an
explicit, inspectable graph edge. Install with: pip install -e ".[graph]"
"""
from __future__ import annotations

import re
import time
from typing import TypedDict

from ..guardrails import BLOCK_MESSAGE, check_input, check_output
from ..llm import BaseClient, Message, Usage, get_llm
from ..llm.offline import NO_INFO, OfflineClient
from ..llm.pricing import estimate_cost
from ..ml.registry import Registry, load_registry
from .agent import (
    MAX_STEPS,
    MAX_TOKENS,
    MAX_TOOL_CALLS_PER_STEP,
    ROLE,
    SYSTEM_PROMPT,
    AgentResponse,
)
from .memory import SessionMemory
from .router import REFUSAL
from .tools import ToolContext, ToolResult, execute_tool, tool_defs

MAX_REVISIONS = 1


class State(TypedDict, total=False):
    query: str
    mode: str
    user_id: int | None
    store_id: int | None
    messages: list
    system: str
    tools_schema: list
    tools_log: list
    blocked: list
    guard: dict
    draft: str
    answer: str
    pending_calls: list
    steps: int
    revisions: int
    input_tokens: int
    output_tokens: int
    cost: float
    llm_calls: int
    ctx_history: list
    short_circuit: bool
    verdict: str
    key: str | None


class GraphAgent:
    def __init__(self, registry: Registry | None = None, llm: BaseClient | None = None,
                 memory: SessionMemory | None = None):
        try:
            from langgraph.graph import END, START, StateGraph  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError('GraphAgent needs LangGraph: pip install -e ".[graph]"') from e
        self.llm = llm or get_llm()
        self.reg = registry or load_registry(llm=self.llm)
        self.reg.retriever.rewriter.llm = self.llm
        self.offline = OfflineClient()
        self.memory = memory or SessionMemory()
        self._shingles = self.reg.retriever.manager_shingles()
        self.graph = self._build()

    # ------------------------------------------------------------------ graph
    def _build(self):
        from langgraph.graph import END, START, StateGraph
        g = StateGraph(State)
        g.add_node("input_guard", self._input_guard)
        g.add_node("agent", self._agent)
        g.add_node("tools", self._tools)
        g.add_node("critic", self._critic)
        g.add_node("finalize", self._finalize)
        g.add_edge(START, "input_guard")
        g.add_conditional_edges("input_guard", lambda s: "end" if s.get("short_circuit") else "agent",
                                {"end": END, "agent": "agent"})
        g.add_conditional_edges("agent", lambda s: "tools" if s.get("pending_calls") else "critic",
                                {"tools": "tools", "critic": "critic"})
        g.add_edge("tools", "agent")
        g.add_conditional_edges("critic", lambda s: s["verdict"],
                                {"revise": "agent", "finalize": "finalize"})
        g.add_edge("finalize", END)
        return g.compile()

    # ------------------------------------------------------------------ nodes
    def _input_guard(self, s: State) -> dict:
        ic = check_input(s["query"])
        guard = {"input": ic.flags, "output": []}
        if ic.blocked:
            return {"guard": guard, "answer": BLOCK_MESSAGE, "short_circuit": True}
        mode = s["mode"]
        summary, history = self.memory.get(s["key"]) if "key" in s else ("", [])
        system = SYSTEM_PROMPT.format(
            role=ROLE[mode],
            summary=f"\nEarlier in this conversation: {summary}" if summary else "")
        return {"guard": guard, "query": ic.text, "system": system,
                "messages": history + [Message("user", ic.text)],
                "ctx_history": [m.content for m in history if m.role == "user"],
                "tools_schema": tool_defs(mode, self.reg)}

    def _agent(self, s: State) -> dict:
        client: BaseClient = self.llm
        msgs = list(s["messages"])
        step = s.get("steps", 0)
        if step >= MAX_STEPS:  # step budget spent: stop calling tools, go to the critic
            return {"pending_calls": [], "draft": s.get("draft", "")}
        try:
            resp = client.chat(s["system"], msgs, s["tools_schema"], MAX_TOKENS)
        except Exception:
            if not client.online:
                raise
            client = self.offline  # graceful degradation, same as the hand-rolled agent
            resp = client.chat(s["system"], msgs, s["tools_schema"], MAX_TOKENS)
        usage = Usage(resp.usage.input_tokens, resp.usage.output_tokens)
        upd = {"steps": step + 1, "llm_calls": s.get("llm_calls", 0) + 1,
               "input_tokens": s.get("input_tokens", 0) + usage.input_tokens,
               "output_tokens": s.get("output_tokens", 0) + usage.output_tokens,
               "cost": s.get("cost", 0.0) + estimate_cost(resp.model or client.model, resp.usage)}
        if resp.tool_calls:
            msgs.append(Message("assistant", resp.text, resp.tool_calls))
            upd.update(messages=msgs, pending_calls=list(resp.tool_calls))
        else:
            upd.update(pending_calls=[], draft=resp.text.strip())
        return upd

    def _tools(self, s: State) -> dict:
        ctx = ToolContext(self.reg, s["mode"], s.get("user_id"), s.get("store_id"),
                          s.get("ctx_history", []))
        msgs, log, blocked = list(s["messages"]), list(s.get("tools_log", [])), list(s.get("blocked", []))
        for i, tc in enumerate(s["pending_calls"]):
            if i >= MAX_TOOL_CALLS_PER_STEP:
                res = ToolResult(tc.name, "Too many tool calls in one step.", ok=False)
            else:
                res = execute_tool(tc.name, tc.arguments, ctx)
            if res.denied:
                blocked.append(tc.name)
            data = dict(res.data)
            if res.denied:
                data["denied"] = True
            log.append({"name": tc.name, "arguments": tc.arguments, "ok": res.ok,
                        "denied": res.denied, "summary": res.summary,
                        "latency_ms": res.latency_ms, "data": res.data})
            msgs.append(Message("tool", res.summary, tool_call_id=tc.id, name=tc.name, data=data))
        return {"messages": msgs, "tools_log": log, "blocked": blocked, "pending_calls": []}

    def _critic(self, s: State) -> dict:
        tool_msgs = [m for m in s["messages"] if m.role == "tool"]
        draft = s.get("draft") or (OfflineClient._compose(tool_msgs, query=s.get("query")) if tool_msgs else NO_INFO)
        evidence = "\n".join(m.content for m in tool_msgs)
        oc = check_output(draft, evidence, s["query"], s["mode"], self._shingles, s["system"])

        # Match the hand-rolled agent's document-citation validation.
        sources = [
            hit
            for record in s.get("tools_log", [])
            if record["name"] == "search_knowledge"
            for hit in (record.get("data") or {}).get("hits", [])
        ]
        valid_doc_ids = {
            str(hit["doc_id"]) for hit in sources if hit.get("doc_id")
        }
        citations = re.findall(r"\[([A-Za-z0-9_.:-]+)\]", draft)

        if sources and not citations and "missing_citation" not in oc.flags:
            oc.flags.append("missing_citation")
        if any(doc_id not in valid_doc_ids for doc_id in citations):
            if "invalid_citation" not in oc.flags:
                oc.flags.append("invalid_citation")
        if re.search(r"\(\s*doc_id\s*:", draft, re.IGNORECASE):
            if "invalid_citation_format" not in oc.flags:
                oc.flags.append("invalid_citation_format")

        guard = dict(s["guard"])
        guard["output"] = list(oc.flags)
        if (oc.flags and "system_prompt_leak" not in oc.flags
                and s.get("revisions", 0) < MAX_REVISIONS and self.llm.online):
            fb = ("Reviewer: your draft was rejected (" + ", ".join(oc.flags) + "). Rewrite it "
                  "using ONLY facts from the tool results above; do not add numbers that are not "
                  "in them.")
            msgs = list(s["messages"]) + [Message("assistant", draft), Message("user", fb)]
            return {"draft": draft, "guard": guard, "messages": msgs, "verdict": "revise",
                    "revisions": s.get("revisions", 0) + 1, "steps": 0}
        return {"draft": draft, "guard": guard, "verdict": "finalize"}

    def _finalize(self, s: State) -> dict:
        draft, guard = s["draft"], dict(s["guard"])
        tool_msgs = [m for m in s["messages"] if m.role == "tool"]
        answer = draft
        if guard["output"]:
            if "system_prompt_leak" in guard["output"]:
                answer = BLOCK_MESSAGE
            elif s.get("blocked") and "manager_leak" in guard["output"]:
                answer = REFUSAL
            else:
                answer = OfflineClient._compose(tool_msgs, query=s.get("query")) if tool_msgs else NO_INFO
            guard["output"].append("answer_replaced")
        return {"answer": answer, "guard": guard}

    # ------------------------------------------------------------------ public API
    def run(self, query: str, mode: str = "customer", user_id: int | None = None,
            store_id: int | None = None, session_id: str | None = None) -> AgentResponse:
        t0 = time.perf_counter()
        mode = "manager" if mode == "manager" else "customer"
        key = SessionMemory.key(mode, user_id, session_id)
        out = self.graph.invoke({"query": query, "mode": mode, "user_id": user_id,
                                 "store_id": store_id, "key": key, "tools_log": [], "blocked": [],
                                 "steps": 0, "revisions": 0})
        answer = out["answer"]
        if not out.get("short_circuit"):
            self.memory.add(key, out["query"], answer)
        tools_log = out.get("tools_log", [])
        sources = [h for r in tools_log if r["name"] == "search_knowledge"
                   for h in (r["data"] or {}).get("hits", [])]
        return AgentResponse(
            answer=answer, mode=mode, session_id=session_id, tools=tools_log, sources=sources,
            blocked=out.get("blocked", []), guardrails=out["guard"],
            usage={"input_tokens": out.get("input_tokens", 0),
                   "output_tokens": out.get("output_tokens", 0),
                   "cost_usd": round(out.get("cost", 0.0), 6), "llm_calls": out.get("llm_calls", 0)},
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            llm_backend=f"langgraph:{self.llm.name}", steps=out.get("steps", 0))

    def mermaid(self) -> str:
        return self.graph.get_graph().draw_mermaid()
