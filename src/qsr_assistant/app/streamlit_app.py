"""Streamlit UI: streaming chat (customer / manager) with a visible agent trace + dashboard."""
import uuid
from datetime import date, timedelta

import streamlit as st

from qsr_assistant.agent.agent import Agent

st.set_page_config(page_title="QSR Assistant", page_icon="🍔", layout="wide")


@st.cache_resource(show_spinner="Loading models and knowledge base...")
def get_agent() -> Agent:
    return Agent()


agent = get_agent()
reg = agent.reg
ret = reg.retriever

with st.sidebar:
    st.title("🍔 QSR Assistant")
    mode = st.radio("Mode", ["customer", "manager"], horizontal=True)
    user_id = st.number_input("Member ID (customer)", 1, 2500, 1)
    store_id = st.number_input("Store ID", 1, 20, 1)
    st.caption(f"LLM: **{agent.llm.name}** ({agent.llm.model}) · retrieval: **{ret.mode}** "
               f"+ {ret.embedder.name}{' + rerank' if ret.use_rerank else ''} · "
               f"{ret.n_docs} docs / {len(ret.chunks)} chunks")
    if st.button("New conversation"):
        st.session_state.pop("history", None)
        st.session_state["session_id"] = uuid.uuid4().hex
    st.divider()
    st.caption("Try:")
    examples = {
        "customer": ["How long is the wait right now?", "What should I order with a Double Cheese Burger?",
                     "Any deals on ice cream?", "It's 36°C, what's on the menu board?",
                     "Which burgers are vegetarian?", "And what about the allergens?",
                     "Does the Riverside store have a drive-thru?"],
        "manager": ["How much beef and potato should I order for 3 days?", "Forecast demand for this week",
                    "How long can fries be held?", "And chicken nuggets?",
                    "What's the refund approval limit for crew?"],
    }
    for ex in examples[mode]:
        if st.button(ex, width="stretch"):
            st.session_state["pending"] = ex

chat_tab, dash_tab, obs_tab = st.tabs(["💬 Chat", "📊 Manager dashboard", "🔍 Observability"])
session_id = st.session_state.setdefault("session_id", uuid.uuid4().hex)

with chat_tab:
    history = st.session_state.setdefault("history", [])
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("trace"):
                with st.expander("How I answered"):
                    st.json(msg["trace"])

    prompt = st.chat_input("Ask about the menu, offers, stores, wait times...") \
        or st.session_state.pop("pending", None)
    if prompt:
        history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            box, streamed, final = st.empty(), "", None
            status = st.status("Thinking...", expanded=False)
            for ev in agent.stream(prompt, mode, int(user_id), int(store_id), session_id):
                if ev["type"] == "token":
                    streamed += ev["text"]
                    box.markdown(streamed + "▌")
                elif ev["type"] == "tool_call":
                    status.update(label=f"Calling {ev['name']}...")
                elif ev["type"] == "final":
                    final = ev["response"]
            status.update(label="Done", state="complete")
            box.markdown(final.answer)  # authoritative answer (may differ if a guardrail replaced it)
            trace = {"tools": [{"name": t["name"], "args": t["arguments"], "ok": t["ok"],
                                "denied": t["denied"]} for t in final.tools],
                     "sources": [f"{s['doc_id']} ({s['heading']}) conf={s['confidence']}" for s in final.sources],
                     "guardrails": final.guardrails, "usage": final.usage,
                     "latency_ms": final.latency_ms, "backend": final.llm_backend,
                     "cached": final.cached, "trace_id": final.trace_id}
            with st.expander("How I answered"):
                st.json(trace)
        history.append({"role": "assistant", "content": final.answer, "trace": trace})

with dash_tab:
    if mode != "manager":
        st.info("Switch to **manager** mode in the sidebar to view store analytics.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader(f"Demand forecast - store {int(store_id)}")
            fc = reg.forecaster.predict(int(store_id), date.today() + timedelta(days=1), 14)
            st.line_chart(fc.set_index("date")["forecast"])
        with c2:
            st.subheader("Ingredient order (next 3 days)")
            inv = reg.forecaster.inventory_plan(int(store_id), date.today() + timedelta(days=1), 3)
            st.dataframe(inv, hide_index=True, width="stretch")
        st.subheader("Model metrics (hold-out)")
        st.json(reg.metrics)

with obs_tab:
    snap = agent.metrics.snapshot()
    a, b, c = st.columns(3)
    a.metric("Requests", snap["counts"].get("requests", 0))
    b.metric("Cache hits", snap["counts"].get("cache_hits", 0))
    c.metric("Est. cost (USD)", snap["estimated_cost_usd"])
    st.json(snap)
    st.subheader("Recent traces")
    for t in reversed(agent.tracer.last(5)):
        with st.expander(f"{t['name']} · {t['duration_ms']} ms · {t['attrs'].get('tools')}"):
            st.json(t)
