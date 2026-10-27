import asyncio
import json

import pytest

pytest.importorskip("mcp")

from qsr_assistant.mcp_server.server import build_server  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


def test_manager_tools_are_not_registered_for_customers(registry):
    names = {t.name for t in _run(build_server("customer", registry).list_tools())}
    assert names == {"search_knowledge", "wait_time", "recommend"}


def test_manager_mode_adds_manager_tools(registry):
    names = {t.name for t in _run(build_server("manager", registry).list_tools())}
    assert {"forecast_demand", "plan_inventory"} <= names


def _call(server, name, args):
    out = _run(server.call_tool(name, args))
    blocks = out[0] if isinstance(out, tuple) else out
    return json.loads(blocks[0].text)


def test_search_goes_through_shared_executor(registry):
    r = _call(build_server("customer", registry), "search_knowledge",
              {"query": "how much is the large fries"})
    assert r["ok"] and "3.40" in r["summary"]


def test_customer_server_cannot_see_manager_docs(registry):
    r = _call(build_server("customer", registry), "search_knowledge",
              {"query": "refund approval authority limits crew shift lead"})
    assert "Crew can refund up to 5" not in r["summary"]
    m = _call(build_server("manager", registry), "search_knowledge",
              {"query": "refund approval authority limits crew shift lead"})
    assert "5 dollars" in m["summary"]
