"""MCP server: exposes the assistant's knowledge search and ML tools to any MCP client
(Claude Desktop, Claude Code, Cursor, MCP Inspector...).

    pip install -e ".[mcp]"
    python -m qsr_assistant.mcp_server.server                 # stdio, customer permissions
    MCP_MODE=manager python -m qsr_assistant.mcp_server.server   # also exposes manager tools
    MCP_TRANSPORT=streamable-http python -m qsr_assistant.mcp_server.server

Security: identity is fixed by the *server operator* through MCP_MODE, never by tool arguments, and
every call still goes through the same `execute_tool` executor as the chat agent, so the permission
checks, argument clamping and context sanitising are shared. Manager tools are not even registered
in customer mode.
"""
from __future__ import annotations

import json

from ..agent.tools import ToolContext, ToolResult, execute_tool
from ..config import env
from ..ml.registry import load_registry


def _render(res: ToolResult) -> str:
    return json.dumps({"ok": res.ok, "denied": res.denied, "summary": res.summary}, ensure_ascii=False)


def build_server(mode: str | None = None, registry=None):
    from mcp.server.fastmcp import FastMCP

    mode = "manager" if (mode or env("MCP_MODE", "customer")) == "manager" else "customer"
    reg = registry or load_registry()
    mcp = FastMCP("qsr-assistant")

    def run(name: str, args: dict, store_id: int | None = None) -> str:
        ctx = ToolContext(reg, mode, None, store_id)
        return _render(execute_tool(name, args, ctx))

    @mcp.tool()
    def search_knowledge(query: str, doc_type: str | None = None) -> str:
        """Search menu, prices, allergens, offers, store info, FAQs and policies
        (and operations manuals in manager mode). Write a standalone query."""
        return run("search_knowledge", {"query": query, "doc_type": doc_type})

    @mcp.tool()
    def wait_time(store_id: int = 1, hour: int | None = None, queue_length: int | None = None) -> str:
        """Predict the wait time at a store (ML model, includes a P90)."""
        return run("wait_time", {"store_id": store_id, "hour": hour, "queue_length": queue_length})

    @mcp.tool()
    def recommend(cart_items: list[str] | None = None, temp_c: float | None = None) -> str:
        """Recommend menu items, or cross-sell items to go with the given cart."""
        return run("recommend", {"cart_items": cart_items, "temp_c": temp_c})

    if mode == "manager":
        @mcp.tool()
        def forecast_demand(store_id: int = 1, days: int = 3) -> str:
            """Forecast daily order volume for a store (manager only)."""
            return run("forecast_demand", {"store_id": store_id, "days": days}, store_id)

        @mcp.tool()
        def plan_inventory(store_id: int = 1, days: int = 3, ingredient: str | None = None) -> str:
            """Recommend ingredient order quantities (manager only)."""
            return run("plan_inventory", {"store_id": store_id, "days": days,
                                          "ingredient": ingredient}, store_id)

    return mcp


def main() -> None:
    transport = env("MCP_TRANSPORT", "stdio")
    build_server().run(transport=transport)


if __name__ == "__main__":
    main()
