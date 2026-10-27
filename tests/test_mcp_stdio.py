"""End-to-end: spawn the MCP server as a subprocess and talk to it with the official MCP client."""
import asyncio
import json
import os
import sys

import pytest

pytest.importorskip("mcp")

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402


async def _session(mode):
    env = {**os.environ, "MCP_MODE": mode, "LLM_BACKEND": "offline"}
    params = StdioServerParameters(command=sys.executable,
                                   args=["-m", "qsr_assistant.mcp_server.server"], env=env)
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = {t.name for t in (await s.list_tools()).tools}
            res = await s.call_tool("search_knowledge", {"query": "how much is the large fries"})
            return tools, json.loads(res.content[0].text)


@pytest.mark.parametrize("mode,has_manager", [("customer", False), ("manager", True)])
def test_stdio_roundtrip(mode, has_manager):
    tools, result = asyncio.run(asyncio.wait_for(_session(mode), 120))
    assert ("forecast_demand" in tools) is has_manager
    assert result["ok"] and "3.40" in result["summary"]
