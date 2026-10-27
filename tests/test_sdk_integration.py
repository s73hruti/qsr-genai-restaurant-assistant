"""Run the real Anthropic / OpenAI SDKs against a local fake server that speaks their wire formats.

This verifies request construction, tool-result round trips, SSE streaming and response parsing
end to end, without API keys or network access. Skipped if the SDKs are not installed.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from qsr_assistant.agent.agent import Agent

ANSWER = "The Large Fries costs $3.40 [menu-large-fries]."
QUERY = {"query": "large fries price"}
requests_seen: list[dict] = []


def sse(handler, events, with_event_name=True):
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.end_headers()
    for name, data in events:
        if with_event_name:
            handler.wfile.write(f"event: {name}\n".encode())
        handler.wfile.write(f"data: {json.dumps(data) if not isinstance(data, str) else data}\n\n".encode())
    handler.wfile.flush()


def anthropic_events(tool: bool):
    usage = {"input_tokens": 12, "output_tokens": 1}
    msg = {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5",
           "content": [], "stop_reason": None, "stop_sequence": None, "usage": usage}
    ev = [("message_start", {"type": "message_start", "message": msg})]
    if tool:
        ev += [("content_block_start", {"type": "content_block_start", "index": 0,
                                        "content_block": {"type": "tool_use", "id": "toolu_1",
                                                          "name": "search_knowledge", "input": {}}}),
               ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                        "delta": {"type": "input_json_delta",
                                                  "partial_json": json.dumps(QUERY)}})]
        stop = "tool_use"
    else:
        ev += [("content_block_start", {"type": "content_block_start", "index": 0,
                                        "content_block": {"type": "text", "text": ""}}),
               ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                        "delta": {"type": "text_delta", "text": ANSWER[:20]}}),
               ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                        "delta": {"type": "text_delta", "text": ANSWER[20:]}})]
        stop = "end_turn"
    ev += [("content_block_stop", {"type": "content_block_stop", "index": 0}),
           ("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
                              "usage": {"output_tokens": 9}}),
           ("message_stop", {"type": "message_stop"})]
    return ev


def anthropic_json(tool: bool):
    content = ([{"type": "tool_use", "id": "toolu_1", "name": "search_knowledge", "input": QUERY}]
               if tool else [{"type": "text", "text": ANSWER}])
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5",
            "content": content, "stop_reason": "tool_use" if tool else "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 12, "output_tokens": 9}}


def openai_chunks(tool: bool):
    def ch(delta, finish=None):
        return {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "gpt-4o-mini",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if tool:
        args = json.dumps(QUERY)
        chunks = [ch({"role": "assistant", "tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                                           "function": {"name": "search_knowledge", "arguments": ""}}]}),
                  ch({"tool_calls": [{"index": 0, "function": {"arguments": args[:10]}}]}),
                  ch({"tool_calls": [{"index": 0, "function": {"arguments": args[10:]}}]}),
                  ch({}, "tool_calls")]
    else:
        chunks = [ch({"role": "assistant", "content": ANSWER[:20]}), ch({"content": ANSWER[20:]}), ch({}, "stop")]
    chunks.append({"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "gpt-4o-mini",
                   "choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 9, "total_tokens": 21}})
    return chunks


def openai_json(tool: bool):
    msg = ({"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "type": "function",
         "function": {"name": "search_knowledge", "arguments": json.dumps(QUERY)}}]}
           if tool else {"role": "assistant", "content": ANSWER})
    return {"id": "c1", "object": "chat.completion", "created": 1, "model": "gpt-4o-mini",
            "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if tool else "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 9, "total_tokens": 21}}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        requests_seen.append({"path": self.path, "body": body})
        msgs = body["messages"]
        last = msgs[-1]
        after_tool = (last.get("role") == "tool" or (isinstance(last.get("content"), list) and any(
            b.get("type") == "tool_result" for b in last["content"])))
        has_image = any(isinstance(m.get("content"), list) and any(
            b.get("type") in ("image", "image_url") for b in m["content"]) for m in msgs)
        tool = not after_tool and not has_image
        anth = self.path.endswith("/v1/messages")
        if body.get("stream"):
            if anth:
                return sse(self, anthropic_events(tool))
            events = [("", c) for c in openai_chunks(tool)] + [("", "[DONE]")]
            return sse(self, events, with_event_name=False)
        payload = json.dumps(anthropic_json(tool) if anth else openai_json(tool)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture(scope="module")
def server():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


@pytest.fixture()
def clients(server, monkeypatch):
    pytest.importorskip("anthropic")
    pytest.importorskip("openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", server)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", server + "/v1")
    requests_seen.clear()
    from qsr_assistant.llm import AnthropicClient, OpenAIClient
    return {"anthropic": AnthropicClient(), "openai": OpenAIClient()}


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
@pytest.mark.parametrize("streaming", [False, True])
def test_agent_round_trip_with_real_sdk(registry, clients, provider, streaming):
    agent = Agent(registry, clients[provider])
    if streaming:
        events = list(agent.stream("How much are large fries?", "customer"))
        resp = events[-1]["response"]
        assert "".join(e["text"] for e in events if e["type"] == "token") == ANSWER
    else:
        resp = agent.run("How much are large fries?", "customer")
    assert resp.answer == ANSWER and resp.guardrails["output"] == []
    assert [t["name"] for t in resp.tools] == ["search_knowledge"] and resp.steps == 2
    assert resp.usage["llm_calls"] == 2 and resp.usage["input_tokens"] == 24
    first, second = requests_seen
    names = ([t["name"] for t in first["body"]["tools"]] if provider == "anthropic"
             else [t["function"]["name"] for t in first["body"]["tools"]])
    assert "search_knowledge" in names and "forecast_demand" not in names  # hidden from customers
    if provider == "anthropic":
        assert second["body"]["messages"][-1]["content"][0]["type"] == "tool_result"
        assert second["body"]["messages"][-2]["content"][0]["type"] == "tool_use"
    else:
        assert second["body"]["messages"][-1]["role"] == "tool"
        assert second["body"]["messages"][-2]["tool_calls"][0]["function"]["name"] == "search_knowledge"


def test_sdk_error_degrades_to_offline(registry, monkeypatch):
    pytest.importorskip("anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:9")  # nothing listens here
    from qsr_assistant.llm import AnthropicClient
    c = AnthropicClient()
    c.client = c.client.with_options(max_retries=0, timeout=2)
    r = Agent(registry, c).run("When do refunds arrive?", "customer")
    assert "3 to 5 business days" in r.answer and r.llm_backend == "anthropic->offline"


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_vision_request_shape_with_real_sdk(clients, provider):
    """The image block reaches the provider in its native wire format and the text reply is parsed."""
    import base64
    png = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16).decode()
    from qsr_assistant.llm.vision import image_to_query
    q, info = image_to_query(clients[provider], "what is this?", png, "image/png")
    assert info["described"] is True and "costs $3.40" in q
    content = requests_seen[-1]["body"]["messages"][-1]["content"]
    if provider == "anthropic":
        img = next(b for b in content if b["type"] == "image")
        assert img["source"] == {"type": "base64", "media_type": "image/png", "data": png}
    else:
        img = next(b for b in content if b["type"] == "image_url")
        assert img["image_url"]["url"] == f"data:image/png;base64,{png}"
