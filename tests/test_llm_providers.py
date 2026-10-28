import json
from types import SimpleNamespace as NS

from qsr_assistant.llm import AnthropicClient, Message, OpenAIClient, ToolCall, Usage, get_llm
from qsr_assistant.llm.convert import (
    parse_anthropic,
    parse_openai,
    to_anthropic_messages,
    to_anthropic_tools,
    to_openai_messages,
    to_openai_tools,
)
from qsr_assistant.llm.offline import OfflineClient
import qsr_assistant.llm.clients as llm_clients
from qsr_assistant.llm.pricing import estimate_cost

CONVO = [
    Message("user", "wait at store 2 and a recommendation?"),
    Message("assistant", "Checking.", [ToolCall("t1", "wait_time", {"store_id": 2}),
                                       ToolCall("t2", "recommend", {})]),
    Message("tool", "wait is 5 min", tool_call_id="t1", name="wait_time"),
    Message("tool", "try fries", tool_call_id="t2", name="recommend"),
]
TOOLS = [{"name": "wait_time", "description": "d",
          "input_schema": {"type": "object", "properties": {}, "required": []}}]


def test_anthropic_message_conversion_merges_parallel_tool_results():
    out = to_anthropic_messages(CONVO)
    assert out[1]["content"][1]["type"] == "tool_use" and out[1]["content"][1]["id"] == "t1"
    assert out[2]["role"] == "user" and len(out[2]["content"]) == 2
    assert out[2]["content"][0] == {"type": "tool_result", "tool_use_id": "t1", "content": "wait is 5 min"}
    assert to_anthropic_tools(TOOLS)[0]["input_schema"]["type"] == "object"


def test_openai_message_conversion():
    out = to_openai_messages("SYS", CONVO)
    assert out[0] == {"role": "system", "content": "SYS"}
    assert json.loads(out[2]["tool_calls"][0]["function"]["arguments"]) == {"store_id": 2}
    assert out[3]["role"] == "tool" and out[3]["tool_call_id"] == "t1"
    assert to_openai_tools(TOOLS)[0]["function"]["parameters"]["type"] == "object"


def test_parse_anthropic_response():
    resp = NS(content=[NS(type="text", text="Hi"), NS(type="tool_use", id="x", name="wait_time",
                                                      input={"store_id": 3})],
              usage=NS(input_tokens=11, output_tokens=7), stop_reason="tool_use", model="m")
    r = parse_anthropic(resp)
    assert r.text == "Hi" and r.tool_calls[0].arguments == {"store_id": 3}
    assert (r.usage.input_tokens, r.usage.output_tokens) == (11, 7)


def test_parse_openai_response_with_bad_json_arguments():
    msg = NS(content=None, tool_calls=[NS(id="a", function=NS(name="recommend", arguments="{not json"))])
    resp = NS(choices=[NS(message=msg, finish_reason="tool_calls")],
              usage=NS(prompt_tokens=5, completion_tokens=2), model="m")
    r = parse_openai(resp)
    assert r.tool_calls[0].arguments == {} and r.usage.input_tokens == 5


def test_anthropic_stream_with_fake_sdk():
    final = NS(content=[NS(type="text", text="Hello there")], usage=NS(input_tokens=3, output_tokens=2),
               stop_reason="end_turn", model="m")

    class Stream:
        text_stream = ["Hello", " there"]
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get_final_message(self): return final

    c = object.__new__(AnthropicClient)
    c.model, c.client = "m", NS(messages=NS(stream=lambda **kw: Stream()))
    events = list(c.stream("sys", [Message("user", "hi")], TOOLS))
    assert [e[1] for e in events[:2]] == ["Hello", " there"] and events[-1][1].text == "Hello there"


def test_openai_stream_accumulates_tool_call_deltas():
    def chunk(content=None, tcs=None, usage=None, finish=None, choices=True):
        delta = NS(content=content, tool_calls=tcs)
        return NS(choices=[NS(delta=delta, finish_reason=finish)] if choices else [], usage=usage)

    def tc(i, id=None, name=None, args=None):
        return NS(index=i, id=id, function=NS(name=name, arguments=args))

    chunks = [chunk(tcs=[tc(0, "c1", "wait_time", '{"store')]), chunk(tcs=[tc(0, None, None, '_id": 4}')]),
              chunk(finish="tool_calls"),
              chunk(choices=False, usage=NS(prompt_tokens=9, completion_tokens=4))]
    c = object.__new__(OpenAIClient)
    c.model, c.client = "m", NS(chat=NS(completions=NS(create=lambda **kw: iter(chunks))))
    final = list(c.stream("sys", [Message("user", "hi")], TOOLS))[-1][1]
    assert final.tool_calls[0].name == "wait_time" and final.tool_calls[0].arguments == {"store_id": 4}
    assert final.usage.input_tokens == 9


def test_factory_falls_back_to_offline_without_keys(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert isinstance(get_llm("auto"), OfflineClient)
    assert isinstance(get_llm("offline"), OfflineClient)


def test_cost_estimate_and_override(monkeypatch):
    u = Usage(1_000_000, 1_000_000)
    assert estimate_cost("claude-sonnet-x", u) == 18.0
    assert estimate_cost("unknown-model", u) == 0.0
    monkeypatch.setenv("LLM_PRICE_IN", "1")
    monkeypatch.setenv("LLM_PRICE_OUT", "2")
    assert estimate_cost("unknown-model", u) == 3.0


def test_factory_passes_model_override(monkeypatch):
    def fake_openai_client(model=None):
        return NS(name="openai", model=model, online=True)

    monkeypatch.setattr(llm_clients, "OpenAIClient", fake_openai_client)
    client = get_llm("openai", model="qwen2.5:1.5b")

    assert client.model == "qwen2.5:1.5b"
