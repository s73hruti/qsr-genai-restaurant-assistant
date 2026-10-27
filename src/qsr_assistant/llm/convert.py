"""Convert neutral messages/tools to provider formats and parse provider responses.

Pure functions (no SDK imports) so they can be unit-tested with fake response objects.
"""
from __future__ import annotations

import json

from .types import LLMResponse, Message, ToolCall, Usage

# ---------------------------------------------------------------- Anthropic


def to_anthropic_messages(messages: list[Message]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        if m.role == "user":
            out.append({"role": "user", "content": m.content})
        elif m.role == "assistant":
            blocks = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            for tc in m.tool_calls:
                blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name,
                               "input": tc.arguments})
            out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})
        elif m.role == "tool":
            block = {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)  # parallel tool results share one user turn
            else:
                out.append({"role": "user", "content": [block]})
    return out


def to_anthropic_tools(tools: list[dict]) -> list[dict]:
    return [{"name": t["name"], "description": t["description"],
             "input_schema": t["input_schema"]} for t in tools]


def parse_anthropic(resp) -> LLMResponse:
    text, calls = [], []
    for b in resp.content:
        kind = getattr(b, "type", "")
        if kind == "text":
            text.append(b.text)
        elif kind == "tool_use":
            calls.append(ToolCall(b.id, b.name, dict(b.input or {})))
    u = getattr(resp, "usage", None)
    usage = Usage(getattr(u, "input_tokens", 0) or 0, getattr(u, "output_tokens", 0) or 0)
    return LLMResponse("".join(text), calls, usage, getattr(resp, "stop_reason", "") or "",
                       getattr(resp, "model", "") or "")


# ---------------------------------------------------------------- OpenAI


def to_openai_messages(system: str, messages: list[Message]) -> list[dict]:
    out: list[dict] = [{"role": "system", "content": system}]
    for m in messages:
        if m.role == "user":
            out.append({"role": "user", "content": m.content})
        elif m.role == "assistant":
            msg: dict = {"role": "assistant", "content": m.content or None}
            if m.tool_calls:
                msg["tool_calls"] = [{"id": tc.id, "type": "function",
                                      "function": {"name": tc.name,
                                                   "arguments": json.dumps(tc.arguments)}}
                                     for tc in m.tool_calls]
            out.append(msg)
        elif m.role == "tool":
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
    return out


def to_openai_tools(tools: list[dict]) -> list[dict]:
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                              "parameters": t["input_schema"]}} for t in tools]


def _loads(s: str) -> dict:
    try:
        v = json.loads(s or "{}")
        return v if isinstance(v, dict) else {}
    except (ValueError, TypeError):
        return {}


def parse_openai(resp) -> LLMResponse:
    choice = resp.choices[0]
    msg = choice.message
    calls = [ToolCall(tc.id, tc.function.name, _loads(tc.function.arguments))
             for tc in (getattr(msg, "tool_calls", None) or [])]
    u = getattr(resp, "usage", None)
    usage = Usage(getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0)
    return LLMResponse(msg.content or "", calls, usage, choice.finish_reason or "",
                       getattr(resp, "model", "") or "")


# ---------------------------------------------------------------- vision (image input)


def anthropic_image_content(b64: str, media_type: str, prompt: str) -> list[dict]:
    return [{"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
            {"type": "text", "text": prompt}]


def openai_image_content(b64: str, media_type: str, prompt: str) -> list[dict]:
    return [{"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}}]
