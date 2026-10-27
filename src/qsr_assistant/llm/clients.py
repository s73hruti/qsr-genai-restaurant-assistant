"""LLM clients with a uniform interface: chat(), stream(), complete().

* AnthropicClient / OpenAIClient: native tool calling (function calling), streaming.
* OfflineClient (llm/offline.py): deterministic stand-in that speaks the same protocol, so the whole
  agent loop runs and is testable without network access or API keys.
"""
from __future__ import annotations

from collections.abc import Iterator

from ..config import env
from .convert import (
    anthropic_image_content,
    openai_image_content,
    parse_anthropic,
    parse_openai,
    to_anthropic_messages,
    to_anthropic_tools,
    to_openai_messages,
    to_openai_tools,
)
from .types import LLMResponse, Message, ToolCall, Usage

DEFAULT_MODELS = {"anthropic": "claude-sonnet-5-5", "openai": "gpt-4o-mini"}
TIMEOUT_S = float(env("LLM_TIMEOUT", "30"))


class BaseClient:
    name = "base"
    online = False
    model = ""

    def chat(self, system: str, messages: list[Message], tools: list[dict] | None = None,
             max_tokens: int = 700) -> LLMResponse:
        raise NotImplementedError

    def stream(self, system, messages, tools=None, max_tokens=700) -> Iterator[tuple[str, object]]:
        """Yield ("text", delta) events, then one ("final", LLMResponse)."""
        resp = self.chat(system, messages, tools, max_tokens)
        if resp.text:
            yield "text", resp.text
        yield "final", resp

    def complete(self, system: str, user: str, max_tokens: int = 500) -> str:
        return self.chat(system, [Message("user", user)], None, max_tokens).text

    def describe_image(self, system: str, prompt: str, b64: str, media_type: str,
                       max_tokens: int = 300) -> str:
        """Vision call. Only online multimodal clients implement it."""
        raise NotImplementedError(f"{self.name} client has no vision support")


class AnthropicClient(BaseClient):
    name = "anthropic"
    online = True

    def __init__(self, model: str | None = None):
        import anthropic
        self.client = anthropic.Anthropic(timeout=TIMEOUT_S, max_retries=2)
        self.model = model or env("LLM_MODEL") or DEFAULT_MODELS["anthropic"]

    def _kwargs(self, system, messages, tools, max_tokens) -> dict:
        kw = dict(model=self.model, max_tokens=max_tokens, system=system,
                  messages=to_anthropic_messages(messages))
        if tools:
            kw["tools"] = to_anthropic_tools(tools)
        return kw

    def chat(self, system, messages, tools=None, max_tokens=700) -> LLMResponse:
        return parse_anthropic(self.client.messages.create(
            **self._kwargs(system, messages, tools, max_tokens)))

    def describe_image(self, system, prompt, b64, media_type, max_tokens=300) -> str:
        resp = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user",
                       "content": anthropic_image_content(b64, media_type, prompt)}])
        return parse_anthropic(resp).text

    def stream(self, system, messages, tools=None, max_tokens=700):
        with self.client.messages.stream(**self._kwargs(system, messages, tools, max_tokens)) as s:
            for delta in s.text_stream:
                yield "text", delta
            final = s.get_final_message()
        yield "final", parse_anthropic(final)


class OpenAIClient(BaseClient):
    name = "openai"
    online = True

    def __init__(self, model: str | None = None):
        from openai import OpenAI
        self.client = OpenAI(timeout=TIMEOUT_S, max_retries=2)
        self.model = model or env("LLM_MODEL") or DEFAULT_MODELS["openai"]

    def _kwargs(self, system, messages, tools, max_tokens) -> dict:
        kw = dict(model=self.model, max_completion_tokens=max_tokens,
                  messages=to_openai_messages(system, messages))
        if tools:
            kw["tools"] = to_openai_tools(tools)
        return kw

    def chat(self, system, messages, tools=None, max_tokens=700) -> LLMResponse:
        return parse_openai(self.client.chat.completions.create(
            **self._kwargs(system, messages, tools, max_tokens)))

    def describe_image(self, system, prompt, b64, media_type, max_tokens=300) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, max_completion_tokens=max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": openai_image_content(b64, media_type, prompt)}])
        return parse_openai(resp).text

    def stream(self, system, messages, tools=None, max_tokens=700):
        stream = self.client.chat.completions.create(
            **self._kwargs(system, messages, tools, max_tokens), stream=True,
            stream_options={"include_usage": True})
        text, acc, usage, finish = [], {}, Usage(), ""
        for chunk in stream:
            if getattr(chunk, "usage", None):
                usage = Usage(chunk.usage.prompt_tokens or 0, chunk.usage.completion_tokens or 0)
            if not chunk.choices:
                continue
            ch = chunk.choices[0]
            finish = ch.finish_reason or finish
            d = ch.delta
            if getattr(d, "content", None):
                text.append(d.content)
                yield "text", d.content
            for tc in getattr(d, "tool_calls", None) or []:
                slot = acc.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                slot["id"] = tc.id or slot["id"]
                if tc.function:
                    slot["name"] = tc.function.name or slot["name"]
                    slot["args"] += tc.function.arguments or ""
        from .convert import _loads
        calls = [ToolCall(s["id"], s["name"], _loads(s["args"])) for _, s in sorted(acc.items())]
        yield "final", LLMResponse("".join(text), calls, usage, finish, self.model)


def get_llm(backend: str | None = None) -> BaseClient:
    backend = (backend or env("LLM_BACKEND", "auto")).lower()
    if backend == "auto":
        backend = ("anthropic" if env("ANTHROPIC_API_KEY")
                   else "openai" if env("OPENAI_API_KEY") else "offline")
    try:
        if backend == "anthropic":
            return AnthropicClient()
        if backend == "openai":
            return OpenAIClient()
    except ImportError:  # SDK not installed: degrade instead of crashing at start-up
        pass
    from .offline import OfflineClient
    return OfflineClient()
