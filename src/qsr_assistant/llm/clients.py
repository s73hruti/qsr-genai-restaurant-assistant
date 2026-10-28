"""LLM clients with a uniform interface: chat(), stream(), complete().

* AnthropicClient / OpenAIClient: native tool calling (function calling), streaming.
* OfflineClient (llm/offline.py): deterministic stand-in that speaks the same protocol, so the whole
  agent loop runs and is testable without network access or API keys.
"""
from __future__ import annotations

import threading
import time
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


def role_setting(role: str, name: str, default: str = "") -> str:
    """Setting for the LLM under test (`LLM_*`) or the judge (`JUDGE_*`, falling back to `LLM_*`)."""
    role = role.upper()
    return env(f"{role}_{name}") or (env(f"LLM_{name}") if role != "LLM" else "") or default


class BaseClient:
    name = "base"
    online = False
    model = ""
    _min_interval_s = 0.0
    _last_call = 0.0
    _throttle_lock = threading.Lock()

    def _throttle(self) -> None:
        """Space calls out (LLM_MIN_INTERVAL_S) so free-tier requests-per-minute caps are not hit."""
        gap = self._min_interval_s
        if gap <= 0:
            return
        with BaseClient._throttle_lock:
            wait = BaseClient._last_call + gap - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            BaseClient._last_call = time.monotonic()

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

    def __init__(self, model: str | None = None, role: str = "LLM"):
        import anthropic
        self.client = anthropic.Anthropic(timeout=TIMEOUT_S, max_retries=2)
        self.model = model or role_setting(role, "MODEL") or DEFAULT_MODELS["anthropic"]
        self._min_interval_s = float(role_setting(role, "MIN_INTERVAL_S", "0") or 0)

    def _kwargs(self, system, messages, tools, max_tokens) -> dict:
        kw = dict(model=self.model, max_tokens=max_tokens, system=system,
                  messages=to_anthropic_messages(messages))
        if tools:
            kw["tools"] = to_anthropic_tools(tools)
        return kw

    def chat(self, system, messages, tools=None, max_tokens=700) -> LLMResponse:
        self._throttle()
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
    """OpenAI Chat Completions client, also used for any OpenAI-compatible endpoint.

    Point it at a free or local provider with `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` (and
    `JUDGE_*` for an independent judge). Compatibility switches for providers that differ:
      LLM_TOKEN_PARAM=max_tokens      (default max_completion_tokens)
      LLM_STREAM_USAGE=0              (do not send stream_options)
      LLM_MIN_INTERVAL_S=4            (minimum seconds between calls, for requests/minute caps)
      LLM_MAX_RETRIES=5               (SDK retries with backoff on 429/5xx; default 2)
    """
    name = "openai"
    online = True
    token_param = "max_completion_tokens"
    stream_usage = True

    def __init__(self, model: str | None = None, role: str = "LLM"):
        from openai import OpenAI
        base = role_setting(role, "BASE_URL") or (env("OPENAI_BASE_URL") if role.upper() == "LLM" else "") or None
        key = role_setting(role, "API_KEY") or env("OPENAI_API_KEY") or ("not-needed" if base else None)
        self.client = OpenAI(base_url=base, api_key=key, timeout=TIMEOUT_S,
                             max_retries=int(role_setting(role, "MAX_RETRIES", "2") or 2))
        self.model = model or role_setting(role, "MODEL") or DEFAULT_MODELS["openai"]
        self.base_url = base or "https://api.openai.com/v1"
        self.token_param = env("LLM_TOKEN_PARAM", "max_completion_tokens")
        self.stream_usage = env("LLM_STREAM_USAGE", "1") not in ("0", "false", "False")
        self._min_interval_s = float(role_setting(role, "MIN_INTERVAL_S", "0") or 0)

    def _kwargs(self, system, messages, tools, max_tokens) -> dict:
        kw = {"model": self.model, self.token_param: max_tokens,
              "messages": to_openai_messages(system, messages)}
        if tools:
            kw["tools"] = to_openai_tools(tools)
        return kw

    def _stream_kwargs(self, system, messages, tools, max_tokens) -> dict:
        kw = self._kwargs(system, messages, tools, max_tokens)
        kw["stream"] = True
        if self.stream_usage:
            kw["stream_options"] = {"include_usage": True}
        return kw

    def chat(self, system, messages, tools=None, max_tokens=700) -> LLMResponse:
        self._throttle()
        return parse_openai(self.client.chat.completions.create(
            **self._kwargs(system, messages, tools, max_tokens)))

    def describe_image(self, system, prompt, b64, media_type, max_tokens=300) -> str:
        self._throttle()
        resp = self.client.chat.completions.create(
            model=self.model, **{self.token_param: max_tokens},
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": openai_image_content(b64, media_type, prompt)}])
        return parse_openai(resp).text

    def stream(self, system, messages, tools=None, max_tokens=700):
        self._throttle()
        stream = self.client.chat.completions.create(
            **self._stream_kwargs(system, messages, tools, max_tokens))
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


def get_llm(backend: str | None = None, role: str = "LLM", model: str | None = None) -> BaseClient:
    """Build a client with role-specific settings and an optional model override."""
    backend = (backend or role_setting(role, "BACKEND", "auto")).lower()
    if backend == "auto":
        backend = (
            "anthropic" if env("ANTHROPIC_API_KEY")
            else "openai" if (
                role_setting(role, "API_KEY") or env("OPENAI_API_KEY")
                or role_setting(role, "BASE_URL")
                or (env("OPENAI_BASE_URL") if role.upper() == "LLM" else "")
            )
            else "offline"
        )
    try:
        if backend == "anthropic":
            if role.upper() == "LLM":
                return AnthropicClient(model=model)
            return AnthropicClient(model=model, role=role)
        if backend == "openai":
            if role.upper() == "LLM":
                return OpenAIClient(model=model)
            return OpenAIClient(model=model, role=role)
    except ImportError:  # SDK not installed: degrade instead of crashing at start-up
        pass
    from .offline import OfflineClient
    return OfflineClient()
