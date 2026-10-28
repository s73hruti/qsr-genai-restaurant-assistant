import os

os.environ["LLM_BACKEND"] = "offline"  # tests never touch the network
for _var in ("MANAGER_API_KEY", "API_KEY", "USER_TOKEN_SECRET", "APP_ENV", "TRUSTED_PROXY_HOPS"):
    os.environ.pop(_var, None)
# Rate limits are on by default in the service; switch them off for the suite and let the tests
# that exercise the limiter set their own values.
os.environ["RATE_LIMIT_PER_MIN"] = "0"
os.environ["RATE_LIMIT_LLM_PER_MIN"] = "0"

import pytest  # noqa: E402

from qsr_assistant.agent.agent import Agent  # noqa: E402
from qsr_assistant.llm import BaseClient, LLMResponse, ToolCall, Usage  # noqa: E402
from qsr_assistant.llm.offline import OfflineClient  # noqa: E402
from qsr_assistant.ml.registry import load_registry  # noqa: E402


@pytest.fixture(scope="session")
def registry(tmp_path_factory):
    return load_registry(artifacts_dir=tmp_path_factory.mktemp("artifacts"))


@pytest.fixture()
def agent(registry):
    return Agent(registry, OfflineClient())


class ScriptedLLM(BaseClient):
    """Fake online LLM that replays scripted responses (used to test the agent loop)."""
    name = "scripted"
    online = True
    model = "claude-sonnet-test"

    def __init__(self, script, raises=None):
        self.script, self.raises, self.calls = list(script), raises, []

    def chat(self, system, messages, tools=None, max_tokens=700):
        self.calls.append({"system": system, "messages": list(messages), "tools": tools})
        if self.raises:
            raise self.raises
        r = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        return r


def text(t, inp=100, out=20):
    return LLMResponse(t, [], Usage(inp, out), "end_turn", "claude-sonnet-test")


def call(name, args=None, i=0, inp=100, out=20):
    return LLMResponse("", [ToolCall(f"tc{i}", name, args or {})], Usage(inp, out),
                       "tool_use", "claude-sonnet-test")
