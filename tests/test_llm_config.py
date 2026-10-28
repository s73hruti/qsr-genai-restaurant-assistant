"""Provider configuration for free / local OpenAI-compatible endpoints, and honest eval reporting."""
import pytest

from qsr_assistant.agent.agent import Agent
from qsr_assistant.eval import evaluate as ev
from qsr_assistant.eval import retrieval_eval as rev
from qsr_assistant.eval.judge import HeuristicJudge, LLMJudge
from qsr_assistant.llm import BaseClient, get_llm
from qsr_assistant.llm.clients import OpenAIClient

pytest.importorskip("openai")

ENV = ("LLM_BACKEND", "LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY", "LLM_TOKEN_PARAM", "LLM_STREAM_USAGE",
       "LLM_MIN_INTERVAL_S", "LLM_MAX_RETRIES", "JUDGE_BACKEND", "JUDGE_MODEL", "JUDGE_BASE_URL",
       "JUDGE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_BASE_URL")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for v in ENV:
        monkeypatch.delenv(v, raising=False)


def test_base_url_model_and_key_come_from_llm_settings(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("LLM_MODEL", "llama3.1:8b")
    c = get_llm()  # auto: a custom base URL selects the OpenAI-compatible client, no real key needed
    assert isinstance(c, OpenAIClient) and c.model == "llama3.1:8b"
    assert str(c.client.base_url).startswith("http://localhost:11434/v1")


def test_judge_can_be_a_different_model_and_provider(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://a.example/v1")
    monkeypatch.setenv("LLM_API_KEY", "k1")
    monkeypatch.setenv("LLM_MODEL", "model-under-test")
    monkeypatch.setenv("JUDGE_BASE_URL", "https://b.example/v1")
    monkeypatch.setenv("JUDGE_API_KEY", "k2")
    monkeypatch.setenv("JUDGE_MODEL", "judge-model")
    llm, judge = get_llm(), get_llm(role="JUDGE")
    assert (llm.model, judge.model) == ("model-under-test", "judge-model")
    assert "a.example" in str(llm.client.base_url) and "b.example" in str(judge.client.base_url)
    assert llm.client.api_key == "k1" and judge.client.api_key == "k2"


def test_judge_falls_back_to_llm_settings(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://a.example/v1")
    monkeypatch.setenv("LLM_MODEL", "m")
    j = get_llm(role="JUDGE")
    assert j.model == "m" and "a.example" in str(j.client.base_url)


def test_compat_switches(monkeypatch):
    c = OpenAIClient.__new__(OpenAIClient)
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("LLM_TOKEN_PARAM", "max_tokens")
    monkeypatch.setenv("LLM_STREAM_USAGE", "0")
    c = OpenAIClient()
    kw = c._kwargs("sys", [], None, 123)
    assert kw["max_tokens"] == 123 and "max_completion_tokens" not in kw
    skw = c._stream_kwargs("sys", [], None, 5)
    assert skw["stream"] is True and "stream_options" not in skw
    monkeypatch.delenv("LLM_TOKEN_PARAM")
    monkeypatch.delenv("LLM_STREAM_USAGE")
    d = OpenAIClient()
    assert "max_completion_tokens" in d._kwargs("s", [], None, 9)
    assert d._stream_kwargs("s", [], None, 9)["stream_options"] == {"include_usage": True}


def test_throttle_spaces_calls(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("LLM_MIN_INTERVAL_S", "0.2")
    c = OpenAIClient()
    sleeps = []
    import time as _t
    monkeypatch.setattr("qsr_assistant.llm.clients.time.sleep", lambda s: sleeps.append(s))
    BaseClientCls = type(c).__mro__[1]
    BaseClientCls._last_call = _t.monotonic()
    c._throttle()
    assert sleeps and 0 < sleeps[0] <= 0.2


class _Boom(BaseClient):
    name, online, model = "boom", True, "boom-1"

    def chat(self, *a, **k):
        raise RuntimeError("429 rate limited")


def test_judge_failures_are_counted_not_hidden():
    j = LLMJudge(_Boom())
    assert j.faithfulness("The fries cost $3.", "evidence") is None
    assert j.correctness("q", "a", "ref") is None
    assert (j.calls, j.failures) == (2, 2)


def test_eval_reports_answers_served_by_the_offline_fallback(registry):
    """If the provider fails, the agent degrades to offline: the eval must say so, not hide it."""
    agent = Agent(registry, _Boom())
    qa = rev.load_qa()[:3]
    aq, _ = ev.answer_quality(agent, HeuristicJudge(), qa, workers=1)
    assert aq["degraded_answers_%"] == 100.0
    assert aq["judge_calls"] == 0 and aq["faithfulness_n_scored"] >= 0
