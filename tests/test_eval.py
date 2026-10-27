import json

import pytest
from conftest import ScriptedLLM, text

from qsr_assistant import config
from qsr_assistant.eval import evaluate, retrieval_eval
from qsr_assistant.eval.abstain import calibrate
from qsr_assistant.eval.judge import HeuristicJudge, LLMJudge, split_claims


def test_heuristic_faithfulness():
    j = HeuristicJudge()
    ev = "The Large Fries costs $3.40. Category: sides."
    assert j.faithfulness("The Large Fries costs $3.40 [menu-large-fries].", ev) == 1.0
    assert j.faithfulness("The Large Fries costs $4.90. Free soda included.", ev) < 1.0
    assert split_claims("One thing. Another thing [doc-1].") == ["One thing.", "Another thing ."]


def test_llm_judge_parses_json_and_survives_garbage():
    good = LLMJudge(ScriptedLLM([text('ok {"claims":[{"claim":"a","supported":true},'
                                      '{"claim":"b","supported":false}]}')]))
    assert good.faithfulness("a. b.", "ev") == 0.5
    assert LLMJudge(ScriptedLLM([text('{"correct": true}')])).correctness("q", "a", "r") == 1.0
    assert LLMJudge(ScriptedLLM([text("no json here")])).faithfulness("a", "e") is None
    assert LLMJudge(ScriptedLLM([], raises=RuntimeError("x"))).correctness("q", "a", "r") is None


def test_ablation_runs_and_default_is_reported(registry):
    qa = retrieval_eval.load_qa()[:40]
    rep = retrieval_eval.run_ablation(qa=qa)
    names = [r["config"] for r in rep["rows"]]
    assert retrieval_eval.DEFAULT_CONFIG in names and len(names) >= 8
    assert all(0 <= r["hit@3"] <= 100 and r["n"] == 40 for r in rep["rows"])
    assert "| Configuration |" in retrieval_eval.to_markdown(rep)


def test_abstention_calibration_respects_answer_rate_constraint(registry):
    c = calibrate(registry.retriever)
    assert c["dev"]["answered_in_scope"] >= 0.88 and 0 < c["threshold"] < 0.8


def test_evaluate_cli_offline_smoke(tmp_path, monkeypatch, registry):
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path)
    assert evaluate.main(["--limit", "30", "--out", str(tmp_path / "e.json")]) == 0
    rep = json.loads((tmp_path / "e.json").read_text())
    assert rep["routing"]["manager_tool_executed_for_customer"] == 0
    assert rep["redteam"]["manager_info_leaked"] == 0
    assert (tmp_path / "ablation.md").exists() and (tmp_path / "summary_offline.md").exists()


def test_online_eval_without_key_exits_cleanly(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert evaluate.main(["--llm", "online"]) == 2


def test_thresholds_check_detects_regression():
    rep = {"retrieval": {"default": {"hit@3": 10}}, "answer_quality": {
        "answer_contains_fact_%": 99, "faithfulness_%": 99, "ungrounded_number_flags_%": 0},
        "routing": {"tool_selection_exact_match_%": 99, "manager_tool_executed_for_customer": 1},
        "redteam": {"input_guardrail_block_rate_%": 99, "benign_false_positive_rate_%": 0,
                    "manager_info_leaked": 0}}
    fails = evaluate.check_thresholds(rep)
    assert any("retrieval_hit@3_min" in f for f in fails)
    assert any("manager_tool_violations_max" in f for f in fails)
    pytest.importorskip("json")
