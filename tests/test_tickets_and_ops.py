import json
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from qsr_assistant.api import main as api_main
from qsr_assistant.data.tickets import build_dataset, to_chat
from qsr_assistant.eval.ticket_eval import rule_baseline, score
from qsr_assistant.llm.structured import TICKET_INSTRUCTION

ROOT = Path(__file__).resolve().parents[1]


def test_ticket_dataset_is_deterministic_and_deduplicated():
    tr1, va1 = build_dataset(200, 3)
    tr2, va2 = build_dataset(200, 3)
    assert (tr1, va1) == (tr2, va2)
    assert not {x["text"] for x in tr1} & {x["text"] for x in va1}
    rec = to_chat(tr1[0], TICKET_INSTRUCTION)
    assert [m["role"] for m in rec["messages"]] == ["system", "user", "assistant"]
    assert json.loads(rec["messages"][2]["content"])["category"] == tr1[0]["ticket"]["category"]


def test_scorer_counts_invalid_json_as_invalid():
    gold = [{"category": "wait_time", "severity": "low", "store_id": None, "items": ["cola"],
             "needs_human": False}]
    assert score(["not json"], gold)["valid_json_%"] == 0.0
    ok = {"category": "wait_time", "severity": "low", "store_id": None, "items": ["cola"],
          "needs_human": False, "summary": "s"}
    s = score([json.dumps(ok)], gold)
    assert s["exact_match_%"] == 100.0
    ok["category"] = "other"
    assert score([ok], gold)["category_acc_%"] == 0.0


def test_rule_baseline_runs():
    _, val = build_dataset(100, 5)
    assert rule_baseline(val)["valid_json_%"] == 100.0


def test_rate_limiter_returns_429_then_recovers(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "3")
    api_main._limiter.buckets.clear()
    c = TestClient(api_main.app)
    codes = [c.get("/wait-time/1").status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[3:] == [429, 429]
    assert c.get("/health").status_code == 200  # health is never limited
    api_main._limiter.buckets.clear()
    assert c.get("/wait-time/1").status_code == 200


def test_resume_script_refuses_offline_report(tmp_path):
    rep = ROOT / "docs" / "results" / "eval_offline.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_resume_bullets.py"),
                        "--report", str(rep)], capture_output=True, text=True)
    assert r.returncode != 0 and "offline" in (r.stderr + r.stdout).lower()


def test_resume_script_fills_numbers_from_a_real_report(tmp_path):
    rep = json.loads((ROOT / "docs" / "results" / "eval_offline.json").read_text())
    rep["llm_backend"] = "anthropic:test-model"
    rep["answer_quality"]["faithfulness_judge"] = "llm"  # a real report is scored by a real judge
    rep["answer_quality"]["degraded_answers_%"] = 0.0
    p = tmp_path / "eval_anthropic.json"
    p.write_text(json.dumps(rep))
    out = ROOT / "docs" / "RESUME_BULLETS.md"
    before = out.read_text() if out.exists() else None
    try:
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_resume_bullets.py"),
                            "--report", str(p)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        txt = out.read_text()
        assert "[X]" not in txt and "[N]" not in txt and "anthropic" in txt
        assert f"{rep['answer_quality']['faithfulness_%']}%" in txt
    finally:
        (out.write_text(before) if before is not None else out.unlink(missing_ok=True))
        (ROOT / "docs" / "results" / "eval_anthropic.json").unlink(missing_ok=True)


def _fake_real_report(tmp_path, **aq_overrides):
    rep = json.loads((ROOT / "docs" / "results" / "eval_offline.json").read_text())
    rep["llm_backend"] = "openai:some-free-model"
    rep["answer_quality"].update({"faithfulness_judge": "llm", "degraded_answers_%": 0.0})
    rep["answer_quality"].update(aq_overrides)
    p = tmp_path / "eval_free.json"
    p.write_text(json.dumps(rep))
    return p


def _run_bullets(path, *extra):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "make_resume_bullets.py"),
                           "--report", str(path), *extra], capture_output=True, text=True)


def test_resume_script_refuses_heuristic_judge_report(tmp_path):
    r = _run_bullets(_fake_real_report(tmp_path, faithfulness_judge="heuristic"))
    assert r.returncode != 0 and "heuristic" in (r.stderr + r.stdout).lower()


def test_resume_script_refuses_report_contaminated_by_offline_fallback(tmp_path):
    r = _run_bullets(_fake_real_report(tmp_path, **{"degraded_answers_%": 12.5}))
    assert r.returncode != 0 and "offline fallback" in (r.stderr + r.stdout).lower()
