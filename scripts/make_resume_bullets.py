"""Turn a real-LLM evaluation report into resume bullets with the measured numbers filled in.

    make eval-llm                              # writes reports/eval_anthropic.json (or _openai)
    python scripts/make_resume_bullets.py      # writes docs/RESUME_BULLETS.md, copies results to docs/results/

Refuses to run on the offline report: offline faithfulness and tool-selection are circular (see README),
so they must not appear on a resume. Nothing here invents a number; every figure comes from the JSON.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def find_report(explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit)
    for name in ("eval_anthropic.json", "eval_openai.json"):
        p = ROOT / "reports" / name
        if p.exists():
            return p
    return None


def bullets(rep: dict) -> str:
    aq, rt, rd = rep["answer_quality"], rep["routing"], rep["redteam"]
    n_att = len(rd["attacks"]) if isinstance(rd["attacks"], list) else rd["attacks"]
    ab, perf = rep["abstention"], rep["performance"]
    best = max(rep["ablation_rows"], key=lambda r: r["hit@3"])
    return f"""# Resume bullets (generated from {rep['llm_backend']}, n={rep['n_questions']} questions)

* Built a tool-calling GenAI assistant ({rep.get('provider') or rep['llm_backend'].split(':')[0]} function calling, bounded agent loop,
  LangGraph critic/verifier variant, streaming FastAPI, MCP server) combining hybrid RAG over
  {rep['retrieval']['n_docs']} documents with demand-forecast, wait-time and recommender models as tools.
* Evaluated the RAG stack with {len(rep['ablation_rows'])} retrieval configurations and bootstrap confidence intervals
  (best Hit@3 {best['hit@3']}% for "{best['config']}"); LLM-as-judge faithfulness {aq['faithfulness_%']}% and
  reference-fact correctness {aq['answer_contains_fact_%']}% on {aq['n']} questions.
* Designed layered LLM safety with permissions enforced in code: {rd['attack_success_rate_%']}% attack success on a
  {n_att}-prompt red-team set, {rd['benign_false_positive_rate_%']}% benign false positives,
  {rd['manager_info_leaked']} manager-info leaks; model tool-selection accuracy {rt['tool_selection_exact_match_%']}%.
* Added tracing, token/cost tracking (about ${perf['estimated_cost_usd_per_100_queries']} per 100 queries,
  p50/p95 latency {perf['latency_ms_p50']}/{perf['latency_ms_p95']} ms) and a CI evaluation gate; abstention on
  out-of-scope questions {ab['abstained_%']}%.

Caveats to keep in the README next to these numbers: synthetic corpus, eval sets written by the author,
single run (re-run for variance), judge model = {rep.get('judge', 'see report')}, model under test = {rep['llm_backend']}.
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report")
    ap.add_argument("--allow-degraded", action="store_true",
                    help="write bullets even if some answers came from the offline fallback")
    a = ap.parse_args()
    path = find_report(a.report)
    if path is None or not path.exists():
        sys.exit("No real-LLM report found. Run `make eval-llm` first (needs ANTHROPIC_API_KEY or OPENAI_API_KEY).")
    rep = json.loads(path.read_text())
    if rep["llm_backend"].startswith("offline"):
        sys.exit("This is an offline report; its LLM-dependent metrics are circular. Run `make eval-llm`.")
    aq = rep["answer_quality"]
    if aq.get("faithfulness_judge") != "llm":
        sys.exit("This report was scored by the heuristic judge, which is circular. Re-run with a real judge "
                 "(see docs/FREE_EVAL.md).")
    if aq.get("degraded_answers_%", 0) > 0 and not a.allow_degraded:
        sys.exit(f"{aq['degraded_answers_%']}% of answers came from the offline fallback after provider errors, "
                 "so these are not purely real-LLM numbers. Re-run more slowly (LLM_MIN_INTERVAL_S) or pass "
                 "--allow-degraded.")
    if rep.get("judge") == rep["llm_backend"]:
        print("note: the judge is the same model as the one under test; say so next to the numbers.",
              file=sys.stderr)
    (ROOT / "docs" / "RESUME_BULLETS.md").write_text(bullets(rep))
    res = ROOT / "docs" / "results"
    shutil.copy(path, res / path.name)
    summary = path.parent / path.name.replace("eval_", "summary_").replace(".json", ".md")
    if summary.exists():
        shutil.copy(summary, res / summary.name)
    print(f"wrote docs/RESUME_BULLETS.md and copied {path.name} to docs/results/")
