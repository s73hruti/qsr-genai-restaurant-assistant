"""Measure the numeric-grounding guard on its own: false positives on legitimate answers and
detection of hallucinated numbers, with no LLM involved.

    python -m qsr_assistant.eval.guard_eval            # print the table
    python -m qsr_assistant.eval.guard_eval --check    # exit 1 if FPR / detection thresholds are missed

The cases (`eval/data/guard_cases.json`) are realistic paraphrases a model writes (rounding, sums,
quotients, clock conversion, list numbering, counts) versus invented prices, times and counts. Cases
marked `known_gap` are hallucinations a digit-only guard cannot see (number words, codes like SAVE20);
they are reported, not hidden. This measures the guard's *logic*; the rate at which a real model's
answers get replaced is reported by the real-LLM run as `answers_replaced_by_guard_%`.
"""
from __future__ import annotations

import argparse
import json
import sys

from ..guardrails.output import check_output
from .retrieval_eval import DATA

TH = {"guard_false_positive_max_%": 5.0, "guard_detection_min_%": 90.0}


def run(cases: list[dict] | None = None) -> dict:
    cases = cases if cases is not None else json.loads((DATA / "guard_cases.json").read_text())
    legit = [c for c in cases if c["label"] == "legit"]
    bad = [c for c in cases if c["label"] == "hallucinated" and not c.get("known_gap")]
    gaps = [c for c in cases if c.get("known_gap")]

    def flagged(c) -> bool:
        return "ungrounded_numbers" in check_output(
            c["answer"], c["evidence"], c["question"], "customer").flags

    fp = [c for c in legit if flagged(c)]
    miss = [c for c in bad if not flagged(c)]
    gap_caught = [c for c in gaps if flagged(c)]
    pct = lambda a, b: round(100.0 * a / b, 1) if b else 0.0  # noqa: E731
    return {"legit_n": len(legit), "hallucinated_n": len(bad), "known_gap_n": len(gaps),
            "false_positive_%": pct(len(fp), len(legit)),
            "detection_%": pct(len(bad) - len(miss), len(bad)),
            "known_gaps_caught": len(gap_caught),
            "false_positives": [{"id": c["id"], "note": c["note"]} for c in fp],
            "missed": [{"id": c["id"], "note": c["note"]} for c in miss]}


def check(res: dict) -> list[str]:
    fails = []
    if res["false_positive_%"] > TH["guard_false_positive_max_%"]:
        fails.append(f"guard false positives {res['false_positive_%']}% > {TH['guard_false_positive_max_%']}%")
    if res["detection_%"] < TH["guard_detection_min_%"]:
        fails.append(f"guard detection {res['detection_%']}% < {TH['guard_detection_min_%']}%")
    return fails


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    r = run()
    print(f"legitimate answers wrongly flagged : {r['false_positive_%']}% of {r['legit_n']}")
    print(f"hallucinated numbers detected      : {r['detection_%']}% of {r['hallucinated_n']}")
    print(f"known-gap cases caught             : {r['known_gaps_caught']} of {r['known_gap_n']}")
    for k in ("false_positives", "missed"):
        for x in r[k]:
            print(f"  {k[:-1]:15s} {x['id']}: {x['note']}")
    if args.check:
        fails = check(r)
        if fails:
            print("\n".join(fails), file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
