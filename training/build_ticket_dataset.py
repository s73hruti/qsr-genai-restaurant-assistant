"""Build the synthetic ticket-extraction SFT dataset (chat format) and print the rule baseline.

    python training/build_ticket_dataset.py --n 600
"""
import argparse
import json
from pathlib import Path

from qsr_assistant.data.tickets import build_dataset, to_chat, write_jsonl
from qsr_assistant.eval.ticket_eval import rule_baseline
from qsr_assistant.llm.structured import TICKET_INSTRUCTION

OUT = Path(__file__).parent / "data"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    train, val = build_dataset(a.n, a.seed)
    write_jsonl([to_chat(x, TICKET_INSTRUCTION) for x in train], OUT / "tickets_train.jsonl")
    write_jsonl(val, OUT / "tickets_val.jsonl")
    base = rule_baseline(val)
    (OUT / "tickets_rule_baseline.json").write_text(json.dumps(base, indent=1))
    print(f"train={len(train)} val={len(val)}")
    print("rule-based baseline on val:", base)
