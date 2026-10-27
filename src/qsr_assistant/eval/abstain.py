"""Calibrate the retrieval abstention threshold on a dev split, report on the held-out test split.

Run: python -m qsr_assistant.eval.abstain
Rule: the largest threshold that keeps >=88% of valid dev questions answerable (false refusals
are worse than a weak answer). The chosen value is hard-coded as ABSTAIN_THRESHOLD in agent/tools.py (and overridable with the
ABSTAIN_THRESHOLD env var). Split: even indices = dev, odd indices = test.
"""
from __future__ import annotations

import json

import numpy as np

from ..rag.retriever import HybridRetriever
from .retrieval_eval import DATA, load_qa


def confidences(ret: HybridRetriever, items: list[dict]) -> np.ndarray:
    out = []
    for it in items:
        hits = ret.search(it["q"], k=4, audience=it.get("audience", it.get("mode", "customer")))
        out.append(max((h.confidence for h in hits), default=0.0))
    return np.array(out)


def split(items):
    return items[0::2], items[1::2]


def rates(c_in: np.ndarray, c_oos: np.ndarray, thr: float) -> dict:
    return {"answered_in_scope": float((c_in >= thr).mean()),
            "abstained_out_of_scope": float((c_oos < thr).mean())}


MIN_ANSWER_RATE = 0.88  # never refuse more than ~12% of valid questions on the dev split


def calibrate(ret: HybridRetriever | None = None) -> dict:
    ret = ret or HybridRetriever()
    qa = [dict(it, audience=it["audience"]) for it in load_qa()]
    oos = [dict(it, audience=it["mode"]) for it in json.loads((DATA / "out_of_scope.json").read_text())]
    (in_dev, in_test), (oo_dev, oo_test) = split(qa), split(oos)
    c = {k: confidences(ret, v) for k, v in
         dict(in_dev=in_dev, in_test=in_test, oo_dev=oo_dev, oo_test=oo_test).items()}
    best = 0.0  # largest threshold that still answers >= MIN_ANSWER_RATE of in-scope dev questions
    for thr in np.arange(0.0, 0.8, 0.01):
        if rates(c["in_dev"], c["oo_dev"], thr)["answered_in_scope"] >= MIN_ANSWER_RATE:
            best = float(round(thr, 2))
    return {"threshold": best, "dev": rates(c["in_dev"], c["oo_dev"], best),
            "test": rates(c["in_test"], c["oo_test"], best),
            "n": {k: len(v) for k, v in c.items()}}


if __name__ == "__main__":
    print(json.dumps(calibrate(), indent=2))
