"""Explicit neural retrieval benchmark with latency and per-query failures.

Install optional dependencies first: pip install -e '.[vector]'
Run: python scripts/run_neural_retrieval_benchmark.py
Optional: --limit 20 --model BAAI/bge-small-en-v1.5 --cross-encoder
The script intentionally fails if neural dependencies/model loading fail; it never silently reports
lexical-only numbers as neural retrieval results.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qsr_assistant.eval import retrieval_eval as rev  # noqa: E402
from qsr_assistant.rag.embeddings import SentenceTransformerEmbedder  # noqa: E402
from qsr_assistant.rag.retriever import HybridRetriever  # noqa: E402


def pct(x: float) -> float:
    return round(100 * x, 2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0, help="limit QA rows; 0 means full set")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--cross-encoder", action="store_true", help="also benchmark cross-encoder reranking")
    ap.add_argument("--out", default="reports/retrieval_neural_benchmark.json")
    args = ap.parse_args()
    try:
        import sentence_transformers
    except ImportError as exc:
        raise SystemExit("Missing neural dependencies. Run: pip install -e '.[vector]'") from exc

    qa = rev.load_qa()
    if args.limit:
        qa = qa[:args.limit]
    if not qa:
        raise SystemExit("Evaluation dataset is empty")

    print(f"Loading neural model: {args.model}")
    t0 = time.perf_counter()
    embedder = SentenceTransformerEmbedder(args.model)
    ret = HybridRetriever(embedder=embedder, llm=None, rerank=False, rewrite=False)
    build_seconds = time.perf_counter() - t0
    configs = [
        ("BM25", "bm25", False, None),
        (f"Dense: {args.model}", "dense", False, None),
        (f"Hybrid: BM25 + {args.model}", "hybrid", False, None),
        (f"Hybrid + lexical rerank: {args.model}", "hybrid", True, "lexical"),
    ]
    if args.cross_encoder:
        # Fail loudly if the cross-encoder dependency or model is unavailable.
        configs.append((f"Hybrid + cross-encoder: {args.model}", "hybrid", True, "cross-encoder"))

    rows, failures = [], {}
    for name, mode, rerank, reranker in configs:
        if reranker:
            ret.reranker = __import__("qsr_assistant.rag.reranker", fromlist=["get_reranker"]).get_reranker(
                ret.bm25.idf_of, reranker)
        else:
            ret.reranker = None
        timings, h1, h3, h5, mrr, ndcg = [], [], [], [], [], []
        misses = []
        for item in qa:
            start = time.perf_counter()
            hits = ret.search(item["q"], k=10, audience=item["audience"], mode=mode,
                              rerank=rerank, rewrite=False, graph_expand=False)
            timings.append((time.perf_counter() - start) * 1000)
            ranked = rev.dedupe([h.chunk.doc_id for h in hits])
            h1.append(rev.hit_at_k(ranked, item["doc_id"], 1))
            h3.append(rev.hit_at_k(ranked, item["doc_id"], 3))
            h5.append(rev.hit_at_k(ranked, item["doc_id"], 5))
            mrr.append(rev.mrr_at_k(ranked, item["doc_id"], 5))
            ndcg.append(rev.ndcg_at_k(ranked, item["doc_id"], 5))
            if not h3[-1]:
                misses.append({"question": item["q"], "expected_doc_id": item["doc_id"],
                               "top3_doc_ids": ranked[:3]})
        rows.append({
            "config": name, "mode": mode, "reranker": reranker or "none", "n": len(qa),
            "recall_at_1_pct": pct(statistics.mean(h1)),
            "recall_at_3_pct": pct(statistics.mean(h3)),
            "recall_at_5_pct": pct(statistics.mean(h5)),
            "mrr_at_5": round(statistics.mean(mrr), 4),
            "ndcg_at_5": round(statistics.mean(ndcg), 4),
            "latency_ms_median": round(statistics.median(timings), 3),
            "latency_ms_p95": round(sorted(timings)[min(len(timings)-1, int(.95*len(timings)))], 3),
            "latency_ms_mean": round(statistics.mean(timings), 3),
            "failures_at_3": len(misses),
        })
        failures[name] = misses
        print(f"{name}: Recall@3={rows[-1]['recall_at_3_pct']}%, "
              f"MRR@5={rows[-1]['mrr_at_5']}, median={rows[-1]['latency_ms_median']} ms")

    out = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "qsr_assistant.eval.data/retrieval_qa.json",
        "n_questions": len(qa), "model": args.model,
        "sentence_transformers_version": sentence_transformers.__version__,
        "python_version": platform.python_version(),
        "index_build_seconds": round(build_seconds, 3),
        "limitations": ["Latency is local single-process retrieval latency, not end-to-end LLM latency.",
                        "No claim of improvement should be made unless compared on the same dataset/hardware.",
                        "Run the full dataset; --limit is intended only for smoke tests."],
        "results": rows, "failure_examples": failures,
    }
    dest = (ROOT / args.out).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved benchmark: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
