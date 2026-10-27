"""Run the retrieval ablation on a real public benchmark in BEIR format (e.g. SciFact, FiQA, NFCorpus).

BEIR layout:  corpus.jsonl  {"_id","title","text"}   queries.jsonl {"_id","text"}   qrels/test.tsv
(query-id <TAB> corpus-id <TAB> score, with a header row).

Why this exists: the bundled corpus is synthetic and written by the author. A public benchmark with
human relevance labels shows how the retrieval stack behaves on text you did not write. Download with
`python scripts/run_beir.py --dataset scifact` (needs internet; ~3 MB for SciFact).
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from ..rag.loaders import Document
from ..rag.retriever import HybridRetriever
from .metrics import bootstrap_ci


def load_beir(root: Path, split: str = "test") -> tuple[list[Document], list[dict]]:
    docs = []
    with open(root / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            docs.append(Document(doc_id=str(d["_id"]), title=d.get("title") or str(d["_id"]),
                                 audience="all", type="doc",
                                 text=f"# {d.get('title') or d['_id']}\n\n{d.get('text', '')}",
                                 source=str(d["_id"])))
    queries = {}
    with open(root / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries[str(q["_id"])] = q["text"]
    rel: dict[str, dict[str, int]] = {}
    with open(root / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 3 or not row[2].lstrip("-").isdigit():
                continue  # header
            if int(row[2]) > 0:
                rel.setdefault(row[0], {})[row[1]] = int(row[2])
    qa = [{"id": qid, "q": queries[qid], "relevant": r} for qid, r in rel.items() if qid in queries]
    return docs, qa


def ndcg10(ranked: list[str], rel: dict[str, int]) -> float:
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
    ideal = sorted(rel.values(), reverse=True)[:10]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def evaluate(ret: HybridRetriever, qa: list[dict], **kw) -> dict:
    ndcg, rec, mrr = [], [], []
    for item in qa:
        hits = ret.search(item["q"], k=100, audience="customer", **kw)
        ranked: list[str] = []
        for h in hits:
            if h.chunk.doc_id not in ranked:
                ranked.append(h.chunk.doc_id)
        rel = item["relevant"]
        ndcg.append(ndcg10(ranked, rel))
        rec.append(len(set(ranked[:100]) & set(rel)) / len(rel))
        first = next((i for i, d in enumerate(ranked[:10]) if d in rel), None)
        mrr.append(0.0 if first is None else 1 / (first + 1))
    lo, hi = bootstrap_ci(ndcg)
    mean = lambda x: sum(x) / len(x)  # noqa: E731
    return {"ndcg@10": round(mean(ndcg), 4), "recall@100": round(mean(rec), 4),
            "mrr@10": round(mean(mrr), 4), "ndcg@10_ci95": [round(lo, 4), round(hi, 4)],
            "n_queries": len(qa)}


def run(root: Path, neural: bool = False, limit: int | None = None) -> list[dict]:
    from ..rag.embeddings import get_embedder
    docs, qa = load_beir(root)
    if limit:
        qa = qa[:limit]
    base = dict(rerank=False, rewrite=False)
    tfidf = HybridRetriever(documents=docs, embedder=get_embedder("tfidf"), llm=None,
                            chunk_strategy="section")
    rows = [("BM25", evaluate(tfidf, qa, mode="bm25", **base)),
            ("Dense: TF-IDF", evaluate(tfidf, qa, mode="dense", **base)),
            ("Hybrid: BM25 + TF-IDF (RRF)", evaluate(tfidf, qa, mode="hybrid", **base))]
    if neural:
        from ..rag.embeddings import SentenceTransformerEmbedder
        emb = SentenceTransformerEmbedder()
        nr = HybridRetriever(documents=docs, embedder=emb, llm=None, chunk_strategy="section")
        rows += [(f"Dense: {emb.model_name.split('/')[-1]}", evaluate(nr, qa, mode="dense", **base)),
                 (f"Hybrid: BM25 + {emb.model_name.split('/')[-1]} (RRF)",
                  evaluate(nr, qa, mode="hybrid", **base))]
        try:
            ce = HybridRetriever(documents=docs, embedder=emb, llm=None, reranker="cross-encoder",
                                 chunk_strategy="section")
            rows.append(("Hybrid + cross-encoder rerank",
                         evaluate(ce, qa, mode="hybrid", rerank=True, rewrite=False)))
        except Exception:
            pass
    return [{"config": n, **r} for n, r in rows]


def to_markdown(name: str, rows: list[dict], n_docs: int) -> str:
    lines = [f"BEIR/{name}: {rows[0]['n_queries']} queries over {n_docs} documents.", "",
             "| Configuration | nDCG@10 | MRR@10 | Recall@100 | nDCG@10 95% CI |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['config']} | {r['ndcg@10']} | {r['mrr@10']} | {r['recall@100']} | "
                     f"{r['ndcg@10_ci95'][0]}-{r['ndcg@10_ci95'][1]} |")
    return "\n".join(lines)
