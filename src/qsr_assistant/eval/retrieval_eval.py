"""Retrieval ablation: BM25 vs dense vs hybrid, +query rewrite, +rerank, chunking strategy."""
from __future__ import annotations

import json
from pathlib import Path

from ..config import KNOWLEDGE_DIR
from ..rag.embeddings import get_embedder
from ..rag.retriever import HybridRetriever
from .metrics import bootstrap_ci, dedupe, hit_at_k, mrr_at_k, ndcg_at_k, norm, pct

DATA = Path(__file__).parent / "data"
DEFAULT_CONFIG = "Hybrid (BM25 + TF-IDF) + rewrite + rerank"


def load_qa() -> list[dict]:
    return json.loads((DATA / "retrieval_qa.json").read_text())


def build_variants(knowledge_dir: Path = KNOWLEDGE_DIR) -> dict[str, HybridRetriever]:
    """One retriever per embedding/chunking variant. Neural variants are added when available."""
    v = {
        "lsa": HybridRetriever(knowledge_dir, embedder=get_embedder("lsa"), llm=None),
        "tfidf": HybridRetriever(knowledge_dir, embedder=get_embedder("tfidf"), llm=None),
        "tfidf-section": HybridRetriever(knowledge_dir, embedder=get_embedder("tfidf"), llm=None,
                                         chunk_strategy="section"),
    }
    # Optional neural variants. They need `pip install ".[vector]"` and a model download, so they
    # are skipped silently when unavailable and appear in the report only when they actually ran.
    try:
        from ..rag.embeddings import SentenceTransformerEmbedder
        minilm = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2")
        v["minilm"] = HybridRetriever(knowledge_dir, embedder=minilm, llm=None)
        try:
            v["minilm-ce"] = HybridRetriever(knowledge_dir, embedder=minilm, llm=None,
                                             reranker="cross-encoder")
        except Exception:
            pass
        try:
            bge = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")
            v["bge"] = HybridRetriever(knowledge_dir, embedder=bge, llm=None)
            if "minilm-ce" in v:
                v["bge-ce"] = HybridRetriever(knowledge_dir, embedder=bge, llm=None,
                                              reranker="cross-encoder")
        except Exception:
            pass
    except Exception:
        pass
    return v


def configs(variants: dict) -> list[tuple[str, str, dict]]:
    off = dict(rerank=False, rewrite=False)
    cfg = [
        ("BM25", "tfidf", dict(mode="bm25", **off)),
        ("Dense: TF-IDF cosine", "tfidf", dict(mode="dense", **off)),
        ("Dense: LSA (SVD)", "lsa", dict(mode="dense", **off)),
        ("Hybrid: BM25 + LSA (RRF)", "lsa", dict(mode="hybrid", **off)),
        ("Hybrid: BM25 + TF-IDF (RRF)", "tfidf", dict(mode="hybrid", **off)),
        ("BM25 + rewrite", "tfidf", dict(mode="bm25", rerank=False, rewrite=True)),
        ("BM25 + rewrite + rerank", "tfidf", dict(mode="bm25", rerank=True, rewrite=True)),
        ("Hybrid (BM25 + TF-IDF) + rewrite", "tfidf", dict(mode="hybrid", rerank=False, rewrite=True)),
        ("Hybrid (BM25 + TF-IDF) + rewrite + rerank", "tfidf",
         dict(mode="hybrid", rerank=True, rewrite=True)),
        ("Hybrid (BM25 + LSA) + rewrite + rerank", "lsa", dict(mode="hybrid", rerank=True, rewrite=True)),
        ("Hybrid (BM25 + TF-IDF) + rewrite + rerank, section chunks", "tfidf-section",
         dict(mode="hybrid", rerank=True, rewrite=True)),
        ("Default + graph-neighbour slot (GraphRAG-lite)", "tfidf",
         dict(mode="hybrid", rerank=True, rewrite=True, graph_expand=True)),
    ]
    if "minilm" in variants:
        cfg += [("Dense: MiniLM", "minilm", dict(mode="dense", **off)),
                ("Hybrid: BM25 + MiniLM + rewrite", "minilm",
                 dict(mode="hybrid", rerank=False, rewrite=True)),
                ("Hybrid: BM25 + MiniLM + rewrite + lexical rerank", "minilm",
                 dict(mode="hybrid", rerank=True, rewrite=True))]
    if "minilm-ce" in variants:
        cfg += [("Hybrid: BM25 + MiniLM + rewrite + cross-encoder", "minilm-ce",
                 dict(mode="hybrid", rerank=True, rewrite=True))]
    if "bge" in variants:
        cfg += [("Dense: bge-small", "bge", dict(mode="dense", **off)),
                ("Hybrid: BM25 + bge-small + rewrite", "bge",
                 dict(mode="hybrid", rerank=False, rewrite=True))]
    if "bge-ce" in variants:
        cfg += [("Hybrid: BM25 + bge-small + rewrite + cross-encoder", "bge-ce",
                 dict(mode="hybrid", rerank=True, rewrite=True))]
    return cfg


def evaluate_config(ret: HybridRetriever, qa: list[dict], **kw) -> tuple[dict, list[dict]]:
    h1, h3, h5, mrr, ndcg, fact, fails = [], [], [], [], [], [], []
    for item in qa:
        hits = ret.search(item["q"], k=10, audience=item["audience"], **kw)
        ranked = dedupe([h.chunk.doc_id for h in hits])
        h1.append(hit_at_k(ranked, item["doc_id"], 1))
        h3.append(hit_at_k(ranked, item["doc_id"], 3))
        h5.append(hit_at_k(ranked, item["doc_id"], 5))
        mrr.append(mrr_at_k(ranked, item["doc_id"], 5))
        ndcg.append(ndcg_at_k(ranked, item["doc_id"], 5))
        fact.append(float(any(norm(item["expect"]) in norm(h.chunk.text) for h in hits[:3])))
        if not h3[-1]:
            fails.append({"q": item["q"], "gold": item["doc_id"], "got": ranked[:3]})
    mean = lambda x: sum(x) / len(x)  # noqa: E731
    lo, hi = bootstrap_ci(h3)
    mlo, mhi = bootstrap_ci(mrr)
    row = {"hit@1": pct(mean(h1)), "hit@3": pct(mean(h3)), "hit@5": pct(mean(h5)),
           "mrr@5": round(mean(mrr), 3), "ndcg@5": round(mean(ndcg), 3),
           "fact_recall@3": pct(mean(fact)), "hit@3_ci95": [pct(lo), pct(hi)],
           "mrr@5_ci95": [round(mlo, 3), round(mhi, 3)], "n": len(qa)}
    return row, fails


def load_multihop() -> list[dict]:
    return json.loads((DATA / "multihop_qa.json").read_text())


def run_multihop(variants: dict, k: int = 4) -> list[dict]:
    """Two-document questions: does the top-k (k=4, the agent's tool default) contain BOTH gold docs?"""
    qa = load_multihop()
    ret = variants["tfidf"]
    out = []
    for name, kw in [("Default (hybrid + rewrite + rerank)", {}),
                     ("Default + graph-neighbour slot", {"graph_expand": True})]:
        both, recall = [], []
        for item in qa:
            docs = {h.chunk.doc_id for h in ret.search(item["q"], k=k, audience=item["audience"], **kw)}
            gold = set(item["doc_ids"])
            both.append(float(gold <= docs))
            recall.append(len(gold & docs) / len(gold))
        lo, hi = bootstrap_ci(both)
        out.append({"config": name, "both_docs@4": pct(sum(both) / len(both)),
                    "doc_recall@4": pct(sum(recall) / len(recall)),
                    "both_docs_ci95": [pct(lo), pct(hi)], "n": len(qa)})
    return out


def run_ablation(knowledge_dir: Path = KNOWLEDGE_DIR, qa: list[dict] | None = None) -> dict:
    qa = qa or load_qa()
    variants = build_variants(knowledge_dir)
    rows, default_fails = [], []
    for name, variant, kw in configs(variants):
        row, fails = evaluate_config(variants[variant], qa, **kw)
        rows.append({"config": name, **row})
        if name == DEFAULT_CONFIG:
            default_fails = fails
    # per-type breakdown for the default configuration
    default = variants["tfidf"]
    by_type: dict[str, list[float]] = {}
    for item in qa:
        hits = default.search(item["q"], k=3, audience=item["audience"])
        by_type.setdefault(item["type"], []).append(
            hit_at_k(dedupe([h.chunk.doc_id for h in hits]), item["doc_id"], 3))
    multihop = run_multihop(variants)
    return {"rows": rows, "multihop": multihop, "by_type_hit@3": {t: pct(sum(v) / len(v)) for t, v in sorted(by_type.items())},
            "n_docs": default.n_docs, "n_chunks": len(default.chunks), "n_questions": len(qa),
            "default_failures": default_fails[:25]}


def to_markdown(report: dict) -> str:
    lines = [f"Retrieval ablation - {report['n_questions']} questions over {report['n_docs']} documents "
             f"({report['n_chunks']} chunks in the default chunking).", "",
             "| Configuration | Hit@1 | Hit@3 | Hit@5 | MRR@5 | nDCG@5 | Fact recall@3 | Hit@3 95% CI |",
             "|---|---|---|---|---|---|---|---|"]
    for r in report["rows"]:
        lines.append(f"| {r['config']} | {r['hit@1']} | {r['hit@3']} | {r['hit@5']} | {r['mrr@5']} | "
                     f"{r['ndcg@5']} | {r['fact_recall@3']} | {r['hit@3_ci95'][0]}-{r['hit@3_ci95'][1]} |")
    if report.get("multihop"):
        lines += ["", f"Multi-hop (two-document) questions, n={report['multihop'][0]['n']}, top-4 results:", "",
                  "| Configuration | Both gold docs in top-4 | Doc recall@4 | 95% CI (both) |", "|---|---|---|---|"]
        for r in report["multihop"]:
            lines.append(f"| {r['config']} | {r['both_docs@4']} | {r['doc_recall@4']} | "
                         f"{r['both_docs_ci95'][0]}-{r['both_docs_ci95'][1]} |")
    return "\n".join(lines)
