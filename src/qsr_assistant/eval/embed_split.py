"""Leak-free train/held-out split for embedding fine-tuning, with BM25 hard negatives.

The split is by *document*: every question about a held-out document is excluded from training,
so improvement on the held-out questions measures generalisation to unseen documents' vocabulary
rather than memorisation of the training questions. (Documents are still in the retrieval index at
test time, as they would be in production.)
"""
from __future__ import annotations

import random

from ..rag.retriever import HybridRetriever


def split_by_doc(qa: list[dict], held_out_frac: float = 0.3, seed: int = 13):
    docs = sorted({q["doc_id"] for q in qa})
    rng = random.Random(seed)
    rng.shuffle(docs)
    held = set(docs[: int(len(docs) * held_out_frac)])
    return [q for q in qa if q["doc_id"] not in held], [q for q in qa if q["doc_id"] in held]


def build_triplets(ret: HybridRetriever, train_qa: list[dict]) -> list[dict]:
    """(query, positive chunk text, hard-negative chunk text). Negatives = top BM25 chunks from other docs."""
    by_doc: dict[str, list] = {}
    for c in ret.chunks:
        by_doc.setdefault(c.doc_id, []).append(c)
    out = []
    for q in train_qa:
        pos_chunks = by_doc.get(q["doc_id"], [])
        if not pos_chunks:
            continue
        pos = next((c for c in pos_chunks if q["expect"].lower() in c.text.lower()), pos_chunks[0])
        hits = ret.search(q["q"], k=6, audience=q["audience"], mode="bm25", rerank=False,
                          rewrite=False)
        neg = next((h.chunk for h in hits if h.chunk.doc_id != q["doc_id"]), None)
        if neg is not None:
            out.append({"query": q["q"], "positive": pos.text, "negative": neg.text,
                        "doc_id": q["doc_id"]})
    return out
