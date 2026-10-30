"""Hybrid retriever: query rewrite -> BM25 + dense -> RRF fusion -> rerank, with access control.

Access control is applied as a *mask before ranking*: chunks the caller may not see never enter
the candidate set, so they cannot influence scores, rank positions or confidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import config
from ..config import env
from .bm25 import BM25
from .chunking import Chunk, chunk_document
from .embeddings import get_embedder, unigram_analyzer
from .loaders import load_documents
from .reranker import get_reranker
from .rewrite import THESAURUS, QueryRewriter, Rewrite, expand
from .vectorstore import get_store

RRF_K = 60
CANDIDATES = 20
RERANK_DEPTH = 10
MODES = ("bm25", "dense", "hybrid")


@dataclass
class Hit:
    chunk: Chunk
    score: float
    confidence: float = 0.0
    ranks: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"doc_id": self.chunk.doc_id, "chunk_id": self.chunk.chunk_id,
                "source": self.chunk.source, "heading": self.chunk.heading, "type": self.chunk.type,
                "score": round(float(self.score), 4), "confidence": round(self.confidence, 3),
                "text": self.chunk.text}


def allowed_audiences(audience: str) -> set[str]:
    return {"all", "manager"} if audience == "manager" else {"all"}


class HybridRetriever:
    def __init__(self, knowledge_dir: Path = config.KNOWLEDGE_DIR, embedder=None, llm=None,
                 mode: str | None = None, rerank: bool | None = None,
                 rewrite: bool | None = None, chunk_strategy: str | None = None,
                 documents: list | None = None, reranker: str | None = None,
                 graph: bool | None = None, vector_store: str | None = None,
                 reranker_revision: str | None = None,
                 reranker_local_files_only: bool = False):
        # `documents` lets callers index any corpus (e.g. a BEIR dataset) instead of a folder
        docs = documents if documents is not None else load_documents(knowledge_dir)
        self.chunk_strategy = chunk_strategy or env("CHUNK_STRATEGY", "merged")
        self.chunks: list[Chunk] = [c for d in docs
                                    for c in chunk_document(d, strategy=self.chunk_strategy)]
        if not self.chunks:
            raise RuntimeError("No knowledge docs found to index")
        self.n_docs = len(docs)
        texts = [c.text for c in self.chunks]
        self.embedder = embedder or get_embedder(env("EMBEDDING_BACKEND", "tfidf"))
        self.embedder.fit(texts)
        vectors = self.embedder.encode(texts)
        self.store = get_store(vectors.shape[1], vector_store)
        self.store.add(vectors)
        self.bm25 = BM25().fit(texts)
        self._idf_max = float(self.bm25.idf.max())
        self.reranker = get_reranker(
            self.bm25.idf_of,
            reranker,
            revision=reranker_revision,
            local_files_only=reranker_local_files_only,
        )
        self.rewriter = QueryRewriter(llm)
        self.mode = mode or env("RETRIEVAL_MODE", "hybrid")
        self.use_rerank = rerank if rerank is not None else env("RERANK", "1") == "1"
        self.use_rewrite = rewrite if rewrite is not None else env("QUERY_REWRITE", "1") == "1"
        self.use_graph = graph if graph is not None else env("GRAPH_EXPAND", "0") == "1"
        self._graph = None
        self._chunk_tokens = [set(unigram_analyzer(c.text)) for c in self.chunks]
        self._types = {c.audience: set() for c in self.chunks}
        for c in self.chunks:
            self._types[c.audience].add(c.type)

    # ------------------------------------------------------------------ helpers
    @property
    def graph(self):
        """Chunk graph for GraphRAG-style expansion, built lazily on first use."""
        if self._graph is None:
            from .graph import ChunkGraph
            self._graph = ChunkGraph.build(self.chunks, self.bm25.idf_of)
        return self._graph

    def doc_types(self, audience: str) -> list[str]:
        ok = allowed_audiences(audience)
        return sorted({c.type for c in self.chunks if c.audience in ok})

    def manager_shingles(self, n: int = 7) -> set[tuple]:
        """n-gram shingles of manager-only text, used by the output guardrail to detect leaks."""
        out: set[tuple] = set()
        for c in self.chunks:
            if c.audience != "all":
                toks = [t for t in c.text.lower().split()]
                out.update(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))
        return out

    def confidence(self, query: str, idx: int) -> float:
        """idf-weighted share of the user's content words found in the chunk (via thesaurus too)."""
        words = [w for w in {t for t in query.lower().replace("?", " ").split()}
                 if unigram_analyzer(w)]
        if not words:
            return 0.0
        toks = self._chunk_tokens[idx]
        num = den = 0.0
        for w in words:
            stem = unigram_analyzer(w)[0]
            variants = {stem} | set(unigram_analyzer(THESAURUS.get(w, "")))
            weight = self.bm25.idf_of(stem) or self._idf_max
            den += weight
            if toks & variants:
                num += weight
        return num / den if den else 0.0

    def _self_contained(self, query: str, audience: str, threshold: float = 0.9) -> bool:
        """True if the question alone already matches a chunk almost completely."""
        ok = allowed_audiences(audience)
        mask = np.array([c.audience in ok for c in self.chunks])
        scores = self.bm25.scores(expand(query))
        if not mask.any() or not np.any(scores[mask] > 0):
            return False
        idx = int(np.argmax(np.where(mask, scores, -np.inf)))
        return self.confidence(query, idx) >= threshold

    def _ranked(self, scores: np.ndarray, mask: np.ndarray, min_score: float) -> list[int]:
        s = np.where(mask, scores, -np.inf)
        order = np.argsort(-s)
        return [int(i) for i in order if s[i] > min_score][:CANDIDATES * 2]

    # ------------------------------------------------------------------ search
    def search(self, query: str, k: int = 3, audience: str = "customer",
               doc_types: list[str] | None = None, mode: str | None = None,
               rerank: bool | None = None, rewrite: bool | None = None,
               history: list[str] | None = None,
               graph_expand: bool | None = None) -> list[Hit]:
        mode = mode or self.mode
        rerank = self.use_rerank if rerank is None else rerank
        rewrite = self.use_rewrite if rewrite is None else rewrite
        rw = self.rewriter.rewrite(query, history) if rewrite else Rewrite(query, query, "none")
        if rw.method.startswith("followup") and self._self_contained(query, audience):
            rw = self.rewriter.rewrite(query, None)  # don't pollute a clear question with history
        q = rw.query

        ok = allowed_audiences(audience)
        mask = np.array([c.audience in ok and (not doc_types or c.type in doc_types)
                         for c in self.chunks])
        if not mask.any():
            return []

        rank_lists: dict[str, list[int]] = {}
        raw: dict[str, np.ndarray] = {}
        if mode in ("bm25", "hybrid"):
            raw["bm25"] = self.bm25.scores(q)
            rank_lists["bm25"] = self._ranked(raw["bm25"], mask, 0.0)
        if mode in ("dense", "hybrid"):
            dense = np.full(len(self.chunks), -1.0, dtype=np.float32)
            encode_q = getattr(self.embedder, "encode_query", self.embedder.encode)
            for i, s in self.store.search(encode_q([q])[0], len(self.chunks)):
                dense[i] = s
            raw["dense"] = dense
            rank_lists["dense"] = self._ranked(dense, mask, 0.05)

        if mode == "hybrid":
            fused: dict[int, float] = {}
            for lst in rank_lists.values():
                for r, i in enumerate(lst[:CANDIDATES]):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + r + 1)
            ordered = sorted(fused.items(), key=lambda kv: -kv[1])
            top = ordered[0][1] if ordered else 1.0
            cands = [(i, s / top) for i, s in ordered]
        else:
            key = "bm25" if mode == "bm25" else "dense"
            lst = rank_lists[key]
            top = float(raw[key][lst[0]]) if lst else 1.0
            cands = [(i, float(raw[key][i]) / (top or 1.0)) for i in lst[:CANDIDATES]]

        if rerank and self.reranker is not None and cands:
            depth = cands[:RERANK_DEPTH]
            rr = self.reranker.rerank(q, [self.chunks[i] for i, _ in depth], len(depth))
            cands = [(depth[j][0], s) for j, s in rr] + cands[RERANK_DEPTH:]

        if graph_expand is None:
            graph_expand = self.use_graph
        if graph_expand and cands:
            cands = self.graph.expand(cands, mask)

        hits = []
        for i, s in cands[:k]:
            hits.append(Hit(self.chunks[i], s, self.confidence(query, i),
                            {m: (rank_lists[m].index(i) + 1 if i in rank_lists[m] else None)
                             for m in rank_lists}))
        return hits


Retriever = HybridRetriever
