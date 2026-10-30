"""Second-stage rerankers.

* LexicalReranker (default, offline): hand-weighted features - idf-weighted query-term coverage,
  bigram phrase matches, title/heading overlap and a prior from first-stage rank. It is NOT a
  learned model; it exists so the reranking stage is testable without downloads.
* CrossEncoderReranker: `pip install ".[vector]"`, RERANKER=cross-encoder. Uses a sentence-
  transformers cross-encoder (default cross-encoder/ms-marco-MiniLM-L-6-v2).
"""
from __future__ import annotations

from ..config import env
from .embeddings import unigram_analyzer


class LexicalReranker:
    name = "lexical"

    def __init__(self, idf_fn):
        self.idf = idf_fn

    def score(self, query: str, chunk, prior_rank: int) -> float:
        q = unigram_analyzer(query)
        if not q:
            return 0.0
        body = unigram_analyzer(chunk.text)
        bset = set(body)
        weights = {t: max(self.idf(t), 0.1) for t in set(q)}
        cover = sum(w for t, w in weights.items() if t in bset) / sum(weights.values())
        bigrams = {f"{a} {b}" for a, b in zip(body, body[1:])}
        qb = [f"{a} {b}" for a, b in zip(q, q[1:])]
        phrase = sum(b in bigrams for b in qb) / len(qb) if qb else 0.0
        head = set(unigram_analyzer(f"{chunk.title} {chunk.heading}"))
        head_hit = sum(w for t, w in weights.items() if t in head) / sum(weights.values())
        return 0.55 * cover + 0.2 * phrase + 0.15 * head_hit + 0.1 / (1 + prior_rank)

    def rerank(self, query: str, chunks: list, top_k: int) -> list[tuple[int, float]]:
        scored = [(i, self.score(query, c, i)) for i, c in enumerate(chunks)]
        return sorted(scored, key=lambda x: -x[1])[:top_k]


class CrossEncoderReranker:
    name = "cross-encoder"

    def __init__(
        self,
        model: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
    ):
        from sentence_transformers import CrossEncoder
        self.model_name = model or env(
            "CROSS_ENCODER_MODEL",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        )
        self.revision = revision
        self.local_files_only = local_files_only
        self.model = CrossEncoder(
            self.model_name,
            revision=revision,
            local_files_only=local_files_only,
        )

    def rerank(self, query: str, chunks: list, top_k: int) -> list[tuple[int, float]]:
        scores = self.model.predict([(query, c.text) for c in chunks])
        order = sorted(range(len(chunks)), key=lambda i: -scores[i])[:top_k]
        return [(i, float(scores[i])) for i in order]


def get_reranker(
    idf_fn,
    backend: str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
):
    backend = (backend or env("RERANKER", "lexical")).lower()
    if backend == "cross-encoder":
        return CrossEncoderReranker(
            revision=revision,
            local_files_only=local_files_only,
        )
    if backend == "none":
        return None
    return LexicalReranker(idf_fn)
