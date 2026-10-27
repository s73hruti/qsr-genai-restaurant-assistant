"""Lightweight GraphRAG: a chunk graph linked by shared rare entities, used to expand retrieval.

Nodes are chunks. Two chunks from *different* documents are linked when they share rare terms
(menu-item names, store names, allergens, policy names: terms that occur in only a few chunks),
weighted by the summed idf of the shared terms. At query time the top-ranked chunks act as seeds
and the strongest unseen neighbour is promoted into one reserved slot of the result list
(position GRAPH_SLOT, 0-based), so a question that spans two documents can surface both.

This is deliberately small: no LLM entity extraction, no community summaries (the full
Microsoft-style GraphRAG). It is the cheap, inspectable variant, and `make ablation` reports whether
it helps on this corpus rather than assuming it does.
"""
from __future__ import annotations

from collections import defaultdict

from .embeddings import unigram_analyzer

GRAPH_SLOT = 3          # 0-based rank that is reserved for a graph neighbour (the 4th result)
MAX_DF = 6              # a term shared by more chunks than this is too generic to be a link
MIN_IDF = 2.0
MAX_NEIGHBOURS = 8
SEEDS = 3


class ChunkGraph:
    def __init__(self, neighbours: dict[int, list[tuple[int, float]]]):
        self.neighbours = neighbours

    @classmethod
    def build(cls, chunks: list, idf_fn, max_df: int = MAX_DF, min_idf: float = MIN_IDF) -> "ChunkGraph":
        postings: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(chunks):
            for t in set(unigram_analyzer(c.text)):
                postings[t].append(i)
        weights: dict[tuple[int, int], float] = defaultdict(float)
        for term, ids in postings.items():
            if not 2 <= len(ids) <= max_df:
                continue
            idf = idf_fn(term)
            if idf < min_idf:
                continue
            for a in ids:
                for b in ids:
                    if a < b and chunks[a].doc_id != chunks[b].doc_id:
                        weights[(a, b)] += idf
        adj: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for (a, b), w in weights.items():
            adj[a].append((b, w))
            adj[b].append((a, w))
        out = {}
        for i, nb in adj.items():
            top = sorted(nb, key=lambda x: -x[1])[:MAX_NEIGHBOURS]
            norm = top[0][1]
            out[i] = [(j, w / norm) for j, w in top]
        return cls(out)

    @property
    def n_edges(self) -> int:
        return sum(len(v) for v in self.neighbours.values()) // 2

    def expand(self, cands: list[tuple[int, float]], mask, slot: int = GRAPH_SLOT,
               seeds: int = SEEDS) -> list[tuple[int, float]]:
        """Promote the best unseen neighbour of the top seeds into rank `slot`.

        `mask` is the access-control mask, so a neighbour the caller may not see is never promoted.
        """
        if len(cands) <= slot - 1:
            return cands
        head = cands[:slot]
        seen = {i for i, _ in head}
        best: tuple[int, float] | None = None
        for i, s in cands[:seeds]:
            for j, w in self.neighbours.get(i, []):
                if j in seen or not bool(mask[j]):
                    continue
                score = s * w
                if best is None or score > best[1]:
                    best = (j, score)
        if best is None:
            return cands
        return head + [best] + [c for c in cands[slot:] if c[0] != best[0]]
