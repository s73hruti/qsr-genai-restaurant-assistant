import numpy as np

from qsr_assistant.rag.graph import GRAPH_SLOT, ChunkGraph
from qsr_assistant.rag.retriever import HybridRetriever


def test_graph_links_only_across_documents():
    r = HybridRetriever(llm=None)
    g = ChunkGraph.build(r.chunks, r.bm25.idf_of)
    assert g.n_edges > 0
    for i, nbrs in g.neighbours.items():
        for j, w in nbrs:
            assert r.chunks[i].doc_id != r.chunks[j].doc_id
            assert 0 < w <= 1.0


def test_expand_promotes_one_neighbour_into_the_reserved_slot():
    g = ChunkGraph({0: [(9, 1.0), (8, 0.5)]})
    cands = [(0, 1.0), (1, 0.9), (2, 0.8), (3, 0.7), (4, 0.6)]
    mask = np.ones(10, dtype=bool)
    out = g.expand(cands, mask)
    assert [i for i, _ in out[:GRAPH_SLOT]] == [0, 1, 2]
    assert out[GRAPH_SLOT][0] == 9
    assert len(out) == len(cands) + 1


def test_expand_respects_access_mask():
    g = ChunkGraph({0: [(9, 1.0)]})
    mask = np.ones(10, dtype=bool)
    mask[9] = False  # restricted chunk must never be promoted
    cands = [(0, 1.0), (1, 0.9), (2, 0.8), (3, 0.7)]
    assert g.expand(cands, mask) == cands


def test_customer_search_with_graph_never_returns_manager_docs():
    r = HybridRetriever(llm=None, graph=True)
    for q in ["refund approval limits", "how long can fries be held", "delivery late credit"]:
        for h in r.search(q, k=4, audience="customer", graph_expand=True):
            assert h.chunk.audience == "all"
