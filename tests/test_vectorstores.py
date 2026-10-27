import numpy as np
import pytest

from qsr_assistant.rag.vectorstore import NumpyStore, get_store


def _vecs():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(30, 16)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _agree(store):
    v = _vecs()
    store.add(v[:20])
    store.add(v[20:])  # incremental add keeps ids contiguous
    ref = NumpyStore()
    ref.add(v)
    for qi in (0, 7, 25):
        got = [i for i, _ in store.search(v[qi], 5)]
        want = [i for i, _ in ref.search(v[qi], 5)]
        assert got == want and got[0] == qi


def test_numpy_store_self_match():
    _agree(NumpyStore())


def test_faiss_matches_numpy():
    pytest.importorskip("faiss")
    _agree(get_store(16, "faiss"))


def test_qdrant_matches_numpy():
    pytest.importorskip("qdrant_client")
    _agree(get_store(16, "qdrant"))


def test_retriever_runs_on_qdrant():
    pytest.importorskip("qdrant_client")
    from qsr_assistant.rag.retriever import HybridRetriever
    r = HybridRetriever(llm=None, vector_store="qdrant")
    assert r.store.name == "qdrant"
    hits = r.search("how much is the large fries", k=3, mode="dense", rewrite=False, rerank=False)
    assert "menu-large-fries" in [h.chunk.doc_id for h in hits]
