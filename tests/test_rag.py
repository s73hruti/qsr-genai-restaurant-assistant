import numpy as np
import pytest

from qsr_assistant.eval.metrics import (
    bootstrap_ci,
    fact_in_text,
    hit_at_k,
    is_abstention,
    mrr_at_k,
    ndcg_at_k,
)
from qsr_assistant.rag.bm25 import BM25
from qsr_assistant.rag.chunking import chunk_document
from qsr_assistant.rag.embeddings import get_embedder
from qsr_assistant.rag.loaders import Document, load_documents, parse_front_matter
from qsr_assistant.rag.retriever import HybridRetriever
from qsr_assistant.rag.rewrite import QueryRewriter, expand


def doc(text, audience="all"):
    return Document("d1", "Doc", audience, "faq", text, "d1.md")


def test_front_matter_and_legacy_audience():
    meta, body = parse_front_matter("---\ndoc_id: x\naudience: manager # internal\n---\n# T\nbody")
    assert meta == {"doc_id": "x", "audience": "manager"} and body.startswith("# T")
    meta, _ = parse_front_matter("<!-- audience: manager -->\n# T")
    assert meta["audience"] == "manager"


def test_unknown_audience_fails_closed(tmp_path):
    (tmp_path / "a.md").write_text("---\naudience: secret\n---\n# A\n## S\ntext")
    assert load_documents(tmp_path)[0].audience == "manager"


def test_chunking_strategies():
    d = doc("# Doc\n\n## A\nalpha\n\n## B\nbeta")
    assert [c.heading for c in chunk_document(d, strategy="section")] == ["A", "B"]
    merged = chunk_document(d, strategy="merged")
    assert len(merged) == 1 and "alpha" in merged[0].text and "beta" in merged[0].text
    long = doc("# Doc\n\n## A\n" + "\n\n".join(["word " * 80] * 6))
    assert len(chunk_document(long, max_chars=700)) > 1
    assert all(c.audience == "all" and c.doc_id == "d1" for c in merged)


def test_bm25_prefers_rare_matching_term():
    bm = BM25().fit(["the cat sat", "the dog barked loudly", "a zebra grazed"])
    s = bm.scores("zebra")
    assert int(np.argmax(s)) == 2 and s[0] == 0


def test_embedders_fit_and_encode():
    texts = [f"doc about topic {i} and fries" for i in range(12)]
    for name in ("tfidf", "lsa"):
        e = get_embedder(name).fit(texts)
        v = e.encode(["fries topic 3"])
        assert v.shape[0] == 1 and abs(np.linalg.norm(v[0]) - 1) < 1e-4


def test_hybrid_retrieves_relevant_menu_item(registry):
    hits = registry.retriever.search("is there a vegetarian burger", k=3)
    assert any(h.chunk.doc_id == "menu-veggie-delight-burger" for h in hits)


def test_modes_all_work(registry):
    for mode in ("bm25", "dense", "hybrid"):
        hits = registry.retriever.search("how much are large fries", k=3, mode=mode, rerank=False)
        assert hits and hits[0].chunk.doc_id.startswith("menu-")


def test_customer_never_retrieves_manager_chunks(registry):
    r = registry.retriever
    queries = ["fries holding time", "cash drop limit", "refund approval limit crew",
               "staffing minimum crew peak", "fryer oil temperature", "ingredient order cutoff"]
    for q in queries:
        for mode in ("bm25", "dense", "hybrid"):
            assert all(h.chunk.audience == "all"
                       for h in r.search(q, k=10, audience="customer", mode=mode))
    mgr = r.search("fries holding time minutes", k=3, audience="manager")
    assert "7 minutes" in mgr[0].chunk.text


def test_doc_type_filter(registry):
    hits = registry.retriever.search("fries", k=5, doc_types=["promo"])
    assert hits and all(h.chunk.type == "promo" for h in hits)
    assert "sop" not in registry.retriever.doc_types("customer")
    assert "sop" in registry.retriever.doc_types("manager")


def test_confidence_lower_for_out_of_scope(registry):
    r = registry.retriever
    inn = max(h.confidence for h in r.search("how much are large fries", k=3))
    out = max((h.confidence for h in r.search("quantum chromodynamics lecture notes", k=3)), default=0.0)
    assert inn > 0.5 > out


def test_rewrite_thesaurus_and_followup():
    assert "fries" in expand("how much are chips")
    rw = QueryRewriter().rewrite("and for kids?", ["tell me about the cone offer"])
    assert rw.method.startswith("followup") and "cone" in rw.query


def test_followup_resolution_uses_history(registry):
    hits = registry.retriever.search("And what time does it close?", k=3,
                                     history=["Does the Riverside store have a drive-thru?"])
    assert hits[0].chunk.doc_id == "store-riverside"


def test_self_contained_followup_is_not_polluted(registry):
    hits = registry.retriever.search("What about the medium?", k=3,
                                     history=["How much is the Large Fries?"])
    assert hits[0].chunk.doc_id == "menu-medium-fries"


def test_rerank_keeps_candidates_and_scores_sorted(registry):
    hits = registry.retriever.search("how long can i keep chips", k=5, audience="manager", rerank=True)
    scores = [h.score for h in hits[:5]]
    assert scores == sorted(scores, reverse=True)


def test_retriever_requires_documents(tmp_path):
    with pytest.raises(RuntimeError):
        HybridRetriever(tmp_path)


def test_ranking_metrics():
    ranked = ["a", "b", "c"]
    assert hit_at_k(ranked, "b", 1) == 0 and hit_at_k(ranked, "b", 2) == 1
    assert mrr_at_k(ranked, "c", 5) == pytest.approx(1 / 3)
    assert ndcg_at_k(ranked, "a", 5) == 1.0 and ndcg_at_k(ranked, "z", 5) == 0.0
    lo, hi = bootstrap_ci([1, 0, 1, 1, 0, 1, 1, 1])
    assert 0 <= lo <= hi <= 1


def test_fact_matching_and_abstention_detection():
    assert fact_in_text("costs $3.40", "The Large Fries is $3.40 today [menu-large-fries]")
    assert not fact_in_text("costs $3.40", "It costs $2.80")
    assert is_abstention("I don't have that information.")
    assert not is_abstention("Fries cost $3.40.")
