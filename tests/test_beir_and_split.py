import json

from qsr_assistant.eval import beir
from qsr_assistant.eval.embed_split import build_triplets, split_by_doc
from qsr_assistant.eval.retrieval_eval import load_qa
from qsr_assistant.rag.retriever import HybridRetriever


def _fixture(tmp_path):
    (tmp_path / "qrels").mkdir()
    docs = [{"_id": "d1", "title": "Zebra crossing rules", "text": "Pedestrians cross at the zebra."},
            {"_id": "d2", "title": "Quantum computing", "text": "Qubits use superposition."},
            {"_id": "d3", "title": "Sourdough baking", "text": "Wild yeast ferments the dough."}]
    (tmp_path / "corpus.jsonl").write_text("\n".join(json.dumps(d) for d in docs))
    (tmp_path / "queries.jsonl").write_text("\n".join(json.dumps(q) for q in [
        {"_id": "q1", "text": "how do qubits work with superposition"},
        {"_id": "q2", "text": "baking bread with wild yeast"}]))
    (tmp_path / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td2\t1\nq2\td3\t1\n")
    return tmp_path


def test_beir_loader_and_run(tmp_path):
    root = _fixture(tmp_path)
    docs, qa = beir.load_beir(root)
    assert len(docs) == 3 and len(qa) == 2 and qa[0]["relevant"] == {"d2": 1}
    rows = beir.run(root)
    bm25 = next(r for r in rows if r["config"] == "BM25")
    assert bm25["ndcg@10"] == 1.0 and bm25["n_queries"] == 2


def test_ndcg_known_values():
    assert beir.ndcg10(["a", "b"], {"a": 1}) == 1.0
    assert 0.6 < beir.ndcg10(["x", "a"], {"a": 1}) < 0.64  # 1/log2(3)


def test_embedding_split_has_no_document_leakage():
    qa = load_qa()
    train, held = split_by_doc(qa)
    assert train and held and len(train) + len(held) == len(qa)
    assert not {q["doc_id"] for q in train} & {q["doc_id"] for q in held}
    assert split_by_doc(qa) == (train, held)  # deterministic


def test_triplets_use_other_documents_as_negatives():
    ret = HybridRetriever(llm=None)
    train, _ = split_by_doc(load_qa())
    trips = build_triplets(ret, train[:30])
    assert trips
    texts = {c.text: c.doc_id for c in ret.chunks}
    for t in trips:
        assert texts[t["positive"]] == t["doc_id"] and texts[t["negative"]] != t["doc_id"]
