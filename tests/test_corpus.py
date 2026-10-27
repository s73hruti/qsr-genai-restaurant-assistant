import json
from pathlib import Path

import qsr_assistant.eval as ev
from qsr_assistant.data.knowledge import all_docs, build_qa
from qsr_assistant.rag.loaders import load_documents


def test_corpus_size_and_unique_ids():
    docs = all_docs()
    assert len(docs) >= 100
    ids = [d["doc_id"] for d in docs]
    assert len(ids) == len(set(ids))
    assert {d["audience"] for d in docs} == {"all", "manager"}
    assert {d["type"] for d in docs} >= {"menu", "promo", "faq", "policy", "store", "guide", "sop", "training"}


def test_every_eval_fact_is_in_its_document():
    for d in all_docs():
        body = " ".join(b for _, b in d["sections"]).lower()
        for q, expect in d["qa"]:
            assert expect.lower() in body, (d["doc_id"], expect)


def test_committed_eval_set_matches_generator():
    path = Path(ev.__file__).parent / "data" / "retrieval_qa.json"
    assert json.loads(path.read_text()) == build_qa(), "run: python scripts/build_corpus.py"
    assert len(build_qa()) >= 150


def test_manager_questions_only_target_manager_docs_or_public_docs():
    by_id = {d["doc_id"]: d for d in all_docs()}
    for q in build_qa():
        if q["audience"] == "customer":
            assert by_id[q["doc_id"]]["audience"] == "all"


def test_documents_on_disk_load(registry):
    from qsr_assistant import config
    docs = load_documents(config.KNOWLEDGE_DIR)
    assert len(docs) >= 100 and all(d.doc_id and d.title for d in docs)
