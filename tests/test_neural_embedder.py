"""The sentence-transformers wrapper is tested with a fake model (no downloads)."""
import sys
from types import SimpleNamespace

import numpy as np

from qsr_assistant.rag.embeddings import SentenceTransformerEmbedder, _prefixes_for
from qsr_assistant.rag.reranker import CrossEncoderReranker
from qsr_assistant.rag.retriever import HybridRetriever


class FakeModel:
    def __init__(self):
        self.seen = []

    def encode(self, texts, show_progress_bar=False):
        self.seen += list(texts)
        out = np.zeros((len(texts), 8), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, hash(w) % 8] += 1
        return out


def test_prefixes_by_family():
    assert _prefixes_for("BAAI/bge-small-en-v1.5")[0].startswith("Represent this sentence")
    assert _prefixes_for("intfloat/e5-small-v2") == ("query: ", "passage: ")
    assert _prefixes_for("sentence-transformers/all-MiniLM-L6-v2") == ("", "")


def test_query_and_passage_prefixes_applied():
    m = FakeModel()
    e = SentenceTransformerEmbedder("intfloat/e5-small-v2", _model=m)
    e.encode(["doc text"])
    e.encode_query(["a question"])
    assert m.seen == ["passage: doc text", "query: a question"]


def test_retriever_uses_query_side_encoder():
    m = FakeModel()
    e = SentenceTransformerEmbedder("intfloat/e5-small-v2", _model=m)
    r = HybridRetriever(llm=None, embedder=e, mode="dense", rewrite=False, rerank=False)
    m.seen.clear()
    r.search("large fries price", k=2)
    assert m.seen == ["query: large fries price"]


def test_sentence_transformer_revision_and_offline_options(monkeypatch):
    captured = {}

    class FakeSentenceTransformer:
        def __init__(self, model_name, **kwargs):
            captured["model_name"] = model_name
            captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    SentenceTransformerEmbedder(
        "example/model",
        revision="abc123",
        local_files_only=True,
    )

    assert captured == {
        "model_name": "example/model",
        "revision": "abc123",
        "local_files_only": True,
    }


def test_cross_encoder_revision_and_offline_options(monkeypatch):
    captured = {}

    class FakeCrossEncoder:
        def __init__(self, model_name, **kwargs):
            captured["model_name"] = model_name
            captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(CrossEncoder=FakeCrossEncoder),
    )

    CrossEncoderReranker(
        "example/cross-encoder",
        revision="def456",
        local_files_only=True,
    )

    assert captured == {
        "model_name": "example/cross-encoder",
        "revision": "def456",
        "local_files_only": True,
    }
