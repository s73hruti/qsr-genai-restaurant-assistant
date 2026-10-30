"""Pluggable embedding backends.

* tfidf (default): fits on the knowledge base, no downloads or API keys needed.
* lsa: TF-IDF + truncated SVD, a dense offline embedding
* sentence-transformers: `pip install ".[vector]"`, set EMBEDDING_BACKEND=sentence-transformers
* openai: set EMBEDDING_BACKEND=openai and OPENAI_API_KEY
"""
from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from ..config import env


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-9, None)


_TOKEN = re.compile(r"[a-z0-9$]+")


def _stem(tok: str) -> str:
    """Tiny suffix stripper so 'ordering'/'orders'/'ordered' share a stem."""
    for suf in ("ing", "ed", "es", "s"):
        if len(tok) > len(suf) + 3 and tok.endswith(suf):
            return tok[: -len(suf)]
    return tok


def unigram_analyzer(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in ENGLISH_STOP_WORDS]


def analyzer(text: str) -> list[str]:
    toks = unigram_analyzer(text)
    return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]


class TfidfEmbedder:
    name = "tfidf"

    def __init__(self):
        self.vec = TfidfVectorizer(analyzer=analyzer, sublinear_tf=True)

    def fit(self, texts: list[str]) -> "TfidfEmbedder":
        self.vec.fit(texts)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        return _normalize(self.vec.transform(texts).toarray())


class LSAEmbedder:
    """Latent semantic analysis: TF-IDF followed by truncated SVD.

    A genuinely dense, offline embedding. It captures term co-occurrence, so it is a useful
    stand-in for neural embeddings when no model can be downloaded, but it is much weaker.
    """
    name = "lsa"

    def __init__(self, dim: int = 128):
        from sklearn.decomposition import TruncatedSVD
        self.vec = TfidfVectorizer(analyzer=analyzer, sublinear_tf=True)
        self.dim = dim
        self.svd_cls = TruncatedSVD

    def fit(self, texts):
        X = self.vec.fit_transform(texts)
        # Keep the latent space well below the corpus size, otherwise SVD just reproduces TF-IDF
        k = max(2, min(self.dim, X.shape[0] // 3, X.shape[1] - 1))
        self.svd = self.svd_cls(n_components=k, random_state=0).fit(X)
        return self

    def encode(self, texts):
        return _normalize(self.svd.transform(self.vec.transform(texts)))


# Query/passage prefixes that retrieval-tuned model families expect
_PREFIXES = (
    ("bge", "Represent this sentence for searching relevant passages: ", ""),
    ("e5", "query: ", "passage: "),
)


def _prefixes_for(model: str) -> tuple[str, str]:
    m = model.lower()
    for key, q, d in _PREFIXES:
        if key in m:
            return q, d
    return "", ""


class SentenceTransformerEmbedder:
    """Neural bi-encoder. EMBEDDING_MODEL picks the model (default MiniLM; try BAAI/bge-small-en-v1.5
    or intfloat/e5-small-v2). Query/passage prefixes are applied automatically for bge and e5."""
    name = "sentence-transformers"

    def __init__(
        self,
        model: str | None = None,
        _model=None,
        revision: str | None = None,
        local_files_only: bool = False,
    ):
        self.model_name = model or env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.revision = revision
        self.local_files_only = local_files_only
        if _model is None:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(
                self.model_name,
                revision=revision,
                local_files_only=local_files_only,
            )
        self.model = _model
        self.q_prefix, self.d_prefix = _prefixes_for(self.model_name)

    def fit(self, texts):
        return self

    def _enc(self, texts, prefix):
        return _normalize(self.model.encode([prefix + t for t in texts], show_progress_bar=False))

    def encode(self, texts):  # documents / passages
        return self._enc(list(texts), self.d_prefix)

    def encode_query(self, texts):
        return self._enc(list(texts), self.q_prefix)


class OpenAIEmbedder:
    name = "openai"

    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model

    def fit(self, texts):
        return self

    def encode(self, texts):
        resp = self.client.embeddings.create(model=self.model, input=list(texts))
        return _normalize(np.array([d.embedding for d in resp.data]))


def get_embedder(backend: str | None = None):
    backend = (backend or env("EMBEDDING_BACKEND", "tfidf")).lower()
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder()
    if backend == "openai":
        return OpenAIEmbedder()
    if backend == "lsa":
        return LSAEmbedder()
    return TfidfEmbedder()
