"""BM25 (Okapi) over unigram stems. Scores a query against every chunk, with an optional mask."""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from .embeddings import unigram_analyzer


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.vec = CountVectorizer(analyzer=unigram_analyzer)
        self.tf = None
        self.idf = None

    def fit(self, texts: list[str]) -> "BM25":
        tf = self.vec.fit_transform(texts).tocsr().astype(np.float32)
        n = tf.shape[0]
        df = np.asarray((tf > 0).sum(axis=0)).ravel()
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
        dl = np.asarray(tf.sum(axis=1)).ravel()
        self.avgdl = float(dl.mean()) or 1.0
        self.tf, self.dl = tf.tocsc(), dl
        self.vocab = self.vec.vocabulary_
        return self

    def idf_of(self, token: str) -> float:
        i = self.vocab.get(token)
        return float(self.idf[i]) if i is not None else 0.0

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.tf.shape[0], dtype=np.float32)
        for tok in set(unigram_analyzer(query)):
            i = self.vocab.get(tok)
            if i is None:
                continue
            col = self.tf[:, i]
            rows, tfv = col.indices, col.data
            denom = tfv + self.k1 * (1 - self.b + self.b * self.dl[rows] / self.avgdl)
            out[rows] += self.idf[i] * tfv * (self.k1 + 1) / denom
        return out
