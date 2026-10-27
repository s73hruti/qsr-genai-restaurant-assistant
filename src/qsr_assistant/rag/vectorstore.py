"""Vector stores: FAISS or Qdrant when available, NumPy cosine search otherwise.

VECTOR_STORE = auto | numpy | faiss | qdrant.  Qdrant runs in-process by default (QDRANT_PATH for
on-disk, empty for in-memory) or against a server (QDRANT_URL, optional QDRANT_API_KEY).
"""
from __future__ import annotations

import numpy as np

from ..config import env


class NumpyStore:
    name = "numpy"

    def __init__(self):
        self.mat = None

    def add(self, vectors: np.ndarray) -> None:
        self.mat = vectors if self.mat is None else np.vstack([self.mat, vectors])

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        scores = self.mat @ query
        idx = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in idx]


class FaissStore:
    name = "faiss"

    def __init__(self, dim: int):
        import faiss
        self.index = faiss.IndexFlatIP(dim)

    def add(self, vectors: np.ndarray) -> None:
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        k = min(k, self.index.ntotal)
        scores, idx = self.index.search(np.ascontiguousarray(query[None, :], dtype=np.float32), k)
        return [(int(i), float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]


class QdrantStore:
    """Qdrant collection with cosine distance. Point ids are the chunk positions."""
    name = "qdrant"

    def __init__(self, dim: int, collection: str | None = None):
        from qdrant_client import QdrantClient, models
        self._models = models
        url, path = env("QDRANT_URL"), env("QDRANT_PATH")
        if url:
            self.client = QdrantClient(url=url, api_key=env("QDRANT_API_KEY") or None)
        elif path:
            self.client = QdrantClient(path=path)
        else:
            self.client = QdrantClient(":memory:")
        self.collection = collection or env("QDRANT_COLLECTION", "qsr_knowledge")
        self.n = 0
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))

    def add(self, vectors: np.ndarray) -> None:
        pts = [self._models.PointStruct(id=self.n + i, vector=v.astype(float).tolist())
               for i, v in enumerate(vectors)]
        self.client.upsert(self.collection, points=pts, wait=True)
        self.n += len(pts)

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        res = self.client.query_points(self.collection, query=query.astype(float).tolist(),
                                       limit=min(k, self.n))
        return [(int(p.id), float(p.score)) for p in res.points]


def get_store(dim: int, backend: str | None = None):
    backend = (backend or env("VECTOR_STORE", "auto")).lower()
    if backend == "qdrant":
        return QdrantStore(dim)
    if backend in ("auto", "faiss"):
        try:
            return FaissStore(dim)
        except ImportError:
            if backend == "faiss":
                raise
    return NumpyStore()
