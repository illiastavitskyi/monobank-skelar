"""Qdrant-based RAG backends: sparse BM25, dense embeddings, and hybrid."""

from __future__ import annotations

import numpy as np

import config


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------

class _QdrantBase:
    def _connect(self):
        from qdrant_client import QdrantClient
        self._client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)

    def _needs_reindex(self, expected_count: int, required_vector_names: set[str]) -> bool:
        """True if collection is missing, has wrong count, or wrong vector schema."""
        try:
            existing = {c.name for c in self._client.get_collections().collections}
            if config.QDRANT_COLLECTION not in existing:
                return True
            if self._client.count(config.QDRANT_COLLECTION).count != expected_count:
                return True
            params = self._client.get_collection(config.QDRANT_COLLECTION).config.params
            have: set[str] = set()
            if params.vectors and hasattr(params.vectors, "keys"):
                have.update(params.vectors.keys())
            if params.sparse_vectors:
                have.update(params.sparse_vectors.keys())
            return not required_vector_names.issubset(have)
        except Exception:
            return True

    def _drop_collection(self):
        try:
            self._client.delete_collection(config.QDRANT_COLLECTION)
        except Exception:
            pass

    def _upsert_batched(self, points: list, batch_size: int = 512) -> None:
        for start in range(0, len(points), batch_size):
            self._client.upsert(config.QDRANT_COLLECTION, points[start:start + batch_size])


# ---------------------------------------------------------------------------
# Backend 1: Sparse BM25 (RAG_BACKEND=tfidf)
# ---------------------------------------------------------------------------

class QdrantSparseBackend(_QdrantBase):
    """BM25 sparse vectors via fastembed — keyword search, no neural model needed."""

    VECTOR_NAME = "bm25"

    def __init__(self) -> None:
        self._client = None
        self._encoder = None

    def build_index(self, texts: list[str]) -> None:
        from qdrant_client.models import SparseVectorParams, PointStruct, SparseVector
        from fastembed.sparse.bm25 import Bm25

        self._connect()
        self._encoder = Bm25(config.BM25_MODEL)

        if not self._needs_reindex(len(texts), {self.VECTOR_NAME}):
            print(f"[QdrantSparseBackend] Already indexed ({len(texts)} points).")
            return

        self._drop_collection()
        self._client.create_collection(
            config.QDRANT_COLLECTION,
            vectors_config={},
            sparse_vectors_config={self.VECTOR_NAME: SparseVectorParams()},
        )

        print(f"[QdrantSparseBackend] Encoding {len(texts)} texts with BM25…")
        embeddings = list(self._encoder.passage_embed(texts))

        points = [
            PointStruct(
                id=i,
                vector={self.VECTOR_NAME: SparseVector(
                    indices=embeddings[i].indices.tolist(),
                    values=embeddings[i].values.tolist(),
                )},
            )
            for i in range(len(texts))
        ]
        self._upsert_batched(points)
        print(f"[QdrantSparseBackend] Done.")

    def search(self, query: str, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        from qdrant_client.models import SparseVector

        q = list(self._encoder.query_embed(query))[0]
        results = self._client.query_points(
            config.QDRANT_COLLECTION,
            query=SparseVector(indices=q.indices.tolist(), values=q.values.tolist()),
            using=self.VECTOR_NAME,
            limit=top_k,
        ).points
        return (
            np.array([r.score for r in results]),
            np.array([r.id for r in results], dtype=int),
        )


# ---------------------------------------------------------------------------
# Backend 2: Dense embeddings (RAG_BACKEND=qdrant)
# ---------------------------------------------------------------------------

class QdrantDenseBackend(_QdrantBase):
    """Multilingual sentence-transformer embeddings in Qdrant."""

    VECTOR_NAME = "dense"

    def __init__(self) -> None:
        self._client = None
        self._model = None

    def build_index(self, texts: list[str]) -> None:
        from qdrant_client.models import VectorParams, Distance, PointStruct
        from sentence_transformers import SentenceTransformer

        self._connect()
        self._model = SentenceTransformer(config.EMBEDDING_MODEL)

        if not self._needs_reindex(len(texts), {self.VECTOR_NAME}):
            print(f"[QdrantDenseBackend] Already indexed ({len(texts)} points).")
            return

        self._drop_collection()
        print(f"[QdrantDenseBackend] Encoding {len(texts)} texts…")
        embeddings = self._model.encode(
            texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True
        ).astype(np.float32)

        self._client.create_collection(
            config.QDRANT_COLLECTION,
            vectors_config={self.VECTOR_NAME: VectorParams(size=embeddings.shape[1], distance=Distance.COSINE)},
        )
        points = [
            PointStruct(id=i, vector={self.VECTOR_NAME: embeddings[i].tolist()})
            for i in range(len(texts))
        ]
        self._upsert_batched(points)
        print(f"[QdrantDenseBackend] Done.")

    def search(self, query: str, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        vec = self._model.encode([query], convert_to_numpy=True).astype(np.float32)[0]
        results = self._client.query_points(
            config.QDRANT_COLLECTION,
            query=vec.tolist(),
            using=self.VECTOR_NAME,
            limit=top_k,
        ).points
        return (
            np.array([r.score for r in results]),
            np.array([r.id for r in results], dtype=int),
        )


# ---------------------------------------------------------------------------
# Backend 3: Hybrid BM25 + dense, fused with RRF (RAG_BACKEND=hybrid)
# ---------------------------------------------------------------------------

class QdrantHybridBackend(_QdrantBase):
    """Sparse BM25 + dense embeddings combined with Reciprocal Rank Fusion."""

    SPARSE_NAME = "bm25"
    DENSE_NAME = "dense"

    def __init__(self) -> None:
        self._client = None
        self._encoder = None
        self._model = None

    def build_index(self, texts: list[str]) -> None:
        from qdrant_client.models import VectorParams, Distance, SparseVectorParams, PointStruct, SparseVector
        from fastembed.sparse.bm25 import Bm25
        from sentence_transformers import SentenceTransformer

        self._connect()
        self._encoder = Bm25(config.BM25_MODEL)
        self._model = SentenceTransformer(config.EMBEDDING_MODEL)

        if not self._needs_reindex(len(texts), {self.SPARSE_NAME, self.DENSE_NAME}):
            print(f"[QdrantHybridBackend] Already indexed ({len(texts)} points).")
            return

        self._drop_collection()
        print(f"[QdrantHybridBackend] Encoding {len(texts)} texts (BM25 + dense)…")

        sparse_embs = list(self._encoder.passage_embed(texts))
        dense_embs = self._model.encode(
            texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True
        ).astype(np.float32)

        self._client.create_collection(
            config.QDRANT_COLLECTION,
            vectors_config={
                self.DENSE_NAME: VectorParams(size=dense_embs.shape[1], distance=Distance.COSINE)
            },
            sparse_vectors_config={self.SPARSE_NAME: SparseVectorParams()},
        )

        points = [
            PointStruct(
                id=i,
                vector={
                    self.DENSE_NAME: dense_embs[i].tolist(),
                    self.SPARSE_NAME: SparseVector(
                        indices=sparse_embs[i].indices.tolist(),
                        values=sparse_embs[i].values.tolist(),
                    ),
                },
            )
            for i in range(len(texts))
        ]
        self._upsert_batched(points)
        print(f"[QdrantHybridBackend] Done.")

    def search(self, query: str, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector

        q_sparse = list(self._encoder.query_embed(query))[0]
        q_dense = self._model.encode([query], convert_to_numpy=True).astype(np.float32)[0]

        results = self._client.query_points(
            config.QDRANT_COLLECTION,
            prefetch=[
                Prefetch(
                    query=SparseVector(indices=q_sparse.indices.tolist(), values=q_sparse.values.tolist()),
                    using=self.SPARSE_NAME,
                    limit=top_k * 3,
                ),
                Prefetch(
                    query=q_dense.tolist(),
                    using=self.DENSE_NAME,
                    limit=top_k * 3,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
        ).points
        return (
            np.array([r.score for r in results]),
            np.array([r.id for r in results], dtype=int),
        )
