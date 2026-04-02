"""RetrievalEngine: wraps a backend + DataEngine, exposes find_similar()."""

from __future__ import annotations

import time
import pandas as pd

import config
from retrieval.data_engine import DataEngine
from retrieval.backends import QdrantSparseBackend, QdrantDenseBackend, QdrantHybridBackend


class RetrievalEngine:
    """Unified retrieval interface. Backend selected by config.RAG_BACKEND."""

    def __init__(self) -> None:
        self._data = DataEngine()
        if config.RAG_BACKEND == "hybrid":
            self._backend = QdrantHybridBackend()
        elif config.RAG_BACKEND == "qdrant":
            self._backend = QdrantDenseBackend()
        else:  # tfidf → BM25 in Qdrant
            self._backend = QdrantSparseBackend()

    # ------------------------------------------------------------------
    # Public properties delegated to DataEngine
    # ------------------------------------------------------------------

    @property
    def category_dict(self) -> dict[str, str]:
        return self._data.category_dict

    def get_category_stats(self, category_id: str) -> dict | None:
        return self._data.get_category_stats(category_id)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        self._data.load()
        texts = self._data.df["text"].fillna("").tolist()
        self._backend.build_index(texts)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def find_similar(
        self,
        query: str,
        category_id: str | None = None,
        top_k: int = config.RAG_TOP_K,
    ) -> list[dict]:
        """Return top-k comparable listings for the query.

        When category_id is given, over-fetches then post-filters so the
        interface is identical regardless of backend.
        """
        t0 = time.perf_counter()

        # Over-fetch when filtering to ensure enough results survive the filter
        fetch_k = top_k * 8 if category_id else top_k
        scores, indices = self._backend.search(query, top_k=fetch_k)

        df = self._data.df
        results: list[dict] = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(df):
                continue
            row = df.iloc[int(idx)]
            if category_id and row["category_id"] != str(category_id):
                continue
            results.append({
                "title": row["title"],
                "status": row["status"],
                "original_price": row["original_price"],
                "sold_price": (
                    row["sold_price"]
                    if not pd.isna(row.get("sold_price", float("nan")))
                    else None
                ),
                "sold_via_bargain": row.get("sold_via_bargain"),
                "category_id": row["category_id"],
                "category_name": self._data.category_dict.get(str(row["category_id"]), ""),
                "similarity_score": round(float(score), 4),
            })
            if len(results) >= top_k:
                break

        # If category filter yielded too few results, fall back to unfiltered
        if category_id and len(results) < top_k // 2:
            scores, indices = self._backend.search(query, top_k=top_k)
            for score, idx in zip(scores, indices):
                if idx < 0 or idx >= len(df):
                    continue
                row = df.iloc[int(idx)]
                results.append({
                    "title": row["title"],
                    "status": row["status"],
                    "original_price": row["original_price"],
                    "sold_price": (
                        row["sold_price"]
                        if not pd.isna(row.get("sold_price", float("nan")))
                        else None
                    ),
                    "sold_via_bargain": row.get("sold_via_bargain"),
                    "category_id": row["category_id"],
                    "category_name": self._data.category_dict.get(str(row["category_id"]), ""),
                    "similarity_score": round(float(score), 4),
                })
                if len(results) >= top_k:
                    break

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return results