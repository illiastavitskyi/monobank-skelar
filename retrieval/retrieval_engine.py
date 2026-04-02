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
        top_k: int = config.RAG_TOP_K,
    ) -> list[dict]:
        """Return top-k comparable listings for the query."""
        t0 = time.perf_counter()

        scores, indices = self._backend.search(query, top_k=top_k)

        df = self._data.df
        results: list[dict] = []
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

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return results