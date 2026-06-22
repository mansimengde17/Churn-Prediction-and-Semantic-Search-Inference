"""
Semantic search inference layer using sentence-transformers.
Converts customer profiles into dense vector embeddings and performs
cosine similarity search against an in-memory or pgvector index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

from src.config import settings


def customer_to_text(row: dict | pd.Series) -> str:
    """Serialize a customer record into a natural-language description for embedding."""
    return (
        f"Customer profile: {row.get('gender', 'Unknown')} "
        f"{'senior citizen' if row.get('SeniorCitizen', 0) == 1 else 'non-senior'}, "
        f"tenure {row.get('tenure', 0)} months, "
        f"contract type {row.get('Contract', 'unknown')}, "
        f"internet service {row.get('InternetService', 'none')}, "
        f"monthly charges ${row.get('MonthlyCharges', 0):.2f}, "
        f"payment via {row.get('PaymentMethod', 'unknown')}, "
        f"partner {'yes' if row.get('Partner', 0) == 1 else 'no'}, "
        f"dependents {'yes' if row.get('Dependents', 0) == 1 else 'no'}, "
        f"tech support {'yes' if row.get('TechSupport', 'No') == 'Yes' else 'no'}, "
        f"streaming {'yes' if row.get('StreamingTV', 'No') == 'Yes' else 'no'}."
    )


def query_to_text(query: str) -> str:
    """Pass-through — natural language queries are embedded as-is."""
    return query


class SemanticSearchEngine:
    """
    In-memory semantic search engine backed by sentence-transformers.
    For production scale: replace `_index` with pgvector queries via
    src.data.connectors.postgresql.PostgreSQLConnector.similarity_search().
    """

    def __init__(self, model_name: str = None):
        self._model_name = model_name or settings.EMBEDDING_MODEL
        self._model = None
        self._index: np.ndarray | None = None       # shape (N, D)
        self._metadata: list[dict] = []             # parallel list of record dicts
        self._customer_ids: list[str] = []

    def _load_model(self):
        if self._model is None:
            logger.info(f"Loading embedding model: {self._model_name}")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            logger.info("Embedding model loaded")

    def embed(self, texts: list[str], batch_size: int = 64, normalize: bool = True) -> np.ndarray:
        self._load_model()
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 200,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def build_index(self, df: pd.DataFrame, customer_id_col: str = "customerID"):
        """Embed all customer profiles and build the in-memory index."""
        texts = [customer_to_text(row) for _, row in df.iterrows()]
        logger.info(f"Building index for {len(texts):,} customers…")

        embeddings = self.embed(texts)
        self._index = embeddings
        self._customer_ids = df[customer_id_col].tolist() if customer_id_col in df.columns else [
            f"CUST-{i}" for i in range(len(df))
        ]
        self._metadata = df.to_dict("records")

        logger.info(f"Index built: shape={self._index.shape}")

    def save_index(self, path: str | Path = None):
        path = Path(path or settings.MODEL_ARTIFACT_PATH / "search_index.npz")
        np.savez_compressed(
            path,
            embeddings=self._index,
            customer_ids=np.array(self._customer_ids),
        )
        logger.info(f"Index saved → {path}")

    def load_index(self, path: str | Path = None, df: pd.DataFrame = None):
        path = Path(path or settings.MODEL_ARTIFACT_PATH / "search_index.npz")
        if not path.exists():
            raise FileNotFoundError(f"No index at {path}. Call build_index() first.")
        data = np.load(path, allow_pickle=True)
        self._index = data["embeddings"]
        self._customer_ids = data["customer_ids"].tolist()
        if df is not None:
            self._metadata = df.set_index("customerID").loc[self._customer_ids].to_dict("records")
        logger.info(f"Index loaded: {self._index.shape}")

    def search(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float = 0.0,
    ) -> list[dict]:
        """
        Search for customers semantically similar to a natural-language query.
        Returns a list of dicts with customer metadata + similarity score.
        """
        if self._index is None:
            raise RuntimeError("Index not built. Call build_index() or load_index() first.")

        query_emb = self.embed([query_to_text(query)])  # (1, D)
        # Cosine similarity = dot product of normalised vectors
        scores = (self._index @ query_emb.T).squeeze()  # (N,)

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < score_threshold:
                break
            record = self._metadata[idx].copy() if self._metadata else {}
            record["customer_id"] = self._customer_ids[idx]
            record["similarity_score"] = round(score, 4)
            results.append(record)

        return results

    def find_similar_customers(
        self,
        customer_id: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Find customers with similar profiles to a given customer."""
        if customer_id not in self._customer_ids:
            raise ValueError(f"Customer {customer_id!r} not in index.")
        idx = self._customer_ids.index(customer_id)
        query_emb = self._index[idx : idx + 1]

        scores = (self._index @ query_emb.T).squeeze()
        top_indices = np.argsort(scores)[::-1]

        results = []
        for i in top_indices:
            if self._customer_ids[i] == customer_id:
                continue
            record = self._metadata[i].copy() if self._metadata else {}
            record["customer_id"] = self._customer_ids[i]
            record["similarity_score"] = round(float(scores[i]), 4)
            results.append(record)
            if len(results) >= top_k:
                break

        return results
