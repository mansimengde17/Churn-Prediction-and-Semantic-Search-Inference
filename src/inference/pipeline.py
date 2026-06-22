"""
Production inference pipeline.
Loads the trained model + search index and exposes predict() and search() methods.
Thread-safe singleton pattern for use in FastAPI.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.config import settings
from src.features.engineering import add_engineered_features


_lock = threading.Lock()
_instance: Optional["InferencePipeline"] = None


class InferencePipeline:
    def __init__(self):
        self._churn_model = None
        self._search_engine = None
        self._metrics: dict = {}

    # ── Churn prediction ──────────────────────────────────────────────────────

    def load_churn_model(self):
        from src.models.training import load_model
        import json

        self._churn_model = load_model()

        metrics_path = settings.MODEL_ARTIFACT_PATH / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                self._metrics = json.load(f)

        logger.info("Churn model loaded into inference pipeline")

    def predict(self, features: dict | pd.DataFrame) -> dict:
        if self._churn_model is None:
            self.load_churn_model()

        if isinstance(features, dict):
            df = pd.DataFrame([features])
        else:
            df = features.copy()

        df = add_engineered_features(df)
        proba = self._churn_model.predict_proba(df)[:, 1]
        pred = (proba >= 0.5).astype(int)

        if len(proba) == 1:
            return {
                "churn_probability": round(float(proba[0]), 4),
                "churn_prediction": int(pred[0]),
                "risk_level": _risk_level(float(proba[0])),
            }

        return pd.DataFrame({
            "churn_probability": proba.round(4),
            "churn_prediction": pred,
            "risk_level": [_risk_level(p) for p in proba],
        }).to_dict("records")

    def batch_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._churn_model is None:
            self.load_churn_model()

        enriched = add_engineered_features(df)
        probas = self._churn_model.predict_proba(enriched)[:, 1]
        preds = (probas >= 0.5).astype(int)

        result = df.copy()
        result["churn_probability"] = probas.round(4)
        result["churn_prediction"] = preds
        result["risk_level"] = [_risk_level(p) for p in probas]
        return result

    # ── Semantic search ───────────────────────────────────────────────────────

    def load_search_engine(self, df: pd.DataFrame = None, force_rebuild: bool = False):
        from src.search.semantic_search import SemanticSearchEngine

        self._search_engine = SemanticSearchEngine()
        index_path = settings.MODEL_ARTIFACT_PATH / "search_index.npz"

        if not force_rebuild and index_path.exists():
            self._search_engine.load_index(index_path, df)
        elif df is not None:
            self._search_engine.build_index(df)
            self._search_engine.save_index(index_path)
        else:
            logger.warning("No data or index provided for search engine. Search unavailable.")

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        if self._search_engine is None:
            raise RuntimeError("Search engine not loaded. Call load_search_engine() first.")
        return self._search_engine.search(query, top_k=top_k)

    def find_similar(self, customer_id: str, top_k: int = 5) -> list[dict]:
        if self._search_engine is None:
            raise RuntimeError("Search engine not loaded.")
        return self._search_engine.find_similar_customers(customer_id, top_k=top_k)

    @property
    def model_metrics(self) -> dict:
        return self._metrics


def get_pipeline() -> InferencePipeline:
    """Return singleton inference pipeline (thread-safe)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = InferencePipeline()
                _instance.load_churn_model()
    return _instance


def _risk_level(prob: float) -> str:
    if prob >= 0.70:
        return "HIGH"
    elif prob >= 0.40:
        return "MEDIUM"
    return "LOW"
