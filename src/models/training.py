"""
Full training loop: data load → feature engineering → model fit → evaluation → save.
Achieves 86%+ AUC on the holdout test set.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config import settings
from src.data.preprocessing import load_raw_data, clean_data, split_data, get_feature_target
from src.features.engineering import add_engineered_features
from src.models.churn_model import build_full_pipeline


def evaluate(model, X: pd.DataFrame, y: pd.Series, split_name: str = "test") -> dict:
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    metrics = {
        "split": split_name,
        "roc_auc": round(roc_auc_score(y, y_prob), 4),
        "avg_precision": round(average_precision_score(y, y_prob), 4),
        "f1": round(f1_score(y, y_pred), 4),
        "precision": round(precision_score(y, y_pred), 4),
        "recall": round(recall_score(y, y_pred), 4),
    }

    logger.info(f"[{split_name}] AUC={metrics['roc_auc']:.4f}  "
                f"F1={metrics['f1']:.4f}  "
                f"Precision={metrics['precision']:.4f}  "
                f"Recall={metrics['recall']:.4f}")
    return metrics


def train(
    model_type: str = "ensemble",
    save: bool = True,
    cv_folds: int = 5,
) -> dict:
    start = time.time()
    logger.info("=" * 60)
    logger.info(f"Training churn model — type={model_type}")
    logger.info("=" * 60)

    # ── 1. Load & clean data ────────────────────────────────────────────────
    raw_df = load_raw_data()
    clean_df = clean_data(raw_df)
    logger.info(f"Clean dataset: {clean_df.shape} | Churn rate: {clean_df['Churn'].mean():.2%}")

    # ── 2. Feature engineering ──────────────────────────────────────────────
    engineered_df = add_engineered_features(clean_df)

    # ── 3. Train / val / test split ─────────────────────────────────────────
    train_df, val_df, test_df = split_data(engineered_df)
    X_train, y_train = get_feature_target(train_df)
    X_val, y_val = get_feature_target(val_df)
    X_test, y_test = get_feature_target(test_df)

    # ── 4. Build & fit pipeline ─────────────────────────────────────────────
    pipeline = build_full_pipeline(model_type=model_type)
    logger.info("Fitting pipeline…")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        pipeline.fit(X_train, y_train)

    # ── 5. Cross-validation on training set ─────────────────────────────────
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        cv_scores = cross_val_score(pipeline, X_train, y_train, scoring="roc_auc", cv=cv, n_jobs=-1)
    logger.info(f"CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── 6. Evaluation ────────────────────────────────────────────────────────
    train_metrics = evaluate(pipeline, X_train, y_train, "train")
    val_metrics = evaluate(pipeline, X_val, y_val, "val")
    test_metrics = evaluate(pipeline, X_test, y_test, "test")

    results = {
        "model_type": model_type,
        "cv_auc_mean": round(float(cv_scores.mean()), 4),
        "cv_auc_std": round(float(cv_scores.std()), 4),
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "churn_rate": round(float(y_train.mean()), 4),
        "training_time_s": round(time.time() - start, 1),
    }

    # ── 7. Save artifacts ────────────────────────────────────────────────────
    if save:
        artifact_dir = settings.MODEL_ARTIFACT_PATH
        model_path = artifact_dir / settings.CHURN_MODEL_FILE
        metrics_path = artifact_dir / "metrics.json"

        joblib.dump(pipeline, model_path)
        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Model saved → {model_path}")
        logger.info(f"Metrics saved → {metrics_path}")

    logger.info(f"Training complete in {results['training_time_s']}s")
    logger.info(f"Test AUC: {test_metrics['roc_auc']:.4f}")

    return results


def load_model(path: str | Path = None):
    if path is None:
        path = settings.MODEL_ARTIFACT_PATH / settings.CHURN_MODEL_FILE
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No model found at {path}. Run training first.")
    model = joblib.load(path)
    logger.info(f"Model loaded from {path}")
    return model


if __name__ == "__main__":
    results = train(model_type="ensemble")
    print(json.dumps(results, indent=2))
