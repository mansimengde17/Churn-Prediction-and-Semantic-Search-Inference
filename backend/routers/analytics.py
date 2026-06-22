from fastapi import APIRouter, HTTPException
from loguru import logger

from backend.schemas import MetricsResponse, HealthResponse
from src.inference.pipeline import get_pipeline

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/metrics", response_model=MetricsResponse, summary="Model performance metrics")
async def get_metrics():
    try:
        pipeline = get_pipeline()
        m = pipeline.model_metrics
        if not m:
            raise HTTPException(status_code=404, detail="Metrics not found. Train the model first.")
        return MetricsResponse(
            model_type=m.get("model_type", "unknown"),
            test_auc=m["test"]["roc_auc"],
            test_f1=m["test"]["f1"],
            test_precision=m["test"]["precision"],
            test_recall=m["test"]["recall"],
            cv_auc_mean=m.get("cv_auc_mean", 0.0),
            cv_auc_std=m.get("cv_auc_std", 0.0),
            train_rows=m.get("train_rows", 0),
            test_rows=m.get("test_rows", 0),
            churn_rate=m.get("churn_rate", 0.0),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse, summary="Service health check")
async def health_check():
    pipeline = get_pipeline()
    return HealthResponse(
        status="healthy",
        model_loaded=pipeline._churn_model is not None,
        search_loaded=pipeline._search_engine is not None,
        version="1.0.0",
    )


@router.get("/risk-distribution", summary="Get churn risk distribution from the dataset")
async def risk_distribution():
    try:
        import json
        from src.config import settings
        from pathlib import Path

        metrics_path = settings.MODEL_ARTIFACT_PATH / "metrics.json"
        if not metrics_path.exists():
            raise HTTPException(status_code=404, detail="Run training first")

        with open(metrics_path) as f:
            metrics = json.load(f)

        return {
            "churn_rate": metrics.get("churn_rate", 0.0),
            "test_auc": metrics["test"]["roc_auc"],
            "precision": metrics["test"]["precision"],
            "recall": metrics["test"]["recall"],
            "f1": metrics["test"]["f1"],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
