from fastapi import APIRouter, HTTPException
from loguru import logger

from backend.schemas import (
    CustomerFeatures,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
)
from src.inference.pipeline import get_pipeline

router = APIRouter(prefix="/predict", tags=["Churn Prediction"])


@router.post("/", response_model=PredictionResponse, summary="Predict churn for a single customer")
async def predict_single(customer: CustomerFeatures):
    try:
        pipeline = get_pipeline()
        result = pipeline.predict(customer.model_dump())
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=BatchPredictionResponse, summary="Batch churn prediction")
async def predict_batch(request: BatchPredictionRequest):
    try:
        pipeline = get_pipeline()
        predictions = []
        for c in request.customers:
            result = pipeline.predict(c.model_dump())
            predictions.append(PredictionResponse(**result))
        return BatchPredictionResponse(predictions=predictions, total=len(predictions))
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
