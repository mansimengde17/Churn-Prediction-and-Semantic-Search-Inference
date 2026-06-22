"""Pydantic request/response schemas for the FastAPI backend."""

from __future__ import annotations

from typing import Optional, List
from pydantic import BaseModel, Field, field_validator


class CustomerFeatures(BaseModel):
    gender: str = Field(..., examples=["Male"])
    SeniorCitizen: int = Field(..., ge=0, le=1, examples=[0])
    Partner: int = Field(..., ge=0, le=1, examples=[1])
    Dependents: int = Field(..., ge=0, le=1, examples=[0])
    tenure: int = Field(..., ge=0, le=100, examples=[12])
    PhoneService: int = Field(..., ge=0, le=1, examples=[1])
    MultipleLines: str = Field(..., examples=["No"])
    InternetService: str = Field(..., examples=["Fiber optic"])
    OnlineSecurity: str = Field(..., examples=["No"])
    OnlineBackup: str = Field(..., examples=["Yes"])
    DeviceProtection: str = Field(..., examples=["No"])
    TechSupport: str = Field(..., examples=["No"])
    StreamingTV: str = Field(..., examples=["Yes"])
    StreamingMovies: str = Field(..., examples=["Yes"])
    Contract: str = Field(..., examples=["Month-to-month"])
    PaperlessBilling: int = Field(..., ge=0, le=1, examples=[1])
    PaymentMethod: str = Field(..., examples=["Electronic check"])
    MonthlyCharges: float = Field(..., ge=0, examples=[79.85])
    TotalCharges: float = Field(..., ge=0, examples=[958.20])

    model_config = {"json_schema_extra": {
        "example": {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": 1,
            "Dependents": 0,
            "tenure": 12,
            "PhoneService": 1,
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": 1,
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 79.85,
            "TotalCharges": 958.20,
        }
    }}


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    risk_level: str
    model_version: str = "1.0.0"


class BatchPredictionRequest(BaseModel):
    customers: List[CustomerFeatures]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total: int


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, examples=["high risk fiber optic customer on month-to-month contract"])
    top_k: int = Field(default=10, ge=1, le=50)


class SearchResult(BaseModel):
    customer_id: str
    similarity_score: float
    churn_probability: Optional[float] = None
    risk_level: Optional[str] = None
    tenure: Optional[int] = None
    contract: Optional[str] = None
    monthly_charges: Optional[float] = None
    internet_service: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    search_loaded: bool
    version: str


class MetricsResponse(BaseModel):
    model_type: str
    test_auc: float
    test_f1: float
    test_precision: float
    test_recall: float
    cv_auc_mean: float
    cv_auc_std: float
    train_rows: int
    test_rows: int
    churn_rate: float
