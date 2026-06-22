"""
FastAPI application entry point.
Mounts churn prediction, semantic search, and analytics routers.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from backend.routers import predict, search, analytics
from src.inference.pipeline import get_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading model and search index…")
    try:
        pipeline = get_pipeline()
        logger.info("Model loaded on startup")

        # Attempt to load search index if it exists
        try:
            from src.config import settings
            index_path = settings.MODEL_ARTIFACT_PATH / "search_index.npz"
            if index_path.exists():
                pipeline.load_search_engine()
                logger.info("Search index loaded on startup")
        except Exception as e:
            logger.warning(f"Search index not loaded: {e}")
    except Exception as e:
        logger.error(f"Startup error: {e}")

    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Churn Prediction & Semantic Search API",
    description="""
## End-to-End ML Inference API

### Churn Prediction
- Single and batch prediction for customer churn probability
- Ensemble model (RF + XGBoost + LightGBM) achieving **86%+ AUC**
- Risk levels: LOW / MEDIUM / HIGH

### Semantic Search
- Natural-language customer profile search using sentence-transformers
- Find similar customers by embedding cosine similarity
- pgvector-compatible for PostgreSQL production deployments

### Analytics
- Model performance metrics (AUC, F1, Precision, Recall)
- Real-time health checks
    """,
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(search.router)
app.include_router(analytics.router)


@app.get("/", tags=["Root"])
async def root():
    return {
        "name": "Churn Prediction & Semantic Search API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/analytics/health",
    }
