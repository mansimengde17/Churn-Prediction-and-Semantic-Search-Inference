from fastapi import APIRouter, HTTPException
from loguru import logger

from backend.schemas import SearchRequest, SearchResponse, SearchResult
from src.inference.pipeline import get_pipeline

router = APIRouter(prefix="/search", tags=["Semantic Search"])


@router.post("/", response_model=SearchResponse, summary="Semantic customer search")
async def semantic_search(request: SearchRequest):
    try:
        pipeline = get_pipeline()
        raw_results = pipeline.search(request.query, top_k=request.top_k)

        results = []
        for r in raw_results:
            results.append(SearchResult(
                customer_id=r.get("customer_id", ""),
                similarity_score=r.get("similarity_score", 0.0),
                churn_probability=r.get("churn_probability"),
                risk_level=r.get("risk_level"),
                tenure=r.get("tenure"),
                contract=r.get("Contract"),
                monthly_charges=r.get("MonthlyCharges"),
                internet_service=r.get("InternetService"),
            ))

        return SearchResponse(
            query=request.query,
            results=results,
            total=len(results),
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/similar/{customer_id}", summary="Find similar customers by ID")
async def find_similar(customer_id: str, top_k: int = 5):
    try:
        pipeline = get_pipeline()
        results = pipeline.find_similar(customer_id, top_k=top_k)
        return {"customer_id": customer_id, "similar": results, "total": len(results)}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Similarity search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
