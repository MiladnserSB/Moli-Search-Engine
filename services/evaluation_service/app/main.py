from fastapi import FastAPI, HTTPException
from shared.schemas import EvaluationRequest, EvaluationResponse
from shared.config import settings
import uvicorn

app = FastAPI(title="Evaluation Service", version="1.0.0")

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "evaluation_service"}

@app.post("/evaluate", response_model=EvaluationResponse)
def evaluate_results(request: EvaluationRequest):
    try:
        # Mock evaluation scores
        # In implementation: read qrels for request.dataset from settings.DATASETS_DIR
        # and compute standard metrics.
        return EvaluationResponse(
            dataset=request.dataset,
            map_score=0.784,
            recall_score=0.821,
            precision_at_k=0.700,
            ndcg_score=0.812
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=settings.EVALUATION_PORT, reload=True)
