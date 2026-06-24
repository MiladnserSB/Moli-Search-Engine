from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .config import settings
from .refiner import QueryRefiner
import uvicorn

app = FastAPI(title="Query Refinement Service", version="1.0.0")

refiner = None


@app.on_event("startup")
def load_refiner_model():
    global refiner
    print("[Startup] Preloading QueryRefiner...")
    refiner = QueryRefiner()
    # تحمية الكاش للاستعلامات التلقائية لأشهر المجموعات
    for dataset in ["quora_dev", "lotte_lifestyle_dev"]:
        try:
            print(f"[Startup] Warming up autocomplete cache for {dataset}...")
            refiner.suggest_queries("a", dataset)
        except Exception as e:
            print(f"[Startup] Warning: warm up failed for {dataset}: {e}")


class RefineRequest(BaseModel):
    query: str
    dataset: str = "quora_dev"
    user_id: str = "default_user"


class RefineResponse(BaseModel):
    original_query: str
    corrected_query: str
    expanded_query: str
    personalized_query: str
    alternative_queries: list[str]


class AutocompleteRequest(BaseModel):
    prefix: str
    dataset: str
    user_id: str = "default_user"


class LogRequest(BaseModel):
    query: str
    dataset: str
    user_id: str = "default_user"


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "query_refinement_service"}


@app.post("/log")
def log_user_query(request: LogRequest):
    try:
        refiner.log_query(request.query, request.dataset, request.user_id)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/refine", response_model=RefineResponse)
def refine_query(request: RefineRequest):
    try:
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty.")

        # التصحيح الإملائي
        corrected = refiner.correct_spelling(query)
        # التوسيع بالمرادفات (يستخدم التصحيح داخلياً)
        expanded = refiner.expand_query(query)
        
        # التوسيع بالمرادفات + تاريخ المستخدم
        personalized = refiner.expand_query_with_history(query, request.dataset, request.user_id)

        # استعلامات بديلة
        alternative_queries = refiner.generate_alternative_queries(
            query, max_alternatives=3
        )

        return RefineResponse(
            original_query=query,
            corrected_query=corrected,
            expanded_query=expanded,
            personalized_query=personalized,
            alternative_queries=alternative_queries,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/autocomplete")
def autocomplete_query(request: AutocompleteRequest):
    try:
        suggestions = refiner.suggest_queries(request.prefix, request.dataset, request.user_id)
        return {"suggestions": suggestions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=settings.REFINEMENT_PORT, reload=True)
