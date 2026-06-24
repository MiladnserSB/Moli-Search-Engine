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
    print("[Startup] Preloading QueryRefiner (Spellcheck, Synonyms, Trie)...")
    refiner = QueryRefiner()
    # Preload autocomplete caches for both datasets at startup
    for dataset in ["quora_dev", "lotte_lifestyle_dev"]:
        try:
            print(f"[Startup] Warming up autocomplete cache for {dataset}...")
            refiner.suggest_queries("a", dataset)
        except Exception as e:
            print(f"[Startup] Warning: autocomplete warm up failed for {dataset}: {e}")

class RefineRequest(BaseModel):
    query: str

class RefineResponse(BaseModel):
    original_query: str
    refined_query: str
    suggestions: list[str]

class AutocompleteRequest(BaseModel):
    prefix: str
    dataset: str

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "query_refinement_service"}

@app.post("/refine", response_model=RefineResponse)
def refine_query(request: RefineRequest):
    try:
        query = request.query
        # expand_query does spelling check + WordNet synonyms
        refined = refiner.expand_query(query)
        
        # Suggestions contain spelling correction if different
        corrected_spelling = refiner.correct_spelling(query)
        suggestions = [corrected_spelling] if corrected_spelling != query else []
        
        return RefineResponse(
            original_query=query,
            refined_query=refined,
            suggestions=suggestions
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/autocomplete")
def autocomplete_query(request: AutocompleteRequest):
    try:
        suggestions = refiner.suggest_queries(request.prefix, request.dataset)
        return {"suggestions": suggestions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=settings.REFINEMENT_PORT, reload=True)
