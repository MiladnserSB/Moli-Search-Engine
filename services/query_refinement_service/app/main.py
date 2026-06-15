from fastapi import FastAPI, HTTPException
from shared.config import settings
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Query Refinement Service", version="1.0.0")

class RefineRequest(BaseModel):
    query: str

class RefineResponse(BaseModel):
    original_query: str
    refined_query: str
    suggestions: list[str]

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "query_refinement_service"}

@app.post("/refine", response_model=RefineResponse)
def refine_query(request: RefineRequest):
    try:
        query = request.query
        # Simple mockup of spell check / suggestions
        refined = query # If correct, same
        suggestions = [query + " definition", query + " tutorial"]
        
        return RefineResponse(
            original_query=query,
            refined_query=refined,
            suggestions=suggestions
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=settings.REFINEMENT_PORT, reload=True)
