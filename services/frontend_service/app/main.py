from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from shared.schemas import QueryRequest, SearchResponse, EvaluationRequest, ClusteringRequest
from shared.config import settings
import uvicorn
import requests
import os

app = FastAPI(title="API Gateway & UI Service", version="1.0.0")

# Setup directories for HTML templates and static files (CSS/JS)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Mount static directory for CSS/JS
static_path = os.path.join(BASE_DIR, "static")
if not os.path.exists(static_path):
    os.makedirs(static_path)
    os.makedirs(os.path.join(static_path, "css"))
    os.makedirs(os.path.join(static_path, "js"))
app.mount("/static", StaticFiles(directory=static_path), name="static")

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "gateway_service"}

@app.get("/", response_class=HTMLResponse)
def index_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "settings": settings})

@app.post("/api/search", response_model=SearchResponse)
def search_documents(query_req: QueryRequest):
    try:
        query_text = query_req.query
        
        # Step 1: Query Refinement (if enabled)
        refined_query = query_text
        if query_req.use_additional_features:
            try:
                refine_resp = requests.post(
                    f"{settings.REFINEMENT_SERVICE_URL}/refine",
                    json={"query": query_text},
                    timeout=2.0
                )
                if refine_resp.status_code == 200:
                    refined_query = refine_resp.json().get("refined_query", query_text)
            except Exception as e:
                print(f"Refinement service unavailable: {e}")
        
        # Step 2: Text Preprocessing
        try:
            prep_resp = requests.post(
                f"{settings.PREPROCESSING_SERVICE_URL}/preprocess",
                json={"text": refined_query, "stem": False, "lemmatize": True},
                timeout=2.0
            )
            if prep_resp.status_code == 200:
                preprocessed_query = prep_resp.json().get("processed_text", refined_query)
        except Exception as e:
            print(f"Preprocessing service unavailable: {e}")
            preprocessed_query = refined_query
            
        # Step 3: Retrieval
        retrieval_req = query_req.model_copy(update={"query": preprocessed_query})
        
        retrieval_resp = requests.post(
            f"{settings.RETRIEVAL_SERVICE_URL}/search",
            json=retrieval_req.model_dump(),
            timeout=5.0
        )
        
        if retrieval_resp.status_code != 200:
            raise HTTPException(status_code=retrieval_resp.status_code, detail="Error retrieving search results")
            
        search_data = retrieval_resp.json()
        search_data["refined_query"] = refined_query
        
        return search_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/evaluate")
def evaluate_system(eval_req: EvaluationRequest):
    try:
        resp = requests.post(
            f"{settings.EVALUATION_SERVICE_URL}/evaluate",
            json=eval_req.model_dump(),
            timeout=5.0
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail="Error during evaluation")
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/cluster")
def build_clusters(cluster_req: ClusteringRequest):
    try:
        resp = requests.post(
            f"{settings.CLUSTERING_SERVICE_URL}/cluster/build",
            json=cluster_req.model_dump(),
            timeout=5.0
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail="Error starting clustering")
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=settings.GATEWAY_PORT, reload=True)
