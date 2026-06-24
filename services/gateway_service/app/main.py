from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas import QueryRequest, SearchResponse, EvaluationRequest, ClusteringRequest, BatchQueryRequest, BatchSearchResponse
from .config import settings
import uvicorn
import requests
import os

app = FastAPI(title="API Gateway Service", version="1.0.0")

# Enable CORS for frontend cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "gateway_service"}

@app.post("/api/search", response_model=SearchResponse)
def search_documents(query_req: QueryRequest):
    try:
        query_text = query_req.query
        
        # Log query to history database
        try:
            requests.post(
                f"{settings.REFINEMENT_SERVICE_URL}/log",
                json={"query": query_text, "dataset": query_req.dataset, "user_id": "default_user"},
                timeout=1.0
            )
        except Exception as e:
            print(f"Failed to log query to history service: {e}")
            
        # Step 1: Query Refinement (if enabled)
        refined_query = query_text
        personalized_query = None
        if query_req.use_additional_features:
            try:
                refine_resp = requests.post(
                    f"{settings.REFINEMENT_SERVICE_URL}/refine",
                    json={"query": query_text, "dataset": query_req.dataset, "user_id": "default_user"},
                    timeout=2.0
                )
                if refine_resp.status_code == 200:
                    refine_json = refine_resp.json()
                    # Use corrected_query as refined_query so search runs on clean query
                    refined_query = refine_json.get("corrected_query", query_text)
                    personalized_query = refine_json.get("personalized_query", None)
            except Exception as e:
                print(f"Refinement service unavailable: {e}")
        
        # Step 2: Text Preprocessing
        method = query_req.method.lower()
        preprocessed_query = refined_query
        preprocessed_query_classical = None
        
        try:
            if method in ['vsm', 'bm25']:
                # Classical preprocessing (stemming & lemmatization to match indices)
                prep_resp = requests.post(
                    f"{settings.PREPROCESSING_SERVICE_URL}/preprocess",
                    json={"text": refined_query, "stem": True, "lemmatize": True, "pipeline_type": "classical"},
                    timeout=2.0
                )
                if prep_resp.status_code == 200:
                    preprocessed_query = prep_resp.json().get("processed_text", refined_query)
            elif method == 'embedding':
                # Neural preprocessing (preserves casing/stopwords for BERT)
                prep_resp = requests.post(
                    f"{settings.PREPROCESSING_SERVICE_URL}/preprocess",
                    json={"text": refined_query, "stem": False, "lemmatize": False, "pipeline_type": "neural"},
                    timeout=2.0
                )
                if prep_resp.status_code == 200:
                    preprocessed_query = prep_resp.json().get("processed_text", refined_query)
            elif method in ['hybrid_serial', 'hybrid_parallel']:
                # Hybrid needs both: classical for BM25 and neural/raw for BERT
                # 1. Classical
                prep_resp_c = requests.post(
                    f"{settings.PREPROCESSING_SERVICE_URL}/preprocess",
                    json={"text": refined_query, "stem": True, "lemmatize": True, "pipeline_type": "classical"},
                    timeout=2.0
                )
                if prep_resp_c.status_code == 200:
                    preprocessed_query_classical = prep_resp_c.json().get("processed_text", refined_query)
                
                # 2. Neural
                prep_resp_n = requests.post(
                    f"{settings.PREPROCESSING_SERVICE_URL}/preprocess",
                    json={"text": refined_query, "stem": False, "lemmatize": False, "pipeline_type": "neural"},
                    timeout=2.0
                )
                if prep_resp_n.status_code == 200:
                    preprocessed_query = prep_resp_n.json().get("processed_text", refined_query)
        except Exception as e:
            print(f"Preprocessing service unavailable: {e}")
            if method in ['hybrid_serial', 'hybrid_parallel']:
                preprocessed_query_classical = preprocessed_query_classical or refined_query
            
        # Step 3: Retrieval
        retrieval_payload = query_req.model_dump()
        retrieval_payload["query"] = preprocessed_query
        if preprocessed_query_classical:
            retrieval_payload["preprocessed_query"] = preprocessed_query_classical
            
        retrieval_resp = requests.post(
            f"{settings.RETRIEVAL_SERVICE_URL}/search",
            json=retrieval_payload,
            timeout=60.0
        )
        
        if retrieval_resp.status_code != 200:
            raise HTTPException(status_code=retrieval_resp.status_code, detail="Error retrieving search results")
            
        search_data = retrieval_resp.json()
        search_data["refined_query"] = refined_query
        search_data["personalized_query"] = personalized_query
        
        return search_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search/batch", response_model=BatchSearchResponse)
def search_documents_batch(query_req: BatchQueryRequest):
    try:
        queries = query_req.queries
        method = query_req.method.lower()
        
        refined_queries = queries
        preprocessed_queries = refined_queries
        preprocessed_queries_classical = None
        
        # Batch Preprocessing
        if method in ['vsm', 'bm25']:
            prep_resp = requests.post(
                f"{settings.PREPROCESSING_SERVICE_URL}/preprocess/batch",
                json={"texts": refined_queries, "stem": True, "lemmatize": True, "pipeline_type": "classical"},
                timeout=30.0
            )
            if prep_resp.status_code == 200:
                results_prep = prep_resp.json().get("results", [])
                preprocessed_queries = [r.get("processed_text", "") for r in results_prep]
        elif method == 'embedding':
            prep_resp = requests.post(
                f"{settings.PREPROCESSING_SERVICE_URL}/preprocess/batch",
                json={"texts": refined_queries, "stem": False, "lemmatize": False, "pipeline_type": "neural"},
                timeout=30.0
            )
            if prep_resp.status_code == 200:
                results_prep = prep_resp.json().get("results", [])
                preprocessed_queries = [r.get("processed_text", "") for r in results_prep]
        elif method in ['hybrid_serial', 'hybrid_parallel']:
            # Call classical batch preprocessing
            prep_resp_c = requests.post(
                f"{settings.PREPROCESSING_SERVICE_URL}/preprocess/batch",
                json={"texts": refined_queries, "stem": True, "lemmatize": True, "pipeline_type": "classical"},
                timeout=30.0
            )
            if prep_resp_c.status_code == 200:
                results_prep_c = prep_resp_c.json().get("results", [])
                preprocessed_queries_classical = [r.get("processed_text", "") for r in results_prep_c]
            
            # Call neural batch preprocessing
            prep_resp_n = requests.post(
                f"{settings.PREPROCESSING_SERVICE_URL}/preprocess/batch",
                json={"texts": refined_queries, "stem": False, "lemmatize": False, "pipeline_type": "neural"},
                timeout=30.0
            )
            if prep_resp_n.status_code == 200:
                results_prep_n = prep_resp_n.json().get("results", [])
                preprocessed_queries = [r.get("processed_text", "") for r in results_prep_n]
                
        # Call Retrieval Service batch endpoint
        retrieval_payload = query_req.model_dump()
        retrieval_payload["queries"] = preprocessed_queries
        if preprocessed_queries_classical:
            retrieval_payload["preprocessed_queries"] = preprocessed_queries_classical
            
        retrieval_resp = requests.post(
            f"{settings.RETRIEVAL_SERVICE_URL}/search/batch",
            json=retrieval_payload,
            timeout=120.0
        )
        
        if retrieval_resp.status_code != 200:
            raise HTTPException(status_code=retrieval_resp.status_code, detail="Error retrieving batch search results")
            
        return retrieval_resp.json()
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/evaluate")
def evaluate_system(eval_req: EvaluationRequest):
    try:
        resp = requests.post(
            f"{settings.EVALUATION_SERVICE_URL}/evaluate",
            json=eval_req.model_dump(),
            timeout=120.0
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

@app.post("/api/autocomplete")
def fetch_autocomplete(req: dict):
    try:
        resp = requests.post(
            f"{settings.REFINEMENT_SERVICE_URL}/autocomplete",
            json=req,
            timeout=10.0
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail="Error during autocomplete fetch")
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/refine")
def fetch_refinement(req: dict):
    try:
        resp = requests.post(
            f"{settings.REFINEMENT_SERVICE_URL}/refine",
            json=req,
            timeout=10.0
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail="Error during query refinement fetch")
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=settings.GATEWAY_PORT, reload=True)
