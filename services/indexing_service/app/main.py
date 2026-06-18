from fastapi import FastAPI, HTTPException, BackgroundTasks
from .config import settings
from .schemas import BM25BuildRequest, BuildResponse
from .indexer import Indexer
import uvicorn
import os
import joblib
import json

app = FastAPI(title="Indexing & Representation Service", version="1.0.0")

# Simple in-memory tracker for indexing status (keys: "{dataset_name}_{rep_type}")
index_status = {}

# Global dictionary to cache all loaded models/indices in RAM on bootup
loaded_cache = {}

@app.on_event("startup")
def preload_and_cache_indices():
    """Scans and caches any pre-built models and indices on startup into memory."""
    print("--- Pre-loading and caching all built models into memory (8 GB RAM optimized) ---")
    datasets = ["quora_dev", "lotte_lifestyle_dev"]
    rep_types = ["tfidf", "embedding", "bm25", "hybrid", "inverted_index"]
    
    for dataset in datasets:
        # 1. TF-IDF
        tfidf_path = os.path.join(settings.MODELS_DIR, f"{dataset}_tfidf.joblib")
        if os.path.exists(tfidf_path):
            print(f"   -> Caching TF-IDF model for {dataset}...")
            loaded_cache[f"{dataset}_tfidf"] = joblib.load(tfidf_path)
            index_status[f"{dataset}_tfidf"] = "ready"
            
        # 2. BERT Embeddings
        emb_path = os.path.join(settings.MODELS_DIR, f"{dataset}_embedding.joblib")
        if os.path.exists(emb_path):
            print(f"   -> Caching BERT embeddings for {dataset}...")
            loaded_cache[f"{dataset}_embedding"] = joblib.load(emb_path)
            index_status[f"{dataset}_embedding"] = "ready"
            
        # 3. BM25 Okapi Index
        bm25_path = os.path.join(settings.MODELS_DIR, f"{dataset}_bm25.joblib")
        if os.path.exists(bm25_path):
            print(f"   -> Caching BM25 model for {dataset}...")
            loaded_cache[f"{dataset}_bm25"] = joblib.load(bm25_path)
            index_status[f"{dataset}_bm25"] = "ready"
            
        # 4. Parallel Hybrid Vectors
        hybrid_path = os.path.join(settings.MODELS_DIR, f"{dataset}_hybrid.joblib")
        if os.path.exists(hybrid_path):
            print(f"   -> Caching Hybrid vectors for {dataset}...")
            loaded_cache[f"{dataset}_hybrid"] = joblib.load(hybrid_path)
            index_status[f"{dataset}_hybrid"] = "ready"
            
        # 5. Inverted Index (JSON)
        inv_path = os.path.join(settings.INDICES_DIR, f"{dataset}_index.json")
        if os.path.exists(inv_path):
            print(f"   -> Caching Inverted Index for {dataset}...")
            with open(inv_path, "r", encoding="utf-8") as f:
                loaded_cache[f"{dataset}_inverted_index"] = json.load(f)
            index_status[f"{dataset}_inverted_index"] = "ready"
            
    print("[OK] Pre-loading and caching of models complete.")

def build_task_worker(dataset_name: str, rep_type: str, k1: float = 1.5, b: float = 0.75):
    task_key = f"{dataset_name}_{rep_type}"
    index_status[task_key] = "building"
    try:
        indexer = Indexer(dataset_name)
        if rep_type == "tfidf":
            res = indexer.build_tfidf()
            # Cache it immediately after build
            loaded_cache[task_key] = joblib.load(res["file"])
        elif rep_type == "embedding":
            res = indexer.build_embedding()
            # Cache it immediately after build
            loaded_cache[task_key] = joblib.load(res["file"])
        elif rep_type == "bm25":
            res = indexer.build_bm25(k1=k1, b=b)
            # Cache it immediately after build
            loaded_cache[task_key] = joblib.load(res["file"])
        elif rep_type == "hybrid":
            res = indexer.build_hybrid()
            # Cache it immediately after build
            loaded_cache[task_key] = joblib.load(res["file"])
        elif rep_type == "inverted_index":
            res = indexer.build_inverted_index()
            # Cache it immediately after build
            with open(res["file"], "r", encoding="utf-8") as f:
                loaded_cache[task_key] = json.load(f)
        else:
            raise ValueError(f"Unknown representation type: {rep_type}")
        
        index_status[task_key] = "ready"
    except Exception as e:
        import traceback
        traceback.print_exc()
        index_status[task_key] = f"error: {str(e)}"

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "indexing_service"}

@app.post("/index/build/tfidf/{dataset_name}", response_model=BuildResponse)
def build_tfidf(dataset_name: str, background_tasks: BackgroundTasks):
    task_key = f"{dataset_name}_tfidf"
    if index_status.get(task_key) == "building":
        return BuildResponse(status="building", message="TF-IDF building is already in progress.", dataset=dataset_name)
        
    background_tasks.add_task(build_task_worker, dataset_name, "tfidf")
    return BuildResponse(status="started", message="TF-IDF representation task queued.", dataset=dataset_name)

@app.post("/index/build/embedding/{dataset_name}", response_model=BuildResponse)
def build_embedding(dataset_name: str, background_tasks: BackgroundTasks):
    task_key = f"{dataset_name}_embedding"
    if index_status.get(task_key) == "building":
        return BuildResponse(status="building", message="BERT embedding building is already in progress.", dataset=dataset_name)
        
    background_tasks.add_task(build_task_worker, dataset_name, "embedding")
    return BuildResponse(status="started", message="BERT embedding task queued.", dataset=dataset_name)

@app.post("/index/build/bm25/{dataset_name}", response_model=BuildResponse)
def build_bm25(dataset_name: str, request: BM25BuildRequest, background_tasks: BackgroundTasks):
    task_key = f"{dataset_name}_bm25"
    if index_status.get(task_key) == "building":
        return BuildResponse(status="building", message="BM25 building is already in progress.", dataset=dataset_name)
        
    background_tasks.add_task(build_task_worker, dataset_name, "bm25", request.k1, request.b)
    return BuildResponse(status="started", message="BM25 index building task queued.", dataset=dataset_name)

@app.post("/index/build/hybrid/{dataset_name}", response_model=BuildResponse)
def build_hybrid(dataset_name: str, background_tasks: BackgroundTasks):
    task_key = f"{dataset_name}_hybrid"
    if index_status.get(task_key) == "building":
        return BuildResponse(status="building", message="Hybrid vector building is already in progress.", dataset=dataset_name)
        
    background_tasks.add_task(build_task_worker, dataset_name, "hybrid")
    return BuildResponse(status="started", message="Parallel hybrid building task queued.", dataset=dataset_name)

@app.post("/index/build/inverted_index/{dataset_name}", response_model=BuildResponse)
def build_inverted_index(dataset_name: str, background_tasks: BackgroundTasks):
    task_key = f"{dataset_name}_inverted_index"
    if index_status.get(task_key) == "building":
        return BuildResponse(status="building", message="Inverted index building is already in progress.", dataset=dataset_name)
        
    background_tasks.add_task(build_task_worker, dataset_name, "inverted_index")
    return BuildResponse(status="started", message="Inverted index building task queued.", dataset=dataset_name)

@app.get("/index/status/{dataset_name}/{rep_type}")
def get_index_status(dataset_name: str, rep_type: str):
    task_key = f"{dataset_name}_{rep_type}"
    status = index_status.get(task_key, "not_built")
    return {"dataset": dataset_name, "representation": rep_type, "status": status}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=settings.INDEXING_PORT, reload=True)
