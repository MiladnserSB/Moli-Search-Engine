# from fastapi import FastAPI, HTTPException
# from shared.schemas import QueryRequest, SearchResponse, DocumentSchema
# from shared.config import settings
# import uvicorn
# import time

# app = FastAPI(title="Retrieval Service", version="1.0.0")

# @app.get("/health")
# def health_check():
#     return {"status": "healthy", "service": "retrieval_service"}

# @app.post("/search", response_model=SearchResponse)
# def search_documents(request: QueryRequest):
#     start_time = time.time()
#     try:
#         # Mock documents retrieval response
#         mock_results = [
#             DocumentSchema(
#                 id="doc_001",
#                 title="Introduction to Information Retrieval",
#                 content="This document covers basic concepts of IR including indexing and vector space model.",
#                 score=0.92
#             ),
#             DocumentSchema(
#                 id="doc_002",
#                 title="Understanding BM25",
#                 content="BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms.",
#                 score=0.85
#             ),
#             DocumentSchema(
#                 id="doc_003",
#                 title="Dense Retrieval & Vector Stores",
#                 content="Dense passage retrieval uses neural networks to map text into a vector space for similarity search.",
#                 score=0.74
#             )
#         ]
        
#         # Sort mock results by score
#         mock_results = sorted(mock_results, key=lambda x: x.score, reverse=True)[:request.top_k]
        
#         time_taken_ms = (time.time() - start_time) * 1000
        
#         return SearchResponse(
#             results=mock_results,
#             refined_query=request.query,
#             time_taken_ms=time_taken_ms
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=settings.RETRIEVAL_PORT, reload=True)

from fastapi import FastAPI, HTTPException
import sys
import types

# Dynamically define the missing builder module to enable successful joblib/pickle unpickling
if 'app.vsm_tfidf_builder' not in sys.modules:
    fake_module = types.ModuleType('app.vsm_tfidf_builder')
    fake_module.whitespace_tokenizer = lambda text: str(text).split()
    sys.modules['app.vsm_tfidf_builder'] = fake_module

from .schemas import QueryRequest, SearchResponse, DocumentSchema, BatchQueryRequest, BatchSearchResponse, BatchQueryResult
from .config import settings
from .search_models import TFIDFSearcher, BM25Searcher
from .vector_search import VectorSearcher
from .hybrid import HybridSearcher
from sentence_transformers import SentenceTransformer

import os
import gc
import time
import sqlite3
import uvicorn

app = FastAPI(
    title="Retrieval Service",
    version="1.0.0"
)

# 8 GB RAM Limit Compliant Model Manager
class ModelManager:
    def __init__(self):
        # Cache loaded components by dataset name to avoid loading latency
        self.index_cache = {}
        self.vsm_cache = {}
        self.bm25_cache = {}
        self.vector_cache = {}
        self.bert_model = None

    def get_searchers(self, dataset: str, method: str):
        # Load inverted index if needed and not cached
        if method in ['vsm', 'bm25', 'hybrid_serial', 'hybrid_parallel']:
            if dataset not in self.index_cache:
                index_path = os.path.join(settings.INDICES_DIR, f"{dataset}_index.json")
                if os.path.exists(index_path):
                    print(f"[ModelManager] Loading Inverted Index for {dataset}...")
                    import json
                    with open(index_path, "r", encoding="utf-8") as f:
                        self.index_cache[dataset] = json.load(f)
                else:
                    print(f"[ModelManager] Warning: Inverted Index file not found at {index_path}")
                    self.index_cache[dataset] = None

        # 2. Lazy load the required models for the query method
        # TF-IDF
        if method == 'vsm':
            if dataset not in self.vsm_cache:
                model_path = os.path.join(settings.MODELS_DIR, f"{dataset}_tfidf.joblib")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"TF-IDF model file not found at: {model_path}")
                print(f"[ModelManager] Loading TF-IDF model for {dataset}...")
                self.vsm_cache[dataset] = TFIDFSearcher(model_path)
            return self.vsm_cache[dataset], None

        # BM25
        elif method == 'bm25':
            if dataset not in self.bm25_cache:
                model_path = os.path.join(settings.MODELS_DIR, f"{dataset}_bm25.joblib")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"BM25 model file not found at: {model_path}")
                print(f"[ModelManager] Loading BM25 model for {dataset}...")
                self.bm25_cache[dataset] = BM25Searcher(model_path)
            return self.bm25_cache[dataset], None

        # BERT Embeddings
        elif method == 'embedding':
            if dataset not in self.vector_cache:
                model_path = os.path.join(settings.MODELS_DIR, f"{dataset}_embedding.joblib")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"BERT embeddings file not found at: {model_path}")
                print(f"[ModelManager] Loading BERT embeddings for {dataset}...")
                vector_searcher = VectorSearcher()
                vector_searcher.load_index(model_path)
                self.vector_cache[dataset] = vector_searcher
            if self.bert_model is None:
                print("[ModelManager] Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
                self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
            return self.vector_cache[dataset], self.bert_model

        # Hybrid Serial / Hybrid Parallel
        elif method in ['hybrid_serial', 'hybrid_parallel']:
            if dataset not in self.bm25_cache:
                model_path = os.path.join(settings.MODELS_DIR, f"{dataset}_bm25.joblib")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"BM25 model file not found at: {model_path}")
                print(f"[ModelManager] Loading BM25 model for {dataset}...")
                self.bm25_cache[dataset] = BM25Searcher(model_path)
                
            if dataset not in self.vector_cache:
                model_path = os.path.join(settings.MODELS_DIR, f"{dataset}_embedding.joblib")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"BERT embeddings file not found at: {model_path}")
                print(f"[ModelManager] Loading BERT embeddings for {dataset}...")
                vector_searcher = VectorSearcher()
                vector_searcher.load_index(model_path)
                self.vector_cache[dataset] = vector_searcher
                
            if self.bert_model is None:
                print("[ModelManager] Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
                self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
                
            hybrid_searcher = HybridSearcher(self.bm25_cache[dataset], self.vector_cache[dataset])
            return hybrid_searcher, self.bert_model

        else:
            raise ValueError(f"Unknown retrieval method: {method}")

model_manager = ModelManager()

@app.on_event("startup")
def preload_models():
    print("--- [Startup] Preloading all search indexes and models into RAM (8 GB RAM compliant) ---")
    datasets = ["quora_dev", "lotte_lifestyle_dev"]
    methods = ["bm25", "vsm", "embedding"]
    for dataset in datasets:
        for method in methods:
            try:
                print(f"[Startup] Preloading {dataset} | {method}...")
                model_manager.get_searchers(dataset, method)
            except Exception as e:
                print(f"[Startup] Error preloading {dataset} | {method}: {e}")
    print("--- [Startup] Preloading complete! Retrieval Service is fully warmed up ---")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "retrieval_service"
    }

@app.post("/search")
def search_documents(request: QueryRequest):
    start_time = time.time()
    try:
        method = request.method.lower()
        dataset = request.dataset
        
        # 1. Resolve searcher and BERT model dynamically
        searcher, bert_model = model_manager.get_searchers(dataset, method)
        
        # 2. Execute matching and ranking
        raw_results = []
        inverted_idx = model_manager.index_cache.get(dataset)
        
        # Precompute embedding with history-based vector fusion if needed
        query_emb = None
        fused_info = None
        if method in ['embedding', 'hybrid_serial', 'hybrid_parallel']:
            import numpy as np
            query_emb = bert_model.encode(request.query, convert_to_numpy=True, normalize_embeddings=True)
            query_emb = query_emb.astype('float32')
            
            if request.use_additional_features:
                user_id = request.user_id or "default_user"
                try:
                    conn = sqlite3.connect(settings.DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT DISTINCT query FROM search_history 
                        WHERE user_id = ? AND dataset_name = ?
                        ORDER BY timestamp DESC LIMIT 5
                    ''', (user_id, dataset))
                    rows = cursor.fetchall()
                    conn.close()
                    
                    history_queries = [row[0] for row in rows if row[0].strip().lower() != request.query.strip().lower()]
                    if history_queries:
                        history_embs = [bert_model.encode(hq, convert_to_numpy=True, normalize_embeddings=True) for hq in history_queries]
                        user_interest_vector = np.mean(history_embs, axis=0)
                        user_interest_vector = user_interest_vector / np.linalg.norm(user_interest_vector)
                        
                        fusion_alpha = 0.7
                        query_emb = fusion_alpha * query_emb + (1.0 - fusion_alpha) * user_interest_vector
                        query_emb = query_emb / np.linalg.norm(query_emb)
                        query_emb = query_emb.astype('float32')
                        
                        fused_info = {
                            "historical_queries": history_queries,
                            "alpha": fusion_alpha
                        }
                        print(f"[VectorFusion] Query vector fused with user interests from history: {history_queries}")
                except Exception as ex:
                    print(f"[VectorFusion] Error during history vector fusion: {ex}")
        
        if method == 'vsm':
            raw_results = searcher.search(request.query, top_k=request.top_k, inverted_index=inverted_idx)
        elif method == 'bm25':
            query_tokens = request.query.split()
            raw_results = searcher.search(
                query_tokens, 
                top_k=request.top_k, 
                k1=request.bm25_k1, 
                b=request.bm25_b,
                inverted_index=inverted_idx
            )
        elif method == 'embedding':
            raw_results = searcher.search(query_emb, top_k=request.top_k)
        elif method == 'hybrid_serial':
            bm25_query = request.preprocessed_query if request.preprocessed_query else request.query
            query_tokens = bm25_query.split()
            # serial: sparse candidates, then dense rerank
            # BM25 parameter overrides are applied inside the cached searcher if given
            bm25_searcher = model_manager.bm25_cache.get(dataset)
            if bm25_searcher:
                if request.bm25_k1 is not None:
                    bm25_searcher.bm25.k1 = request.bm25_k1
                if request.bm25_b is not None:
                    bm25_searcher.bm25.b = request.bm25_b
            raw_results = searcher.search_serial(query_tokens, query_emb, top_k=request.top_k, inverted_index=inverted_idx)
        elif method == 'hybrid_parallel':
            bm25_query = request.preprocessed_query if request.preprocessed_query else request.query
            query_tokens = bm25_query.split()
            # parallel: sparse + dense, combined with RRF
            bm25_searcher = model_manager.bm25_cache.get(dataset)
            if bm25_searcher:
                if request.bm25_k1 is not None:
                    bm25_searcher.bm25.k1 = request.bm25_k1
                if request.bm25_b is not None:
                    bm25_searcher.bm25.b = request.bm25_b
            raw_results = searcher.search_parallel(query_tokens, query_emb, top_k=request.top_k, inverted_index=inverted_idx, hybrid_alpha=request.hybrid_alpha)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported method: {method}")
            
        # 3. Retrieve raw uncleaned document contents from SQLite database
        if not raw_results:
            time_taken_ms = (time.time() - start_time) * 1000
            return {
                "results": [],
                "refined_query": request.query,
                "personalized_fusion_info": fused_info,
                "time_taken_ms": time_taken_ms
            }
            
        doc_ids = [str(r["id"]) for r in raw_results]
        scores_dict = {str(r["id"]): r["score"] for r in raw_results}
        
        if not request.retrieve_text:
            # Skip SQLite text fetching during evaluations (MAP, Recall, etc. only need IDs)
            results = [
                DocumentSchema(
                    id=d_id,
                    content="",
                    score=scores_dict.get(d_id, 0.0)
                )
                for d_id in doc_ids
            ]
            time_taken_ms = (time.time() - start_time) * 1000
            return {
                "results": [r.model_dump() for r in results],
                "refined_query": request.query,
                "personalized_fusion_info": fused_info,
                "time_taken_ms": time_taken_ms
            }
            
        conn = sqlite3.connect(settings.DB_PATH)
        cursor = conn.cursor()
        
        placeholders = ",".join(["?"] * len(doc_ids))
        query = f"SELECT doc_id, text FROM documents WHERE dataset_name = ? AND doc_id IN ({placeholders})"
        cursor.execute(query, [dataset] + doc_ids)
        rows = cursor.fetchall()
        conn.close()
        
        docs_text_dict = {str(row[0]): row[1] for row in rows}
        
        # Sort documents exactly according to retrieval rank order
        results = []
        for d_id in doc_ids:
            text = docs_text_dict.get(d_id, "[Document text not found in database]")
            score = scores_dict.get(d_id, 0.0)
            results.append(
                DocumentSchema(
                    id=d_id,
                    content=text,
                    score=score
                )
            )
            
        time_taken_ms = (time.time() - start_time) * 1000
        return {
            "results": [r.model_dump() for r in results],
            "refined_query": request.query,
            "personalized_fusion_info": fused_info,
            "time_taken_ms": time_taken_ms
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/search/batch", response_model=BatchSearchResponse)
def search_documents_batch(request: BatchQueryRequest):
    start_time = time.time()
    try:
        method = request.method.lower()
        dataset = request.dataset
        
        # 1. Resolve searcher and BERT model dynamically
        searcher, bert_model = model_manager.get_searchers(dataset, method)
        
        inverted_idx = model_manager.index_cache.get(dataset)
        
        # Determine queries and preprocessed representation
        queries = request.queries
        preprocessed_queries = request.preprocessed_queries if request.preprocessed_queries else queries
        
        raw_results_list = []
        
        if method == 'vsm':
            raw_results_list = searcher.search_batch(
                preprocessed_queries,
                top_k=request.top_k,
                inverted_index=inverted_idx
            )
        elif method == 'bm25':
            queries_tokens = [q.split() for q in preprocessed_queries]
            raw_results_list = searcher.search_batch(
                queries_tokens,
                top_k=request.top_k,
                k1=request.bm25_k1,
                b=request.bm25_b,
                inverted_index=inverted_idx
            )
        elif method == 'embedding':
            # Encode all queries at once
            query_embs = bert_model.encode(queries, convert_to_numpy=True, normalize_embeddings=True)
            query_embs = query_embs.astype('float32')
            raw_results_list = searcher.search_batch(
                query_embs,
                top_k=request.top_k
            )
        elif method == 'hybrid_serial':
            queries_tokens = [q.split() for q in preprocessed_queries]
            query_embs = bert_model.encode(queries, convert_to_numpy=True, normalize_embeddings=True)
            query_embs = query_embs.astype('float32')
            
            bm25_searcher = model_manager.bm25_cache.get(dataset)
            if bm25_searcher:
                if request.bm25_k1 is not None:
                    bm25_searcher.bm25.k1 = request.bm25_k1
                if request.bm25_b is not None:
                    bm25_searcher.bm25.b = request.bm25_b
                    
            raw_results_list = searcher.search_batch_serial(
                queries_tokens,
                query_embs,
                top_k=request.top_k,
                inverted_index=inverted_idx
            )
        elif method == 'hybrid_parallel':
            queries_tokens = [q.split() for q in preprocessed_queries]
            query_embs = bert_model.encode(queries, convert_to_numpy=True, normalize_embeddings=True)
            query_embs = query_embs.astype('float32')
            
            bm25_searcher = model_manager.bm25_cache.get(dataset)
            if bm25_searcher:
                if request.bm25_k1 is not None:
                    bm25_searcher.bm25.k1 = request.bm25_k1
                if request.bm25_b is not None:
                    bm25_searcher.bm25.b = request.bm25_b
                    
            raw_results_list = searcher.search_batch_parallel(
                queries_tokens,
                query_embs,
                top_k=request.top_k,
                inverted_index=inverted_idx,
                hybrid_alpha=request.hybrid_alpha
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported method: {method}")
            
        # 3. Post-process results (retrieve text if needed)
        batch_results = []
        
        # Build document text database cache if retrieve_text is True and we have documents
        all_doc_ids = set()
        if request.retrieve_text:
            for raw_results in raw_results_list:
                for r in raw_results:
                    all_doc_ids.add(str(r["id"]))
                    
        docs_text_dict = {}
        if request.retrieve_text and all_doc_ids:
            conn = sqlite3.connect(settings.DB_PATH)
            cursor = conn.cursor()
            placeholders = ",".join(["?"] * len(all_doc_ids))
            query = f"SELECT doc_id, text FROM documents WHERE dataset_name = ? AND doc_id IN ({placeholders})"
            cursor.execute(query, [dataset] + list(all_doc_ids))
            rows = cursor.fetchall()
            conn.close()
            docs_text_dict = {str(row[0]): row[1] for row in rows}
            
        for q_idx, raw_results in enumerate(raw_results_list):
            doc_ids = [str(r["id"]) for r in raw_results]
            scores_dict = {str(r["id"]): r["score"] for r in raw_results}
            
            results = []
            for d_id in doc_ids:
                text = docs_text_dict.get(d_id, "") if request.retrieve_text else ""
                score = scores_dict.get(d_id, 0.0)
                results.append(
                    DocumentSchema(
                        id=d_id,
                        content=text,
                        score=score
                    )
                )
            batch_results.append(
                BatchQueryResult(
                    query=queries[q_idx],
                    results=results
                )
            )
            
        time_taken_ms = (time.time() - start_time) * 1000
        return BatchSearchResponse(
            results=batch_results,
            time_taken_ms=time_taken_ms
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.RETRIEVAL_PORT,
        reload=True
    )

