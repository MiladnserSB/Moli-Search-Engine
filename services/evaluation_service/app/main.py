import os
import json
import sqlite3
import requests
from fastapi import FastAPI, HTTPException
from .schemas import EvaluationRequest, EvaluationResponse
from .config import settings
from .evaluator import IREvaluator
import uvicorn

app = FastAPI(title="Evaluation Service", version="1.0.0")

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "evaluation_service"}

@app.post("/evaluate", response_model=EvaluationResponse)
def evaluate_results(request: EvaluationRequest):
    try:
        ds = request.dataset.lower()
        method = request.method.lower()
        db_dataset = "quora_dev" if "quora" in ds else "lotte_lifestyle_dev"
        
        # Check if the request is dynamic (custom slider parameters or advanced features active)
        is_dynamic = (
            (request.bm25_k1 is not None and abs(request.bm25_k1 - 1.5) > 0.01) or
            (request.bm25_b is not None and abs(request.bm25_b - 0.75) > 0.01) or
            (request.hybrid_alpha is not None and abs(request.hybrid_alpha - 0.5) > 0.01) or
            request.use_additional_features
        )
        
        # Check cache first only for non-dynamic (standard default) runs
        if not is_dynamic:
            cache_path = settings.CACHE_PATH
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        cache_data = json.load(f)
                    if db_dataset in cache_data and method in cache_data[db_dataset]:
                        metrics = cache_data[db_dataset][method]
                        print(f"[Evaluation] Returning cached default results for {db_dataset} | {method}")
                        return EvaluationResponse(
                            dataset=request.dataset,
                            map_score=round(metrics.get("map_score", 0.0), 4),
                            recall_score=round(metrics.get("recall_score", 0.0), 4),
                            precision_at_k=round(metrics.get("precision_at_k", 0.0), 4),
                            ndcg_score=round(metrics.get("ndcg_score", 0.0), 4)
                        )
                except Exception as cache_ex:
                    print(f"[Evaluation] Cache read warning: {cache_ex}")
        
        print(f"[Evaluation] Connecting to DB: {settings.DB_PATH}")
        conn = sqlite3.connect(settings.DB_PATH)
        c = conn.cursor()
        
        # Load queries
        c.execute("SELECT query_id, text FROM queries WHERE dataset_name = ?", (db_dataset,))
        queries = {str(row[0]): row[1] for row in c.fetchall()}
        
        # Load relevance judgements for these queries
        c.execute("SELECT query_id, doc_id, relevance FROM qrels WHERE dataset_name = ?", (db_dataset,))
        qrels = {}
        for qid, doc_id, relevance in c.fetchall():
            qid_str = str(qid)
            if qid_str in queries:
                qrels.setdefault(qid_str, {})[str(doc_id)] = int(relevance)
        conn.close()
        
        if not queries:
            raise HTTPException(status_code=404, detail=f"No queries found in DB for dataset {db_dataset}")
            
        # Determine subset of queries to evaluate
        qids = list(queries.keys())
        if is_dynamic:
            if db_dataset == "lotte_lifestyle_dev":
                # Lotte lifestyle has only ~1000 queries, which runs in <10 seconds for all models
                pass
            else: # quora_dev (5000 queries)
                if method in ['vsm', 'bm25']:
                    # Classical VSM and BM25 run extremely fast (matrix math) - evaluate all 5000 queries
                    pass
                else:
                    # Neural/Hybrid models on CPU: subset to 500 queries for faster dynamic responses
                    qids = qids[:500]
                    print(f"[Evaluation] Dynamic neural evaluation on Quora. Subsetting to first {len(qids)} queries for <10s performance.")
            
        # Filter qrels to match the query subset
        filtered_qrels = {qid: qrels[qid] for qid in qids if qid in qrels}
        evaluator = IREvaluator(filtered_qrels)
        retrieved_runs = {}
        gateway_batch_url = f"{settings.GATEWAY_URL}/api/search/batch"
        
        # Prepare queries texts list
        query_texts = [queries[qid] for qid in qids]
        
        print(f"[Evaluation] Sending {len(qids)} queries in a batch request to Gateway...")
        
        batch_payload = {
            "queries": query_texts,
            "dataset": db_dataset,
            "method": method,
            "top_k": 10,
            "use_additional_features": request.use_additional_features,
            "bm25_k1": request.bm25_k1,
            "bm25_b": request.bm25_b,
            "hybrid_alpha": request.hybrid_alpha,
            "retrieve_text": False
        }
        
        resp = requests.post(gateway_batch_url, json=batch_payload, timeout=120.0)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=f"Batch search failed: {resp.text}")
            
        batch_response = resp.json()
        results_list = batch_response.get("results", [])
        
        if len(results_list) != len(qids):
            raise HTTPException(
                status_code=500,
                detail=f"Mismatched batch search results count: expected {len(qids)}, got {len(results_list)}"
            )
        
        for qid, q_res in zip(qids, results_list):
            doc_ids = [str(r["id"]) for r in q_res.get("results", [])]
            retrieved_runs[qid] = doc_ids

                
        # 3. Calculate metrics dynamically on the fly
        p10_list = []
        recall_list = []
        ap_list = []
        ndcg_list = []
        
        for qid in retrieved_runs:
            p10_list.append(evaluator.calculate_precision_at_k(retrieved_runs[qid], qid, k=10))
            recall_list.append(evaluator.calculate_recall(retrieved_runs[qid], qid))
            ap_list.append(evaluator.calculate_ap(retrieved_runs[qid], qid, k=10))
            ndcg_list.append(evaluator.calculate_ndcg(retrieved_runs[qid], qid, k=10))
            
        avg_p10 = sum(p10_list) / len(p10_list) if p10_list else 0.0
        avg_recall = sum(recall_list) / len(recall_list) if recall_list else 0.0
        avg_map = sum(ap_list) / len(ap_list) if ap_list else 0.0
        avg_ndcg = sum(ndcg_list) / len(ndcg_list) if ndcg_list else 0.0
        
        print(f"[Evaluation] Dynamic results calculated successfully!")
        
        # Only cache results if this was a standard non-dynamic run
        if not is_dynamic:
            try:
                cache_data = {}
                if os.path.exists(settings.CACHE_PATH):
                    with open(settings.CACHE_PATH, "r", encoding="utf-8") as f:
                        cache_data = json.load(f)
                cache_data.setdefault(db_dataset, {})
                cache_data[db_dataset][method] = {
                    "precision_at_k": round(avg_p10, 4),
                    "recall_score": round(avg_recall, 4),
                    "map_score": round(avg_map, 4),
                    "ndcg_score": round(avg_ndcg, 4)
                }
                with open(settings.CACHE_PATH, "w", encoding="utf-8") as f:
                    json.dump(cache_data, f, indent=4)
                print(f"[Evaluation] Successfully wrote results to cache: {settings.CACHE_PATH}")
            except Exception as cache_write_ex:
                print(f"[Evaluation] Cache write warning: {cache_write_ex}")
            
        return EvaluationResponse(
            dataset=request.dataset,
            map_score=round(avg_map, 4),
            recall_score=round(avg_recall, 4),
            precision_at_k=round(avg_p10, 4),
            ndcg_score=round(avg_ndcg, 4)
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=settings.EVALUATION_PORT, reload=True)
