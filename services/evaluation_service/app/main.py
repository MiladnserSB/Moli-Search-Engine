"""
Evaluation Service -- IR System
=================================
Measures MAP, Recall, Precision@10, and nDCG for each retrieval model.

Modes:
  - OFFLINE (default): Instantly returns pre-computed results from cache JSON.
                       Base uses ALL queries (5000 Quora / 2076 Lotte).
                       Enhanced uses ALL queries (pre-computed with spell correction).
  - ONLINE:  Runs live retrieval through the Gateway for `online_limit` queries (default 500).
             Applies spell correction for Enhanced mode.

Before/After Advanced Features comparison:
  - BASE     (use_additional_features=False): retrieval without query refinement
  - ENHANCED (use_additional_features=True):  spell-corrected + synonym-expanded queries
"""
import os
import sys
import json
import sqlite3
import requests
from fastapi import FastAPI, HTTPException
from .schemas import EvaluationRequest, EvaluationResponse
from .config import settings
from .evaluator import IREvaluator
import uvicorn

# Fix Windows console encoding - prevents UnicodeEncodeError
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

app = FastAPI(title="Evaluation Service", version="2.0.0")


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "evaluation_service", "version": "2.0.0"}


# ==============================================================================
#  HELPER: Load evaluation cache (base + enhanced per model)
# ==============================================================================

def load_cache() -> dict:
    """Load the pre-computed evaluation results JSON."""
    if os.path.exists(settings.CACHE_PATH):
        try:
            with open(settings.CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Evaluation] Cache read error: {e}")
    return {}


def save_cache(cache_data: dict):
    """Write the evaluation results back to disk."""
    try:
        with open(settings.CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=4)
        print(f"[Evaluation] Cache saved to: {settings.CACHE_PATH}")
    except Exception as e:
        print(f"[Evaluation] Cache write error: {e}")


# ==============================================================================
#  HELPER: Spell-correct a batch of queries via refinement service
# ==============================================================================

def apply_spell_correction_batch(queries: list, dataset: str) -> list:
    """
    Calls the refinement service to spell-correct each query.
    Returns corrected queries (falls back to originals if service unavailable).
    Processes in chunks of 50 to avoid overwhelming the service.
    """
    corrected = list(queries)
    CHUNK = 50
    for start in range(0, len(queries), CHUNK):
        chunk = queries[start:start + CHUNK]
        for i, q in enumerate(chunk):
            try:
                resp = requests.post(
                    f"{settings.REFINEMENT_SERVICE_URL}/refine",
                    json={"query": q, "dataset": dataset, "user_id": "eval_system"},
                    timeout=2.0
                )
                if resp.status_code == 200:
                    data = resp.json()
                    enhanced_q = data.get("expanded_query") or data.get("corrected_query") or q
                    corrected[start + i] = enhanced_q.strip() or q
            except Exception:
                pass
    return corrected


# ==============================================================================
#  OFFLINE EVALUATION: Serve from pre-computed cache
# ==============================================================================

def serve_offline(db_dataset: str, method: str, is_enhanced: bool) -> EvaluationResponse | None:
    """
    Attempt to serve results from the pre-computed cache.
    Returns EvaluationResponse if found, None otherwise.

    Cache structure:
    {
      "quora_dev": {
        "bm25": {
          "base":     { precision_at_k, recall_score, map_score, ndcg_score, num_queries },
          "enhanced": { ... }
        }, ...
      }, ...
    }
    """
    cache_data = load_cache()
    mode_key = "enhanced" if is_enhanced else "base"

    if db_dataset in cache_data and method in cache_data[db_dataset]:
        model_entry = cache_data[db_dataset][method]

        # Support both old flat format and new base/enhanced format
        if isinstance(model_entry, dict):
            if mode_key in model_entry:
                metrics = model_entry[mode_key]
            elif "map_score" in model_entry:
                # Old flat format (only base) - acceptable for base mode
                if not is_enhanced:
                    metrics = model_entry
                else:
                    return None  # No enhanced entry in old format
            else:
                return None
        else:
            return None

        num_q = metrics.get("num_queries", None)
        mode_label = (
            "Enhanced (Spell Correction + Synonym Expansion) -- Offline Full Dataset"
            if is_enhanced
            else "Base (No Additional Features) -- Offline Full Dataset"
        )
        print(f"[Evaluation] [OFFLINE-OK] {db_dataset} | {method} | {mode_key} | "
              f"n={num_q} | MAP={metrics.get('map_score', 0):.4f} | nDCG={metrics.get('ndcg_score', 0):.4f}")

        return EvaluationResponse(
            dataset=db_dataset,
            map_score=round(metrics.get("map_score", 0.0), 4),
            recall_score=round(metrics.get("recall_score", 0.0), 4),
            precision_at_k=round(metrics.get("precision_at_k", 0.0), 4),
            ndcg_score=round(metrics.get("ndcg_score", 0.0), 4),
            num_queries_evaluated=num_q,
            mode=mode_label
        )
    return None


# ==============================================================================
#  ONLINE EVALUATION: Live retrieval via Gateway
# ==============================================================================

def run_online_evaluation(
    db_dataset: str,
    method: str,
    is_enhanced: bool,
    online_limit: int,
    request: EvaluationRequest
) -> EvaluationResponse:
    """
    Runs live evaluation by:
    1. Loading queries + qrels from DB
    2. (Optional) Applying spell correction for Enhanced mode
    3. Sending batch to Gateway -> Retrieval service
    4. Computing MAP, Recall, P@10, nDCG
    """
    print(f"[Evaluation] [ONLINE] Connecting to DB: {settings.DB_PATH}")
    conn = sqlite3.connect(settings.DB_PATH)
    c = conn.cursor()

    # Load queries for this dataset
    c.execute("SELECT query_id, text FROM queries WHERE dataset_name = ?", (db_dataset,))
    queries_dict = {str(row[0]): row[1] for row in c.fetchall()}

    # Load FULL qrels (supervisor requirement: use ALL qrels)
    c.execute("SELECT query_id, doc_id, relevance FROM qrels WHERE dataset_name = ?", (db_dataset,))
    qrels = {}
    for qid, doc_id, relevance in c.fetchall():
        qid_str = str(qid)
        if qid_str in queries_dict:
            qrels.setdefault(qid_str, {})[str(doc_id)] = int(relevance)
    conn.close()

    if not queries_dict:
        raise HTTPException(status_code=404, detail=f"No queries found for dataset: {db_dataset}")

    # Subset queries to online_limit for speed
    qids = list(queries_dict.keys())
    limit = min(online_limit, len(qids))
    qids = qids[:limit]

    total_available = len(queries_dict)
    print(f"[Evaluation] [ONLINE] Dataset={db_dataset} | Using {len(qids)} of {total_available} queries "
          f"({'Enhanced' if is_enhanced else 'Base'} mode)")

    # Filter qrels to subset
    filtered_qrels = {qid: qrels[qid] for qid in qids if qid in qrels}
    evaluator = IREvaluator(filtered_qrels)

    # Build query texts
    query_texts = [queries_dict[qid] for qid in qids]

    # Apply spell correction for Enhanced mode
    if is_enhanced:
        print(f"[Evaluation] [ONLINE-REFINE] Applying spell correction to {len(query_texts)} queries...")
        query_texts = apply_spell_correction_batch(query_texts, db_dataset)
    else:
        print(f"[Evaluation] [ONLINE-BASE] Using raw queries without spell correction")

    # Send batch to Gateway
    print(f"[Evaluation] [ONLINE-HTTP] Sending {len(qids)} queries to Gateway -> Retrieval...")
    batch_payload = {
        "queries": query_texts,
        "dataset": db_dataset,
        "method": method,
        "top_k": 10,
        "use_additional_features": False,
        "bm25_k1": request.bm25_k1,
        "bm25_b": request.bm25_b,
        "hybrid_alpha": request.hybrid_alpha,
        "retrieve_text": False
    }

    gateway_batch_url = f"{settings.GATEWAY_URL}/api/search/batch"
    try:
        resp = requests.post(gateway_batch_url, json=batch_payload, timeout=300.0)
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Gateway batch request timed out (300s). Try a smaller online_limit.")

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Batch search failed [{resp.status_code}]: {resp.text[:400]}"
        )

    batch_response = resp.json()
    results_list = batch_response.get("results", [])

    if len(results_list) != len(qids):
        raise HTTPException(
            status_code=500,
            detail=f"Result count mismatch: expected {len(qids)}, got {len(results_list)}"
        )

    # Build retrieved runs dict
    retrieved_runs = {}
    for qid, q_res in zip(qids, results_list):
        doc_ids = [str(r["id"]) for r in q_res.get("results", [])]
        retrieved_runs[qid] = doc_ids

    # Compute metrics
    p10_list, recall_list, ap_list, ndcg_list = [], [], [], []
    for qid, doc_ids in retrieved_runs.items():
        p10_list.append(evaluator.calculate_precision_at_k(doc_ids, qid, k=10))
        recall_list.append(evaluator.calculate_recall(doc_ids, qid))
        ap_list.append(evaluator.calculate_ap(doc_ids, qid, k=10))
        ndcg_list.append(evaluator.calculate_ndcg(doc_ids, qid, k=10))

    n = len(p10_list)
    avg_p10    = sum(p10_list)    / n if n else 0.0
    avg_recall = sum(recall_list) / n if n else 0.0
    avg_map    = sum(ap_list)     / n if n else 0.0
    avg_ndcg   = sum(ndcg_list)   / n if n else 0.0

    mode_label = (
        f"Enhanced (Spell Correction) -- Online Live | n={n}/{total_available}"
        if is_enhanced
        else f"Base -- Online Live | n={n}/{total_available}"
    )
    print(f"[Evaluation] [ONLINE-OK] {method} | {db_dataset} | "
          f"{'Enhanced' if is_enhanced else 'Base'} | n={n} | "
          f"MAP={avg_map:.4f}  Recall={avg_recall:.4f}  P@10={avg_p10:.4f}  nDCG={avg_ndcg:.4f}")

    return EvaluationResponse(
        dataset=db_dataset,
        map_score=round(avg_map, 4),
        recall_score=round(avg_recall, 4),
        precision_at_k=round(avg_p10, 4),
        ndcg_score=round(avg_ndcg, 4),
        num_queries_evaluated=n,
        mode=mode_label
    )


# ==============================================================================
#  MAIN EVALUATE ENDPOINT
# ==============================================================================

@app.post("/evaluate", response_model=EvaluationResponse)
def evaluate_results(request: EvaluationRequest):
    """
    Main evaluation endpoint.

    Request fields:
      - dataset:               "quora_dev" or "lotte_lifestyle_dev"
      - method:                "bm25", "vsm", "embedding", "hybrid_serial", "hybrid_parallel"
      - use_additional_features: false = BASE, true = ENHANCED (with spell correction)
      - mode:                  "offline" (instant, from cache) or "online" (live, slower)
      - online_limit:          number of queries for online mode (default 500)
      - bm25_k1, bm25_b, hybrid_alpha: optional custom parameters (only used in online mode)

    Response:
      - Standard IR metrics: map_score, recall_score, precision_at_k, ndcg_score
      - num_queries_evaluated: how many queries were actually evaluated
      - mode: description of what was run
    """
    try:
        ds = request.dataset.lower()
        method = request.method.lower()
        db_dataset = "quora_dev" if "quora" in ds else "lotte_lifestyle_dev"
        is_enhanced = request.use_additional_features
        eval_mode = request.mode.lower() if request.mode else "offline"

        print(f"[Evaluation] Request: dataset={db_dataset} | method={method} | "
              f"enhanced={is_enhanced} | mode={eval_mode} | online_limit={request.online_limit}")

        # ── OFFLINE MODE: Serve from pre-computed cache ───────────────────────
        if eval_mode == "offline":
            result = serve_offline(db_dataset, method, is_enhanced)
            if result:
                return result
            else:
                # Fallback: if offline cache is missing, run online with warning
                print(f"[Evaluation] [WARN] Offline cache missing for {db_dataset}/{method}/"
                      f"{'enhanced' if is_enhanced else 'base'} -- falling back to online (500 queries)")
                return run_online_evaluation(db_dataset, method, is_enhanced, 500, request)

        # ── ONLINE MODE: Live retrieval ───────────────────────────────────────
        elif eval_mode == "online":
            online_limit = max(1, min(request.online_limit, 5000))
            return run_online_evaluation(db_dataset, method, is_enhanced, online_limit, request)

        else:
            raise HTTPException(status_code=400, detail=f"Invalid mode '{eval_mode}'. Use 'offline' or 'online'.")

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=settings.EVALUATION_PORT, reload=True)
