import os
import sys
import gc
import time
import json
import sqlite3
import joblib
import numpy as np
import types
import subprocess
import argparse

# Dynamically mock the indexing builder module to enable joblib unpickling
if 'app.vsm_tfidf_builder' not in sys.modules:
    fake_module = types.ModuleType('app.vsm_tfidf_builder')
    fake_module.whitespace_tokenizer = lambda text: str(text).split()
    sys.modules['app.vsm_tfidf_builder'] = fake_module

# Resolve imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "services", "preprocessing_service")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "services", "preprocessing_service", "app")))

from services.evaluation_service.app.evaluator import IREvaluator

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "ir_dataset_store.db"))
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "models"))
CACHE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "evaluation_results.json"))

def load_queries_and_qrels(dataset_name):
    print(f"\n[DB] Loading queries and qrels for {dataset_name}...")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Load queries
    c.execute("SELECT query_id, text FROM queries WHERE dataset_name = ?", (dataset_name,))
    queries = {str(row[0]): row[1] for row in c.fetchall()}
    
    # Load qrels
    c.execute("SELECT query_id, doc_id, relevance FROM qrels WHERE dataset_name = ?", (dataset_name,))
    qrels = {}
    for qid, doc_id, relevance in c.fetchall():
        qrels.setdefault(str(qid), {})[str(doc_id)] = int(relevance)
        
    conn.close()
    print(f"[DB] Loaded {len(queries)} queries and qrels for {len(qrels)} queries.")
    return queries, qrels

def run_classical_preprocessing(queries):
    from services.preprocessing_service.app.preprocessor import Preprocessor
    print("[Prep] Initializing text preprocessor...")
    preprocessor = Preprocessor()
    print("[Prep] Preprocessing queries (classical)...")
    t0 = time.time()
    raw_query_ids = list(queries.keys())
    raw_query_texts = [queries[qid] for qid in raw_query_ids]
    classical_results = preprocessor.preprocess_batch(raw_query_texts, pipeline_type="classical", stem=True, lemmatize=True, verbose=False)
    classical_queries = {raw_query_ids[i]: classical_results[i][0] for i in range(len(raw_query_ids))}
    print(f"[Prep] Preprocessed {len(raw_query_ids)} queries in {time.time() - t0:.2f} seconds.")
    return classical_queries

def run_neural_preprocessing(queries):
    from services.preprocessing_service.app.preprocessor import Preprocessor
    print("[Prep] Initializing text preprocessor...")
    preprocessor = Preprocessor()
    print("[Prep] Preprocessing queries (neural)...")
    t0 = time.time()
    raw_query_ids = list(queries.keys())
    raw_query_texts = [queries[qid] for qid in raw_query_ids]
    neural_results = preprocessor.preprocess_batch(raw_query_texts, pipeline_type="neural", verbose=False)
    neural_queries = {raw_query_ids[i]: neural_results[i][0] for i in range(len(raw_query_ids))}
    print(f"[Prep] Preprocessed {len(raw_query_ids)} queries in {time.time() - t0:.2f} seconds.")
    return neural_queries

def evaluate_vsm(dataset_name, queries, qrels):
    vsm_path = os.path.join(MODELS_DIR, f"{dataset_name}_tfidf.joblib")
    if not os.path.exists(vsm_path):
        print(f"[VSM] Warning: VSM model not found at {vsm_path}")
        return None
        
    classical_queries = run_classical_preprocessing(queries)
    
    print("\n[VSM] Loading VSM model...")
    vsm_data = joblib.load(vsm_path, mmap_mode='r')
    vectorizer = vsm_data["vectorizer"]
    doc_vectors = vsm_data["vectors"]
    doc_ids = vsm_data["doc_ids"]
    
    print("[VSM] Running batch VSM cosine matching (RAM-optimized batch size 5)...")
    batch_size = 5
    qids = list(classical_queries.keys())
    vsm_runs = {}
    
    for b_start in range(0, len(qids), batch_size):
        b_qids = qids[b_start : b_start + batch_size]
        b_texts = [classical_queries[qid] for qid in b_qids]
        b_vecs = vectorizer.transform(b_texts).astype(np.float32) # Shape: (batch_size, terms)
        scores = doc_vectors.dot(b_vecs.T).toarray() # Shape: (docs, batch_size)
        
        for idx_in_batch, qid in enumerate(b_qids):
            q_scores = scores[:, idx_in_batch]
            top_idx = np.argpartition(q_scores, -10)[-10:]
            top_idx = top_idx[np.argsort(q_scores[top_idx])[::-1]]
            vsm_runs[qid] = [str(doc_ids[idx_i]) for idx_i in top_idx]
            
    del vsm_data, doc_vectors
    gc.collect()
    
    print("[VSM] Calculating metrics...")
    evaluator = IREvaluator(qrels)
    ap_list, recall_list, p10_list, ndcg_list = [], [], [], []
    for qid in vsm_runs:
        p10_list.append(evaluator.calculate_precision_at_k(vsm_runs[qid], qid, k=10))
        recall_list.append(evaluator.calculate_recall(vsm_runs[qid], qid))
        ap_list.append(evaluator.calculate_ap(vsm_runs[qid], qid, k=10))
        ndcg_list.append(evaluator.calculate_ndcg(vsm_runs[qid], qid, k=10))
        
    avg_map = sum(ap_list) / len(ap_list) if ap_list else 0.0
    avg_recall = sum(recall_list) / len(recall_list) if recall_list else 0.0
    avg_p10 = sum(p10_list) / len(p10_list) if p10_list else 0.0
    avg_ndcg = sum(ndcg_list) / len(ndcg_list) if ndcg_list else 0.0
    
    print(f"[VSM] Dataset={dataset_name} | MAP: {avg_map*100:.2f}%, Recall: {avg_recall*100:.2f}%, P@10: {avg_p10*100:.2f}%, NDCG: {avg_ndcg*100:.2f}%")
    return {
        "precision_at_k": round(avg_p10, 4),
        "recall_score": round(avg_recall, 4),
        "map_score": round(avg_map, 4),
        "ndcg_score": round(avg_ndcg, 4)
    }

def evaluate_bm25(dataset_name, queries, qrels):
    bm25_path = os.path.join(MODELS_DIR, f"{dataset_name}_bm25.joblib")
    if not os.path.exists(bm25_path):
        print(f"[BM25] Warning: BM25 model not found at {bm25_path}")
        return None
        
    classical_queries = run_classical_preprocessing(queries)
    
    print("\n[BM25] Loading BM25 model (no memory mapping to prevent MemoryError)...")
    bm25_data = joblib.load(bm25_path)
    bm25 = bm25_data["bm25_vectorizer"]
    doc_ids = bm25_data["doc_ids"]
    
    print("[BM25] Setting up optimized BM25 matrix search...")
    import scipy.sparse as sp
    vocab = {token: i for i, token in enumerate(bm25.idf.keys())}
    indptr = [0]
    indices = []
    data_values = []
    for doc in bm25.doc_freqs:
        for token, freq in doc.items():
            col_idx = vocab.get(token)
            if col_idx is not None:
                indices.append(col_idx)
                data_values.append(freq)
        indptr.append(len(indices))
    csr = sp.csr_matrix((data_values, indices, indptr), dtype=np.float32, shape=(len(doc_ids), len(vocab)))
    tf_matrix = csr.tocsc()
    
    k1 = bm25.k1
    b = bm25.b
    avgdl = bm25.avgdl
    doc_len = np.array(bm25.doc_len)
    corpus_size = len(doc_ids)
    
    print("[BM25] Scoring all queries...")
    qids = list(classical_queries.keys())
    bm25_runs = {}
    bm25_top100 = {}
    
    for qid in qids:
        q_tokens = classical_queries[qid].split()
        scores = np.zeros(corpus_size)
        for token in q_tokens:
            idf = bm25.idf.get(token)
            if not idf:
                continue
            col_idx = vocab.get(token)
            if col_idx is None:
                continue
            
            col = tf_matrix[:, col_idx]
            matching_indices = col.indices
            q_freq = col.data
            if len(matching_indices) == 0:
                continue
            
            lengths = doc_len[matching_indices]
            denom = q_freq + k1 * (1.0 - b + b * lengths / avgdl)
            contrib = idf * (q_freq * (k1 + 1.0) / denom)
            scores[matching_indices] += contrib
            
        top_idx = np.argpartition(scores, -100)[-100:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        
        bm25_top100[qid] = [{"id": str(doc_ids[idx_i]), "score": float(scores[idx_i])} for idx_i in top_idx]
        bm25_runs[qid] = [str(doc_ids[idx_i]) for idx_i in top_idx[:10]]
        
    # Save the runs for hybrid merging
    run_save_path = os.path.join(os.path.dirname(CACHE_PATH), f"temp_runs_bm25_{dataset_name}.joblib")
    joblib.dump(bm25_top100, run_save_path)
    print(f"[BM25] Saved top 100 BM25 runs to {run_save_path}")
    
    del bm25_data, tf_matrix
    gc.collect()
    
    print("[BM25] Calculating metrics...")
    evaluator = IREvaluator(qrels)
    ap_list, recall_list, p10_list, ndcg_list = [], [], [], []
    for qid in bm25_runs:
        p10_list.append(evaluator.calculate_precision_at_k(bm25_runs[qid], qid, k=10))
        recall_list.append(evaluator.calculate_recall(bm25_runs[qid], qid))
        ap_list.append(evaluator.calculate_ap(bm25_runs[qid], qid, k=10))
        ndcg_list.append(evaluator.calculate_ndcg(bm25_runs[qid], qid, k=10))
        
    avg_map = sum(ap_list) / len(ap_list) if ap_list else 0.0
    avg_recall = sum(recall_list) / len(recall_list) if recall_list else 0.0
    avg_p10 = sum(p10_list) / len(p10_list) if p10_list else 0.0
    avg_ndcg = sum(ndcg_list) / len(ndcg_list) if ndcg_list else 0.0
    
    print(f"[BM25] Dataset={dataset_name} | MAP: {avg_map*100:.2f}%, Recall: {avg_recall*100:.2f}%, P@10: {avg_p10*100:.2f}%, NDCG: {avg_ndcg*100:.2f}%")
    return {
        "precision_at_k": round(avg_p10, 4),
        "recall_score": round(avg_recall, 4),
        "map_score": round(avg_map, 4),
        "ndcg_score": round(avg_ndcg, 4)
    }

def evaluate_embedding(dataset_name, queries, qrels):
    embedding_path = os.path.join(MODELS_DIR, f"{dataset_name}_embedding.joblib")
    if not os.path.exists(embedding_path):
        print(f"[Embedding] Warning: Embedding model not found at {embedding_path}")
        return None
        
    neural_queries = run_neural_preprocessing(queries)
    
    print("\n[Embedding] Loading dense embeddings with memory mapping...")
    emb_data = joblib.load(embedding_path, mmap_mode='r')
    doc_embeddings = emb_data["embeddings"]
    doc_ids = emb_data["doc_ids"]
    
    print("[Embedding] Initializing SentenceTransformer on CPU...")
    from sentence_transformers import SentenceTransformer
    bert = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("[Embedding] Encoding all queries on CPU...")
    qids = list(neural_queries.keys())
    q_texts = [neural_queries[qid] for qid in qids]
    q_embs = bert.encode(q_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    q_embs = q_embs.astype(np.float32) # Shape: (queries, 384)
    
    print("[Embedding] Calculating cosine similarities in batches (RAM-optimized batch size 20)...")
    batch_size = 20
    emb_top100 = {}
    emb_runs = {}
    
    for b_start in range(0, len(qids), batch_size):
        b_qids = qids[b_start : b_start + batch_size]
        b_q_embs = q_embs[b_start : b_start + batch_size]
        
        scores = np.dot(doc_embeddings, b_q_embs.T) # Shape: (docs, batch_size)
        
        for idx_in_batch, qid in enumerate(b_qids):
            q_scores = scores[:, idx_in_batch]
            top_idx = np.argpartition(q_scores, -100)[-100:]
            top_idx = top_idx[np.argsort(q_scores[top_idx])[::-1]]
            
            emb_top100[qid] = [{"id": str(doc_ids[idx_i]), "score": float(q_scores[idx_i])} for idx_i in top_idx]
            emb_runs[qid] = [str(doc_ids[idx_i]) for idx_i in top_idx[:10]]
            
    # Save the runs for hybrid merging
    run_save_path = os.path.join(os.path.dirname(CACHE_PATH), f"temp_runs_embedding_{dataset_name}.joblib")
    joblib.dump(emb_top100, run_save_path)
    print(f"[Embedding] Saved top 100 Embedding runs to {run_save_path}")
    
    del emb_data, doc_embeddings
    gc.collect()
    
    print("[Embedding] Calculating metrics...")
    evaluator = IREvaluator(qrels)
    ap_list, recall_list, p10_list, ndcg_list = [], [], [], []
    for qid in emb_runs:
        p10_list.append(evaluator.calculate_precision_at_k(emb_runs[qid], qid, k=10))
        recall_list.append(evaluator.calculate_recall(emb_runs[qid], qid))
        ap_list.append(evaluator.calculate_ap(emb_runs[qid], qid, k=10))
        ndcg_list.append(evaluator.calculate_ndcg(emb_runs[qid], qid, k=10))
        
    avg_map = sum(ap_list) / len(ap_list) if ap_list else 0.0
    avg_recall = sum(recall_list) / len(recall_list) if recall_list else 0.0
    avg_p10 = sum(p10_list) / len(p10_list) if p10_list else 0.0
    avg_ndcg = sum(ndcg_list) / len(ndcg_list) if ndcg_list else 0.0
    
    print(f"[Embedding] Dataset={dataset_name} | MAP: {avg_map*100:.2f}%, Recall: {avg_recall*100:.2f}%, P@10: {avg_p10*100:.2f}%, NDCG: {avg_ndcg*100:.2f}%")
    return {
        "precision_at_k": round(avg_p10, 4),
        "recall_score": round(avg_recall, 4),
        "map_score": round(avg_map, 4),
        "ndcg_score": round(avg_ndcg, 4)
    }

def evaluate_hybrid(dataset_name, queries, qrels):
    from services.retrieval_service.app.fusion import reciprocal_rank_fusion
    
    bm25_run_path = os.path.join(os.path.dirname(CACHE_PATH), f"temp_runs_bm25_{dataset_name}.joblib")
    embedding_run_path = os.path.join(os.path.dirname(CACHE_PATH), f"temp_runs_embedding_{dataset_name}.joblib")
    
    if not os.path.exists(bm25_run_path) or not os.path.exists(embedding_run_path):
        print(f"[Hybrid] Warning: BM25 or Embedding runs not found at {bm25_run_path} / {embedding_run_path}")
        return None
        
    print(f"\n[Hybrid] Loading BM25 and Embedding runs for {dataset_name}...")
    bm25_runs = joblib.load(bm25_run_path)
    embedding_runs = joblib.load(embedding_run_path)
    
    qids = list(queries.keys())
    hybrid_serial_runs = {}
    hybrid_parallel_runs = {}
    
    print("[Hybrid] Fusing runs (RRF Parallel and Reranking Serial)...")
    for qid in qids:
        bm25_res = bm25_runs.get(qid, [])
        dense_res = embedding_runs.get(qid, [])
        
        # Parallel Hybrid: RRF
        parallel_fused = reciprocal_rank_fusion([bm25_res, dense_res], weights=[0.5, 0.5])[:10]
        hybrid_parallel_runs[qid] = [r["id"] for r in parallel_fused]
        
        # Serial Hybrid: Dense rerank of top 100 BM25
        dense_scores_dict = {r["id"]: r["score"] for r in dense_res}
        serial_candidates = bm25_res[:100]
        serial_reranked = sorted(
            serial_candidates,
            key=lambda x: dense_scores_dict.get(x["id"], -1.0),
            reverse=True
        )[:10]
        hybrid_serial_runs[qid] = [r["id"] for r in serial_reranked]
        
    evaluator = IREvaluator(qrels)
    
    # Calculate Serial Metrics
    print("[Hybrid] Calculating metrics for Hybrid Serial...")
    ap_list, recall_list, p10_list, ndcg_list = [], [], [], []
    for qid in hybrid_serial_runs:
        p10_list.append(evaluator.calculate_precision_at_k(hybrid_serial_runs[qid], qid, k=10))
        recall_list.append(evaluator.calculate_recall(hybrid_serial_runs[qid], qid))
        ap_list.append(evaluator.calculate_ap(hybrid_serial_runs[qid], qid, k=10))
        ndcg_list.append(evaluator.calculate_ndcg(hybrid_serial_runs[qid], qid, k=10))
    avg_map_s = sum(ap_list) / len(ap_list) if ap_list else 0.0
    avg_rec_s = sum(recall_list) / len(recall_list) if recall_list else 0.0
    avg_p10_s = sum(p10_list) / len(p10_list) if p10_list else 0.0
    avg_ndcg_s = sum(ndcg_list) / len(ndcg_list) if ndcg_list else 0.0
    
    # Calculate Parallel Metrics
    print("[Hybrid] Calculating metrics for Hybrid Parallel...")
    ap_list, recall_list, p10_list, ndcg_list = [], [], [], []
    for qid in hybrid_parallel_runs:
        p10_list.append(evaluator.calculate_precision_at_k(hybrid_parallel_runs[qid], qid, k=10))
        recall_list.append(evaluator.calculate_recall(hybrid_parallel_runs[qid], qid))
        ap_list.append(evaluator.calculate_ap(hybrid_parallel_runs[qid], qid, k=10))
        ndcg_list.append(evaluator.calculate_ndcg(hybrid_parallel_runs[qid], qid, k=10))
    avg_map_p = sum(ap_list) / len(ap_list) if ap_list else 0.0
    avg_rec_p = sum(recall_list) / len(recall_list) if recall_list else 0.0
    avg_p10_p = sum(p10_list) / len(p10_list) if p10_list else 0.0
    avg_ndcg_p = sum(ndcg_list) / len(ndcg_list) if ndcg_list else 0.0
    
    print(f"[Hybrid] Dataset={dataset_name} | SERIAL -> MAP: {avg_map_s*100:.2f}%, Recall: {avg_rec_s*100:.2f}%, P@10: {avg_p10_s*100:.2f}%, NDCG: {avg_ndcg_s*100:.2f}%")
    print(f"[Hybrid] Dataset={dataset_name} | PARALLEL -> MAP: {avg_map_p*100:.2f}%, Recall: {avg_rec_p*100:.2f}%, P@10: {avg_p10_p*100:.2f}%, NDCG: {avg_ndcg_p*100:.2f}%")
    return {
        "hybrid_serial": {
            "precision_at_k": round(avg_p10_s, 4),
            "recall_score": round(avg_rec_s, 4),
            "map_score": round(avg_map_s, 4),
            "ndcg_score": round(avg_ndcg_s, 4)
        },
        "hybrid_parallel": {
            "precision_at_k": round(avg_p10_p, 4),
            "recall_score": round(avg_rec_p, 4),
            "map_score": round(avg_map_p, 4),
            "ndcg_score": round(avg_ndcg_p, 4)
        }
    }

def run_method(method):
    datasets = ["quora_dev", "lotte_lifestyle_dev"]
    method_metrics = {}
    
    for ds in datasets:
        queries, qrels = load_queries_and_qrels(ds)
        if method == "vsm":
            metrics = evaluate_vsm(ds, queries, qrels)
            if metrics:
                method_metrics[ds] = {"vsm": metrics}
        elif method == "bm25":
            metrics = evaluate_bm25(ds, queries, qrels)
            if metrics:
                method_metrics[ds] = {"bm25": metrics}
        elif method == "embedding":
            metrics = evaluate_embedding(ds, queries, qrels)
            if metrics:
                method_metrics[ds] = {"embedding": metrics}
        elif method == "hybrid":
            metrics = evaluate_hybrid(ds, queries, qrels)
            if metrics:
                method_metrics[ds] = metrics
                
    # Save temp metrics to disk
    temp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", f"temp_metrics_{method}.json"))
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(method_metrics, f, indent=4)
    print(f"\n[Subprocess SUCCESS] Saved temp metrics for {method} to {temp_path}")

def run_coordinator():
    print("================================================================================")
    print("=== [COORDINATOR] STARTING SUBPROCESS-ISOLATED FULL EVALUATION RECOMPUTATION ===")
    print("================================================================================")
    
    methods = ["vsm", "bm25", "embedding", "hybrid"]
    
    for method in methods:
        print(f"\n[Coordinator] Launching isolated subprocess for method: {method.upper()}...")
        cmd = [sys.executable, __file__, "--method", method]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\n[Coordinator ERROR] Subprocess for {method} failed with exit code {result.returncode}")
            sys.exit(result.returncode)
        print(f"[Coordinator] Isolated subprocess for {method} completed successfully.")
        
    print("\n[Coordinator] All model runs finished. Merging results...")
    datasets = ["quora_dev", "lotte_lifestyle_dev"]
    results = {}
    
    # Pre-populate empty cache structure if none exists
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception:
            pass
            
    for ds in datasets:
        results.setdefault(ds, {})
        for method in methods:
            temp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", f"temp_metrics_{method}.json"))
            if os.path.exists(temp_path):
                try:
                    with open(temp_path, "r", encoding="utf-8") as f:
                        method_metrics = json.load(f)
                    if ds in method_metrics:
                        if method == "hybrid":
                            results[ds]["hybrid_serial"] = method_metrics[ds]["hybrid_serial"]
                            results[ds]["hybrid_parallel"] = method_metrics[ds]["hybrid_parallel"]
                        else:
                            results[ds][method] = method_metrics[ds][method]
                except Exception as e:
                    print(f"Error loading metrics from {temp_path} for {ds}: {e}")
                    
    # Write merged results
    print(f"\n[Coordinator] Overwriting CACHE with REAL evaluation results at: {CACHE_PATH}...")
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    # Clean up temp files
    print("[Coordinator] Cleaning up temporary run files...")
    for method in methods:
        temp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", f"temp_metrics_{method}.json"))
        if os.path.exists(temp_path):
            os.remove(temp_path)
        for ds in datasets:
            temp_run_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", f"temp_runs_{method}_{ds}.joblib"))
            if os.path.exists(temp_run_path):
                os.remove(temp_run_path)
                
    print("\n=== [Coordinator] Full real evaluation results recomputed and saved! ===")

def main():
    parser = argparse.ArgumentParser(description="Recompute real evaluation metrics.")
    parser.add_argument("--method", choices=["vsm", "bm25", "embedding", "hybrid", "coordinator"], default="coordinator",
                        help="Select the model to evaluate (or run coordinator to run all in separate subprocesses)")
    args = parser.parse_args()
    
    if args.method == "coordinator":
        run_coordinator()
    else:
        run_method(args.method)

if __name__ == "__main__":
    main()
