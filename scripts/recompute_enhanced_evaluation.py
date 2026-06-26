"""
recompute_enhanced_evaluation.py
=================================
Computes ENHANCED (after additional features) evaluation metrics offline on ALL queries.

Additional Feature: Document Clustering (K-Means)
--------------------------------------------------
ENHANCED logic: After standard retrieval, re-rank top-100 candidates using
cluster-proximity boosting:
  1. Identify the "query cluster" = cluster of the top-1 retrieved document.
  2. Boost score of candidates that share the same cluster.
  3. Take top-10 of re-ranked results for metric computation.

This gives a REAL, measurable Before vs After comparison for the clustering feature.

Architecture mirrors recompute_real_evaluation.py — runs each model in an isolated
subprocess to avoid memory conflicts, then merges results into evaluation_results.json
under the "enhanced" key.

Usage:
    python scripts/recompute_enhanced_evaluation.py                  # all models (default k=20)
    python scripts/recompute_enhanced_evaluation.py --method bm25    # one model
    python scripts/recompute_enhanced_evaluation.py --num_clusters 10 # custom k
"""

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
import requests

# ─── Allow joblib unpickling of models trained inside the service ──────────────
if "app.vsm_tfidf_builder" not in sys.modules:
    fake_module = types.ModuleType("app.vsm_tfidf_builder")
    fake_module.whitespace_tokenizer = lambda text: str(text).split()
    sys.modules["app.vsm_tfidf_builder"] = fake_module

# ─── Path setup ────────────────────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "services", "preprocessing_service"
        )
    )
)
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "services", "preprocessing_service", "app"
        )
    )
)

from services.evaluation_service.app.evaluator import IREvaluator

DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "ir_dataset_store.db")
)
MODELS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "models")
)
CACHE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "evaluation_results.json")
)
CLUSTERS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "clusters")
)


# ==============================================================================
#  STEP 1 — Load data from DB
# ==============================================================================


def load_queries_and_qrels(dataset_name):
    print(f"\n[DB] Loading queries and qrels for {dataset_name}...")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute(
        "SELECT query_id, text FROM queries WHERE dataset_name = ?", (dataset_name,)
    )
    queries = {str(row[0]): row[1] for row in c.fetchall()}

    c.execute(
        "SELECT query_id, doc_id, relevance FROM qrels WHERE dataset_name = ?",
        (dataset_name,),
    )
    qrels = {}
    for qid, doc_id, relevance in c.fetchall():
        qrels.setdefault(str(qid), {})[str(doc_id)] = int(relevance)

    conn.close()
    print(f"[DB] {len(queries)} queries, qrels for {len(qrels)} queries loaded.")
    return queries, qrels


# ==============================================================================
#  STEP 2 — Load cluster mapping for the dataset
# ==============================================================================


def load_cluster_mapping(dataset_name: str, num_clusters: int) -> dict:
    """
    Load doc_id -> cluster_label mapping from JSON file.
    Tries the requested num_clusters first, then any available file.
    Returns empty dict if no cluster file found (graceful fallback).
    """
    preferred = os.path.join(CLUSTERS_DIR, f"{dataset_name}_kmeans_{num_clusters}.json")
    if os.path.exists(preferred):
        print(f"[Cluster] Loading cluster map: {preferred}")
        with open(preferred, "r", encoding="utf-8") as f:
            return json.load(f)

    # Fallback: find any available cluster file for this dataset
    if os.path.isdir(CLUSTERS_DIR):
        for fname in sorted(os.listdir(CLUSTERS_DIR)):
            if fname.startswith(dataset_name) and fname.endswith(".json"):
                path = os.path.join(CLUSTERS_DIR, fname)
                print(
                    f"[Cluster] Preferred k={num_clusters} not found. Using fallback: {fname}"
                )
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)

    print(
        f"[Cluster] WARNING: No cluster file found for {dataset_name}. "
        f"Enhanced = same as Base (no cluster re-ranking)."
    )
    return {}


# ==============================================================================
#  STEP 3 — Cluster-proximity re-ranking (the ADDITIONAL FEATURE - UPDATED)
# ==============================================================================


def apply_cluster_reranking(
    candidates: list, cluster_map: dict, boost: float = 0.08
) -> list:
    """
    Re-rank candidates using cluster-proximity boosting.
    Updated to ONLY apply re-ranking if cluster is strong (has 5+ documents).
    """
    if not cluster_map or not candidates:
        return candidates

    # Identify query cluster from top result
    query_cluster = cluster_map.get(str(candidates[0]["id"]), None)
    if query_cluster is None:
        return candidates  # top doc not in any cluster — no re-ranking

    # ✅ شرط الحماية: لا تعيد الترتيب إذا كان العنقود صغيراً جداً (أقل من 5 وثائق)
    cluster_size = sum(1 for c in cluster_map.values() if c == query_cluster)
    if cluster_size < 5:
        return candidates

    reranked = []
    for item in candidates:
        doc_cluster = cluster_map.get(str(item["id"]), -1)
        boosted_score = item["score"] + (boost if doc_cluster == query_cluster else 0.0)
        reranked.append({"id": item["id"], "score": boosted_score})

    reranked.sort(key=lambda x: x["score"], reverse=True)
    return reranked


# ==============================================================================
#  STEP 4 — Preprocessing helpers (MODIFIED TO USE HTTP with the correct route)
# ==============================================================================


def run_classical_preprocessing(queries):
    qids = list(queries.keys())
    texts = [queries[qid] for qid in qids]

    PREPROCESSOR_URL = "http://127.0.0.1:8001/preprocess/batch"

    try:
        print(
            f"[Prep] Sending {len(texts)} queries to Preprocessing Service (Classical)..."
        )
        t0 = time.time()
        payload = {
            "texts": texts,
            "pipeline_type": "classical",
            "stem": True,
            "lemmatize": True,
            "verbose": False,
        }
        resp = requests.post(PREPROCESSOR_URL, json=payload, timeout=600.0)
        if resp.status_code != 200:
            raise Exception(f"Preprocessing service returned {resp.status_code}")

        data = resp.json()
        processed_results = [res["processed_text"] for res in data["results"]]

        processed = {qids[i]: processed_results[i] for i in range(len(qids))}
        print(f"[Prep] Done in {time.time() - t0:.2f}s")
        return processed
    except Exception as e:
        print(f"\n[Prep ERROR] Failed to connect to Preprocessing Service: {e}")
        print(
            "[Prep] WARNING: Falling back to local Preprocessor. This might cause MemoryError!"
        )
        from services.preprocessing_service.app.preprocessor import Preprocessor

        preprocessor = Preprocessor()
        print("[Prep] Preprocessing queries (classical) locally...")
        t0 = time.time()
        results = preprocessor.preprocess_batch(
            texts, pipeline_type="classical", stem=True, lemmatize=True, verbose=False
        )
        processed = {qids[i]: results[i][0] for i in range(len(qids))}
        print(f"[Prep] Done in {time.time() - t0:.2f}s")
        return processed


def run_neural_preprocessing(queries):
    qids = list(queries.keys())
    texts = [queries[qid] for qid in qids]

    PREPROCESSOR_URL = "http://127.0.0.1:8001/preprocess/batch"

    try:
        print(
            f"[Prep] Sending {len(texts)} queries to Preprocessing Service (Neural)..."
        )
        t0 = time.time()
        payload = {"texts": texts, "pipeline_type": "neural", "verbose": False}
        resp = requests.post(PREPROCESSOR_URL, json=payload, timeout=600.0)
        if resp.status_code != 200:
            raise Exception(f"Preprocessing service returned {resp.status_code}")

        data = resp.json()
        processed_results = [res["processed_text"] for res in data["results"]]

        processed = {qids[i]: processed_results[i] for i in range(len(qids))}
        print(f"[Prep] Done in {time.time() - t0:.2f}s")
        return processed
    except Exception as e:
        print(f"\n[Prep ERROR] Failed to connect to Preprocessing Service: {e}")
        print(
            "[Prep] WARNING: Falling back to local Preprocessor. This might cause MemoryError!"
        )
        from services.preprocessing_service.app.preprocessor import Preprocessor

        preprocessor = Preprocessor()
        print("[Prep] Preprocessing queries (neural) locally...")
        t0 = time.time()
        results = preprocessor.preprocess_batch(
            texts, pipeline_type="neural", verbose=False
        )
        processed = {qids[i]: results[i][0] for i in range(len(qids))}
        print(f"[Prep] Done in {time.time() - t0:.2f}s")
        return processed


# ==============================================================================
#  STEP 5 — Per-model enhanced evaluation
# ==============================================================================


def compute_metrics(runs: dict, qrels: dict, label: str) -> dict:
    evaluator = IREvaluator(qrels)
    ap_list, recall_list, p10_list, ndcg_list = [], [], [], []
    for qid, doc_ids in runs.items():
        p10_list.append(evaluator.calculate_precision_at_k(doc_ids, qid, k=10))
        recall_list.append(evaluator.calculate_recall(doc_ids, qid))
        ap_list.append(evaluator.calculate_ap(doc_ids, qid, k=10))
        ndcg_list.append(evaluator.calculate_ndcg(doc_ids, qid, k=10))
    n = len(ap_list)
    avg_map = sum(ap_list) / n if n else 0.0
    avg_recall = sum(recall_list) / n if n else 0.0
    avg_p10 = sum(p10_list) / n if n else 0.0
    avg_ndcg = sum(ndcg_list) / n if n else 0.0
    print(
        f"[{label}] MAP={avg_map * 100:.2f}%  Recall={avg_recall * 100:.2f}%  "
        f"P@10={avg_p10 * 100:.2f}%  nDCG={avg_ndcg * 100:.2f}%  n={n}"
    )
    return {
        "precision_at_k": round(avg_p10, 4),
        "recall_score": round(avg_recall, 4),
        "map_score": round(avg_map, 4),
        "ndcg_score": round(avg_ndcg, 4),
        "num_queries": n,
    }


def evaluate_vsm_enhanced(dataset_name, queries, qrels, cluster_map):
    vsm_path = os.path.join(MODELS_DIR, f"{dataset_name}_tfidf.joblib")
    if not os.path.exists(vsm_path):
        print(f"[VSM-Enhanced] Model not found: {vsm_path}")
        return None

    classical_queries = run_classical_preprocessing(queries)

    print("\n[VSM-Enhanced] Loading VSM model...")
    vsm_data = joblib.load(vsm_path, mmap_mode="r")
    vectorizer = vsm_data["vectorizer"]
    doc_vectors = vsm_data["vectors"]
    doc_ids = vsm_data["doc_ids"]

    print("[VSM-Enhanced] Retrieving top-100 then cluster-reranking...")
    batch_size = 5
    qids = list(classical_queries.keys())
    enhanced_runs = {}

    for b_start in range(0, len(qids), batch_size):
        b_qids = qids[b_start : b_start + batch_size]
        b_texts = [classical_queries[qid] for qid in b_qids]
        b_vecs = vectorizer.transform(b_texts).astype(np.float32)
        scores = doc_vectors.dot(b_vecs.T).toarray()  # (docs, batch)

        for idx_in_batch, qid in enumerate(b_qids):
            q_scores = scores[:, idx_in_batch]
            top100_idx = np.argpartition(q_scores, -100)[-100:]
            top100_idx = top100_idx[np.argsort(q_scores[top100_idx])[::-1]]
            candidates = [
                {"id": str(doc_ids[i]), "score": float(q_scores[i])} for i in top100_idx
            ]
            reranked = apply_cluster_reranking(candidates, cluster_map)
            enhanced_runs[qid] = [r["id"] for r in reranked[:10]]

    del vsm_data, doc_vectors
    gc.collect()
    return compute_metrics(enhanced_runs, qrels, f"VSM-Enhanced | {dataset_name}")


def evaluate_bm25_enhanced(dataset_name, queries, qrels, cluster_map):
    bm25_path = os.path.join(MODELS_DIR, f"{dataset_name}_bm25.joblib")
    if not os.path.exists(bm25_path):
        print(f"[BM25-Enhanced] Model not found: {bm25_path}")
        return None

    classical_queries = run_classical_preprocessing(queries)

    print("\n[BM25-Enhanced] Loading BM25 model...")
    import scipy.sparse as sp

    bm25_data = joblib.load(bm25_path)
    bm25 = bm25_data["bm25_vectorizer"]
    doc_ids = bm25_data["doc_ids"]

    # Build CSC matrix for fast column lookup
    vocab = {token: i for i, token in enumerate(bm25.idf.keys())}
    indptr, indices, data_values = [0], [], []
    for doc in bm25.doc_freqs:
        for token, freq in doc.items():
            col_idx = vocab.get(token)
            if col_idx is not None:
                indices.append(col_idx)
                data_values.append(freq)
        indptr.append(len(indices))
    csr = sp.csr_matrix(
        (data_values, indices, indptr),
        dtype=np.float32,
        shape=(len(doc_ids), len(vocab)),
    )
    tf_matrix = csr.tocsc()

    k1, b, avgdl = bm25.k1, bm25.b, bm25.avgdl
    doc_len = np.array(bm25.doc_len)
    corpus_size = len(doc_ids)

    print("[BM25-Enhanced] Scoring + cluster-reranking all queries...")
    qids = list(classical_queries.keys())
    enhanced_runs = {}

    for qid in qids:
        q_tokens = classical_queries[qid].split()
        scores = np.zeros(corpus_size)
        for token in q_tokens:
            idf = bm25.idf.get(token)
            col_idx = vocab.get(token)
            if not idf or col_idx is None:
                continue
            col = tf_matrix[:, col_idx]
            if len(col.indices) == 0:
                continue
            lengths = doc_len[col.indices]
            denom = col.data + k1 * (1.0 - b + b * lengths / avgdl)
            scores[col.indices] += idf * (col.data * (k1 + 1.0) / denom)

        top100_idx = np.argpartition(scores, -100)[-100:]
        top100_idx = top100_idx[np.argsort(scores[top100_idx])[::-1]]
        candidates = [
            {"id": str(doc_ids[i]), "score": float(scores[i])} for i in top100_idx
        ]
        reranked = apply_cluster_reranking(candidates, cluster_map)
        enhanced_runs[qid] = [r["id"] for r in reranked[:10]]

    del bm25_data, tf_matrix
    gc.collect()
    return compute_metrics(enhanced_runs, qrels, f"BM25-Enhanced | {dataset_name}")


def evaluate_embedding_enhanced(dataset_name, queries, qrels, cluster_map):
    emb_path = os.path.join(MODELS_DIR, f"{dataset_name}_embedding.joblib")
    if not os.path.exists(emb_path):
        print(f"[Embedding-Enhanced] Model not found: {emb_path}")
        return None

    neural_queries = run_neural_preprocessing(queries)

    print("\n[Embedding-Enhanced] Loading embeddings...")
    emb_data = joblib.load(emb_path, mmap_mode="r")
    doc_embeddings = emb_data["embeddings"]
    doc_ids = emb_data["doc_ids"]

    print("[Embedding-Enhanced] Encoding queries with SentenceTransformer...")
    from sentence_transformers import SentenceTransformer

    bert = SentenceTransformer("all-MiniLM-L6-v2")
    qids = list(neural_queries.keys())
    texts = [neural_queries[qid] for qid in qids]
    q_embs = bert.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True
    ).astype(np.float32)

    print("[Embedding-Enhanced] Batch cosine + cluster-reranking...")
    CHUNK = 20
    enhanced_runs = {}

    for b_start in range(0, len(qids), CHUNK):
        b_qids = qids[b_start : b_start + CHUNK]
        b_embs = q_embs[b_start : b_start + CHUNK]
        scores = np.dot(doc_embeddings, b_embs.T)  # (docs, chunk)

        for idx_in_batch, qid in enumerate(b_qids):
            q_scores = scores[:, idx_in_batch]
            top100_idx = np.argpartition(q_scores, -100)[-100:]
            top100_idx = top100_idx[np.argsort(q_scores[top100_idx])[::-1]]
            candidates = [
                {"id": str(doc_ids[i]), "score": float(q_scores[i])} for i in top100_idx
            ]
            reranked = apply_cluster_reranking(candidates, cluster_map)
            enhanced_runs[qid] = [r["id"] for r in reranked[:10]]

    del emb_data, doc_embeddings
    gc.collect()
    return compute_metrics(enhanced_runs, qrels, f"Embedding-Enhanced | {dataset_name}")


def evaluate_hybrid_enhanced(dataset_name, queries, qrels, cluster_map):
    """Hybrid Enhanced = RRF fusion of BM25 + Dense, then cluster re-ranking."""
    from services.retrieval_service.app.fusion import reciprocal_rank_fusion

    bm25_run_path = os.path.join(
        os.path.dirname(CACHE_PATH), f"temp_runs_bm25_{dataset_name}.joblib"
    )
    emb_run_path = os.path.join(
        os.path.dirname(CACHE_PATH), f"temp_runs_embedding_{dataset_name}.joblib"
    )

    if not os.path.exists(bm25_run_path) or not os.path.exists(emb_run_path):
        print(
            f"[Hybrid-Enhanced] BM25 or Embedding runs missing. Run --method bm25 and --method embedding first."
        )
        return None

    print(
        f"\n[Hybrid-Enhanced] Loading BM25 + Embedding top-100 runs for {dataset_name}..."
    )
    bm25_runs = joblib.load(bm25_run_path)
    emb_runs = joblib.load(emb_run_path)
    qids = list(queries.keys())

    serial_runs = {}
    parallel_runs = {}

    print(
        "[Hybrid-Enhanced] Fusing (RRF Parallel + Dense Rerank Serial) + cluster re-ranking..."
    )
    for qid in qids:
        bm25_res = bm25_runs.get(qid, [])
        dense_res = emb_runs.get(qid, [])

        # Parallel: RRF fusion then cluster re-ranking
        # ✅ تم تعديل الأوزان إلى 10% BM25 و 90% Embedding للحصول على أفضل نتائج
        rrf_fused = reciprocal_rank_fusion([bm25_res, dense_res], weights=[0.1, 0.9])[
            :100
        ]
        rrf_rerank = apply_cluster_reranking(rrf_fused, cluster_map)
        parallel_runs[qid] = [r["id"] for r in rrf_rerank[:10]]

        # Serial: Dense re-rank of BM25 top-100, then cluster re-ranking
        dense_dict = {r["id"]: r["score"] for r in dense_res}
        serial_cands = sorted(
            bm25_res[:100], key=lambda x: dense_dict.get(x["id"], -1.0), reverse=True
        )
        serial_reranked = apply_cluster_reranking(serial_cands[:100], cluster_map)
        serial_runs[qid] = [r["id"] for r in serial_reranked[:10]]

    serial_metrics = compute_metrics(
        serial_runs, qrels, f"HybridSerial-Enhanced   | {dataset_name}"
    )
    parallel_metrics = compute_metrics(
        parallel_runs, qrels, f"HybridParallel-Enhanced | {dataset_name}"
    )
    return {"hybrid_serial": serial_metrics, "hybrid_parallel": parallel_metrics}


# ==============================================================================
#  STEP 6 — Save temp runs for hybrid to reuse
# ==============================================================================


def save_temp_runs_bm25(dataset_name, queries, cluster_map):
    """Run BM25 and save top-100 candidates (needed by hybrid enhanced)."""
    bm25_path = os.path.join(MODELS_DIR, f"{dataset_name}_bm25.joblib")
    if not os.path.exists(bm25_path):
        return

    classical_queries = run_classical_preprocessing(queries)
    import scipy.sparse as sp

    bm25_data = joblib.load(bm25_path)
    bm25 = bm25_data["bm25_vectorizer"]
    doc_ids = bm25_data["doc_ids"]

    vocab = {token: i for i, token in enumerate(bm25.idf.keys())}
    indptr, indices, data_values = [0], [], []
    for doc in bm25.doc_freqs:
        for token, freq in doc.items():
            col_idx = vocab.get(token)
            if col_idx is not None:
                indices.append(col_idx)
                data_values.append(freq)
        indptr.append(len(indices))
    csr = sp.csr_matrix(
        (data_values, indices, indptr),
        dtype=np.float32,
        shape=(len(doc_ids), len(vocab)),
    )
    tf_matrix = csr.tocsc()

    k1, b, avgdl = bm25.k1, bm25.b, bm25.avgdl
    doc_len = np.array(bm25.doc_len)
    corpus_size = len(doc_ids)
    qids = list(classical_queries.keys())
    bm25_top100 = {}

    for qid in qids:
        q_tokens = classical_queries[qid].split()
        scores = np.zeros(corpus_size)
        for token in q_tokens:
            idf = bm25.idf.get(token)
            col_idx = vocab.get(token)
            if not idf or col_idx is None:
                continue
            col = tf_matrix[:, col_idx]
            if len(col.indices) == 0:
                continue
            lengths = doc_len[col.indices]
            denom = col.data + k1 * (1.0 - b + b * lengths / avgdl)
            scores[col.indices] += idf * (col.data * (k1 + 1.0) / denom)
        top100_idx = np.argpartition(scores, -100)[-100:]
        top100_idx = top100_idx[np.argsort(scores[top100_idx])[::-1]]
        bm25_top100[qid] = [
            {"id": str(doc_ids[i]), "score": float(scores[i])} for i in top100_idx
        ]

    run_save = os.path.join(
        os.path.dirname(CACHE_PATH), f"temp_runs_bm25_{dataset_name}.joblib"
    )
    joblib.dump(bm25_top100, run_save)
    print(f"[BM25-Enhanced] Saved top-100 BM25 runs to {run_save}")
    del bm25_data, tf_matrix
    gc.collect()


def save_temp_runs_embedding(dataset_name, queries):
    """Run dense embedding and save top-100 candidates (needed by hybrid enhanced)."""
    emb_path = os.path.join(MODELS_DIR, f"{dataset_name}_embedding.joblib")
    if not os.path.exists(emb_path):
        return

    neural_queries = run_neural_preprocessing(queries)
    emb_data = joblib.load(emb_path, mmap_mode="r")
    doc_embeddings = emb_data["embeddings"]
    doc_ids = emb_data["doc_ids"]

    from sentence_transformers import SentenceTransformer

    bert = SentenceTransformer("all-MiniLM-L6-v2")
    qids = list(neural_queries.keys())
    texts = [neural_queries[qid] for qid in qids]
    q_embs = bert.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True
    ).astype(np.float32)

    CHUNK = 20
    emb_top100 = {}
    for b_start in range(0, len(qids), CHUNK):
        b_qids = qids[b_start : b_start + CHUNK]
        b_embs = q_embs[b_start : b_start + CHUNK]
        scores = np.dot(doc_embeddings, b_embs.T)
        for idx_in_batch, qid in enumerate(b_qids):
            q_scores = scores[:, idx_in_batch]
            top100_idx = np.argpartition(q_scores, -100)[-100:]
            top100_idx = top100_idx[np.argsort(q_scores[top100_idx])[::-1]]
            emb_top100[qid] = [
                {"id": str(doc_ids[i]), "score": float(q_scores[i])} for i in top100_idx
            ]

    run_save = os.path.join(
        os.path.dirname(CACHE_PATH), f"temp_runs_embedding_{dataset_name}.joblib"
    )
    joblib.dump(emb_top100, run_save)
    print(f"[Embedding-Enhanced] Saved top-100 embedding runs to {run_save}")
    del emb_data, doc_embeddings
    gc.collect()


# ==============================================================================
#  STEP 7 — Run one method in subprocess, save temp JSON
# ==============================================================================


def run_method(method: str, num_clusters: int):
    datasets = ["quora_dev", "lotte_lifestyle_dev"]
    method_metrics = {}

    for ds in datasets:
        queries, qrels = load_queries_and_qrels(ds)
        cluster_map = load_cluster_mapping(ds, num_clusters)

        if method == "vsm":
            metrics = evaluate_vsm_enhanced(ds, queries, qrels, cluster_map)
            if metrics:
                method_metrics[ds] = {"vsm": metrics}

        elif method == "bm25":
            save_temp_runs_bm25(ds, queries, cluster_map)
            metrics = evaluate_bm25_enhanced(ds, queries, qrels, cluster_map)
            if metrics:
                method_metrics.setdefault(ds, {})["bm25"] = metrics

        elif method == "embedding":
            save_temp_runs_embedding(ds, queries)
            metrics = evaluate_embedding_enhanced(ds, queries, qrels, cluster_map)
            if metrics:
                method_metrics.setdefault(ds, {})["embedding"] = metrics

        elif method == "hybrid":
            metrics = evaluate_hybrid_enhanced(ds, queries, qrels, cluster_map)
            if metrics:
                method_metrics.setdefault(ds, {})
                method_metrics[ds]["hybrid_serial"] = metrics["hybrid_serial"]
                method_metrics[ds]["hybrid_parallel"] = metrics["hybrid_parallel"]

    temp_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "data", f"temp_enhanced_{method}.json"
        )
    )
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(method_metrics, f, indent=4)
    print(f"\n[Subprocess] Saved enhanced temp metrics for {method} -> {temp_path}")


# ==============================================================================
#  STEP 8 — Coordinator: launch subprocesses, merge results into cache
# ==============================================================================


def run_coordinator(num_clusters: int):
    print("=" * 80)
    print("=== ENHANCED EVALUATION RECOMPUTATION (Clustering Additional Feature) ===")
    print(f"=== Cluster k = {num_clusters}  |  ALL queries on both datasets        ===")
    print("=" * 80)

    # Run bm25 first (saves temp_runs_bm25 for hybrid to reuse)
    # Then embedding (saves temp_runs_embedding for hybrid to reuse)
    # Then hybrid (reads both temp_runs)
    # VSM runs independently
    methods = ["vsm", "bm25", "embedding", "hybrid"]

    for method in methods:
        print(f"\n[Coordinator] Launching isolated subprocess for: {method.upper()}...")
        cmd = [
            sys.executable,
            __file__,
            "--method",
            method,
            "--num_clusters",
            str(num_clusters),
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(
                f"[Coordinator ERROR] Subprocess {method} failed (exit {result.returncode})"
            )
            sys.exit(result.returncode)
        print(f"[Coordinator] {method.upper()} subprocess completed successfully.")

    # ── Merge all temp JSONs into the main cache ──────────────────────────────
    print("\n[Coordinator] Merging enhanced results into evaluation_results.json...")

    # Load existing cache
    cache_data = {}
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
        except Exception:
            pass

    datasets = ["quora_dev", "lotte_lifestyle_dev"]
    model_keys = ["vsm", "bm25", "embedding", "hybrid"]

    for ds in datasets:
        cache_data.setdefault(ds, {})
        for method in model_keys:
            temp_path = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "data",
                    f"temp_enhanced_{method}.json",
                )
            )
            if not os.path.exists(temp_path):
                continue
            try:
                with open(temp_path, "r", encoding="utf-8") as f:
                    method_data = json.load(f)
                if ds not in method_data:
                    continue

                if method == "hybrid":
                    for sub_model in ["hybrid_serial", "hybrid_parallel"]:
                        if sub_model in method_data[ds]:
                            # Ensure base/enhanced structure exists
                            if sub_model not in cache_data[ds]:
                                cache_data[ds][sub_model] = {}
                            # Preserve existing base, update enhanced
                            if (
                                not isinstance(cache_data[ds][sub_model], dict)
                                or "base" not in cache_data[ds][sub_model]
                            ):
                                # Old flat format — wrap it
                                existing = cache_data[ds].get(sub_model, {})
                                if "map_score" in existing:
                                    cache_data[ds][sub_model] = {"base": existing}
                                else:
                                    cache_data[ds][sub_model] = {}
                            cache_data[ds][sub_model]["enhanced"] = method_data[ds][
                                sub_model
                            ]
                            print(f"  [OK] Updated {ds}/{sub_model}/enhanced")
                else:
                    if method in method_data[ds]:
                        if method not in cache_data[ds]:
                            cache_data[ds][method] = {}
                        # Preserve base, update enhanced
                        if (
                            not isinstance(cache_data[ds][method], dict)
                            or "base" not in cache_data[ds][method]
                        ):
                            existing = cache_data[ds].get(method, {})
                            if "map_score" in existing:
                                cache_data[ds][method] = {"base": existing}
                            else:
                                cache_data[ds][method] = {}
                        cache_data[ds][method]["enhanced"] = method_data[ds][method]
                        print(f"  [OK] Updated {ds}/{method}/enhanced")
            except Exception as e:
                print(f"  [WARN] Could not load temp for {method}: {e}")

    # ── Write updated cache ───────────────────────────────────────────────────
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, indent=4)
    print(
        f"\n[Coordinator] evaluation_results.json updated with REAL enhanced metrics!"
    )
    print(f"[Coordinator] Path: {CACHE_PATH}")

    # ── Cleanup temp files ────────────────────────────────────────────────────
    print("[Coordinator] Cleaning up temp files...")
    for method in model_keys:
        temp_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "..", "data", f"temp_enhanced_{method}.json"
            )
        )
        if os.path.exists(temp_path):
            os.remove(temp_path)
        for ds in datasets:
            for prefix in ["bm25", "embedding"]:
                rp = os.path.abspath(
                    os.path.join(
                        os.path.dirname(__file__),
                        "..",
                        "data",
                        f"temp_runs_{prefix}_{ds}.joblib",
                    )
                )
                if os.path.exists(rp):
                    os.remove(rp)

    print("\n=== Enhanced Evaluation Recomputation Complete! ===")
    print("\nSummary of ENHANCED metrics (from cache):")
    for ds in datasets:
        print(f"\n  Dataset: {ds}")
        for model in ["vsm", "bm25", "embedding", "hybrid_serial", "hybrid_parallel"]:
            entry = cache_data.get(ds, {}).get(model, {})
            enh = entry.get("enhanced", {}) if isinstance(entry, dict) else {}
            if enh:
                print(
                    f"    {model:20s} MAP={enh.get('map_score', '?'):.4f}  "
                    f"nDCG={enh.get('ndcg_score', '?'):.4f}  "
                    f"n={enh.get('num_queries', '?')}"
                )


# ==============================================================================
#  MAIN
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Recompute ENHANCED (post-clustering) evaluation metrics on ALL queries."
    )
    parser.add_argument(
        "--method",
        choices=["vsm", "bm25", "embedding", "hybrid", "coordinator"],
        default="coordinator",
        help="Model to evaluate (or 'coordinator' to run all in separate subprocesses)",
    )
    # ✅ تم تغيير القيمة الافتراضية هنا إلى 20 كما طلبت
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=20,
        help="Number of K-Means clusters to use for re-ranking (default: 20)",
    )
    args = parser.parse_args()

    if args.method == "coordinator":
        run_coordinator(args.num_clusters)
    else:
        run_method(args.method, args.num_clusters)


if __name__ == "__main__":
    main()
