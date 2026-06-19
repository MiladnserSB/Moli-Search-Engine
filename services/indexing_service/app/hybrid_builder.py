# Parallel Hybrid Representation Builder
import os
import gc
import time
import joblib
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from .config import settings

def build_hybrid_representation(dataset_name: str) -> dict:
    start_time = time.time()
    print("\n" + "="*60)
    print(f"[PARALLEL HYBRID BUILDER] Starting build for dataset: {dataset_name.upper()}")
    print("="*60)
    
    tfidf_path = os.path.join(settings.MODELS_DIR, f"{dataset_name}_tfidf.joblib")
    embedding_path = os.path.join(settings.MODELS_DIR, f"{dataset_name}_embedding.joblib")
    
    print("[CHECK] Checking for TF-IDF and BERT embedding models on disk...")
    if not os.path.exists(tfidf_path) or not os.path.exists(embedding_path):
        raise FileNotFoundError(
            f"Required models are missing. Ensure TF-IDF and BERT are built first.\n"
            f"Missing TF-IDF: {not os.path.exists(tfidf_path)}\n"
            f"Missing BERT: {not os.path.exists(embedding_path)}"
        )
        
    # 1. Load both representations
    print("[LOAD] Loading TF-IDF model...")
    load_start = time.time()
    tfidf_data = joblib.load(tfidf_path)
    print(f"[OK] Loaded TF-IDF in {time.time() - load_start:.2f} seconds.")
    
    print("[LOAD] Loading BERT embedding model...")
    load_start = time.time()
    bert_data = joblib.load(embedding_path)
    print(f"[OK] Loaded BERT in {time.time() - load_start:.2f} seconds.")
    
    # Check alignment
    print("[ALIGNMENT] Checking document alignment between models...")
    if tfidf_data["doc_ids"] != bert_data["doc_ids"]:
        raise ValueError("Critical Mismatch: Document IDs in TF-IDF and BERT do not align.")
    print("[OK] Document alignment verified successfully.")
    
    tfidf_matrix = tfidf_data["vectors"]
    bert_vectors = bert_data["embeddings"]
    
    # 2. L2 Normalization of both vectors
    print("\n[NORMALIZATION] Normalizing TF-IDF sparse matrix (L2 normalization)...")
    tfidf_norm = normalize(tfidf_matrix, norm='l2', axis=1)
    
    print("[NORMALIZATION] Normalizing BERT dense matrix (L2 normalization)...")
    bert_norm = normalize(bert_vectors, norm='l2', axis=1)
    
    # 3. TruncatedSVD Dimension Reduction
    print("\n[DIMENSION_REDUCTION] Reducing TF-IDF dimensions to 300 using TruncatedSVD...")
    svd_start = time.time()
    svd = TruncatedSVD(n_components=300, random_state=42)
    tfidf_reduced = svd.fit_transform(tfidf_norm)
    tfidf_reduced = normalize(tfidf_reduced, norm='l2', axis=1)
    svd_duration = time.time() - svd_start
    print(f"[OK] TruncatedSVD completed in {svd_duration:.2f} seconds.")
    
    # Calculate explained variance
    explained_var = svd.explained_variance_ratio_.sum() * 100
    print(f"[STATS] Total explained variance retained by 300 SVD dimensions: {explained_var:.2f}%")
    
    # 4. Concatenation (Parallel Hybrid)
    print(f"\n[CONCATENATION] Concatenating TF-IDF (SVD 300) + BERT (384) -> {tfidf_reduced.shape[1] + bert_norm.shape[1]} dimensions...")
    hybrid_vectors = np.concatenate([tfidf_reduced, bert_norm], axis=1).astype(np.float32)
    print(f"[OK] Concatenation completed. Hybrid Matrix Shape: {hybrid_vectors.shape}")
    
    # 5. Save to Disk
    os.makedirs(settings.MODELS_DIR, exist_ok=True)
    out_path = os.path.join(settings.MODELS_DIR, f"{dataset_name}_hybrid.joblib")
    
    print(f"\n[SAVE] Dumping hybrid vectors and SVD reducer model to disk at: {out_path}...")
    dimensions = hybrid_vectors.shape[1] #اني ضفتها مشان الايرور تبع المتغير غير موجود في النطاق الحالي.
    save_start = time.time()
    joblib.dump({
        "hybrid_vectors": hybrid_vectors,
        "doc_ids": tfidf_data["doc_ids"],
        "svd_model": svd
    }, out_path)
    print(f"[SAVE] Model dumped successfully in {time.time() - save_start:.2f} seconds.")
    
    # Clean memory
    del tfidf_data, bert_data, tfidf_norm, tfidf_reduced, bert_norm, hybrid_vectors
    gc.collect()
    
    elapsed_time = time.time() - start_time
    print(f"\n[SUCCESS] Hybrid Build completed in {elapsed_time:.2f} seconds.")
    print("="*60 + "\n")
    
    # return {"status": "success", "file": out_path, "dimensions": hybrid_vectors.shape[1]}

    return {"status": "success", "file": out_path, "dimensions": dimensions} # حطينا dimensions مشان مشكلة المتغير غير موجود في النطاق الحالي.
