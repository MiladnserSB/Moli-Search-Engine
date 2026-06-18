# TF-IDF Representation Builder
import os
import gc
import time
import sqlite3
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from .preprocessing import processing, tokenize
from .config import settings

def whitespace_tokenizer(text):
    return str(text).split()

def build_tfidf_representation(dataset_name: str) -> dict:
    start_time = time.time()
    print("\n" + "="*60)
    print(f"[TF-IDF BUILDER] Starting build for dataset: {dataset_name.upper()}")
    print("="*60)
    
    # 1. Database connection and fetching preprocessed text
    print("[DB_CONNECT] Connecting to SQLite database...")
    conn = sqlite3.connect(settings.DB_PATH)
    print("[DB_LOAD] Loading preprocessed text documents from 'processed_documents' table (populated by preprocessing_service)...")
    df = pd.read_sql(
        "SELECT doc_id, processed_text FROM processed_documents WHERE dataset_name IN (?, ?)", 
        conn, 
        params=(dataset_name, f"{dataset_name}_classical")
    )
    conn.close()
    
    total_docs = len(df)
    print(f"[OK] Loaded {total_docs:,} preprocessed documents.")
    
    if df.empty:
        raise ValueError(f"No preprocessed documents found for dataset {dataset_name}. Has the preprocessing service run?")
        
    # 2. Vectorizer Configuration & Fitting
    print("\n[CONFIG] Configuring TfidfVectorizer with pre-tokenized inputs:")
    print("   - Tokenizer: Whitespace split (already preprocessed by preprocessing_service)")
    print("   - Preprocessor: None")
    print("   - Max Features: 50,000 top words")
    print("   - Norm: L2 normalization")
    print("   - Use IDF: True")
    
    vectorizer = TfidfVectorizer(
        preprocessor=None,
        tokenizer=whitespace_tokenizer,
        lowercase=False,
        token_pattern=None,
        max_features=50000,
        norm="l2",
        use_idf=True
    )
    
    print("\n[PROCESSING] Fitting and transforming preprocessed corpus...")
    fit_start = time.time()
    X = vectorizer.fit_transform(df['processed_text'])
    fit_duration = time.time() - fit_start
    print(f"[OK] Fit completed in {fit_duration:.2f} seconds.")
    
    # 3. Compression and Memory Optimizations
    vocab_size = len(vectorizer.vocabulary_)
    matrix_shape = X.shape
    print(f"[SHAPE] TF-IDF Matrix Shape: {matrix_shape[0]} documents x {matrix_shape[1]} terms")
    print(f"[VOCAB] Vocabulary Size: {vocab_size:,} unique words")
    
    print("\n[SAVE] Optimizing memory footprint (8 GB RAM optimization):")
    original_bytes = X.data.nbytes
    print(f"   - Original sparse matrix memory: {original_bytes / (1024*1024):.2f} MB")
    
    # Convert to float32 (halves memory)
    X = X.astype(np.float32)
    new_bytes = X.data.nbytes
    print(f"   - Converted to float32 matrix memory: {new_bytes / (1024*1024):.2f} MB (Saved 50%)")
    
    # 4. Save to Disk
    os.makedirs(settings.MODELS_DIR, exist_ok=True)
    out_path = os.path.join(settings.MODELS_DIR, f"{dataset_name}_tfidf.joblib")
    
    print(f"\n[SAVE] Dumping model and matrix to disk at: {out_path}...")
    save_start = time.time()
    joblib.dump({
        "vectorizer": vectorizer,
        "vectors": X,
        "doc_ids": df['doc_id'].tolist()
    }, out_path)
    print(f"[SAVE] Model dumped successfully in {time.time() - save_start:.2f} seconds.")
    
    # 5. Clean up references
    del df, X, vectorizer
    gc.collect()
    
    elapsed_time = time.time() - start_time
    print(f"\n[SUCCESS] TF-IDF Build completed in {elapsed_time:.2f} seconds.")
    print("="*60 + "\n")
    
    return {"status": "success", "file": out_path, "docs_count": total_docs, "vocab_size": vocab_size}
