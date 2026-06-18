# BERT Embeddings Representation Builder
import os
import gc
import time
import sqlite3
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from .preprocessing import clean_light
from .config import settings

def build_bert_representation(dataset_name: str) -> dict:
    start_time = time.time()
    print("\n" + "="*60)
    print(f"[BERT EMBEDDINGS BUILDER] Starting build for dataset: {dataset_name.upper()}")
    print("="*60)
    
    # 1. Database Connection
    print("[DB_CONNECT] Connecting to SQLite database...")
    conn = sqlite3.connect(settings.DB_PATH)
    print("[DB_LOAD] Loading raw documents from 'documents' table...")
    df = pd.read_sql(
        "SELECT doc_id, text FROM documents WHERE dataset_name = ? AND text IS NOT NULL", 
        conn, 
        params=(dataset_name,)
    )
    conn.close()
    
    total_docs = len(df)
    print(f"[OK] Loaded {total_docs:,} documents from database.")
    
    if df.empty:
        raise ValueError(f"No documents found for dataset {dataset_name}")
        
    # 2. Model Loading
    print("\n[MODEL] Loading pre-trained SentenceTransformer model 'all-MiniLM-L6-v2'...")
    model_start = time.time()
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"[OK] Model loaded successfully in {time.time() - model_start:.2f} seconds.")
    
    # 3. Clean Light Texts on the Fly
    print("\n[CLEANING] Applying lightweight cleaning (clean_light) on the fly...")
    texts = [clean_light(t) for t in df['text']]
    print("[OK] Light cleaning completed.")
    
    # 4. Encoding in Batches (8 GB RAM Safe)
    print(f"\n[ENCODING] Initiating BERT encoding in batches of 500 documents (Memory Optimized)...")
    embeddings = []
    batch_size = 500
    
    enc_start = time.time()
    for idx in range(0, total_docs, batch_size):
        batch_texts = texts[idx:idx+batch_size]
        batch_emb = model.encode(
            batch_texts, 
            convert_to_numpy=True, 
            normalize_embeddings=True, 
            show_progress_bar=False
        )
        
        # Cast to float32 immediately to protect RAM
        embeddings.append(batch_emb.astype(np.float32))
        
        # Print progress every 2,500 docs
        processed = idx + len(batch_texts)
        if processed % 2500 == 0 or processed == total_docs:
            percentage = (processed / total_docs) * 100
            elapsed = time.time() - enc_start
            speed = processed / elapsed if elapsed > 0 else 0
            print(f"   * Progress: {processed:,} / {total_docs:,} docs encoded ({percentage:.1f}%) | Speed: {speed:.1f} docs/sec")
            
    print("[OK] BERT encoding completed!")
    
    # Stack list of batches into a single large matrix
    print("[STACKING] Stacking batch embeddings into a single dense matrix...")
    embeddings = np.vstack(embeddings)
    
    matrix_shape = embeddings.shape
    memory_size_mb = embeddings.nbytes / (1024 * 1024)
    print(f"[SHAPE] BERT Embedding Matrix Shape: {matrix_shape[0]} documents x {matrix_shape[1]} dimensions")
    print(f"[SAVE] In-Memory Matrix size: {memory_size_mb:.2f} MB")
    
    # 5. Save to Disk
    os.makedirs(settings.MODELS_DIR, exist_ok=True)
    out_path = os.path.join(settings.MODELS_DIR, f"{dataset_name}_embedding.joblib")
    
    print(f"\n[SAVE] Dumping BERT model embeddings and doc mapping to disk at: {out_path}...")
    save_start = time.time()
    joblib.dump({
        "embeddings": embeddings,
        "doc_ids": df['doc_id'].tolist(),
        "raw_docs": dict(zip(df['doc_id'], df['text']))
    }, out_path)
    print(f"[SAVE] Model dumped successfully in {time.time() - save_start:.2f} seconds.")
    
    # Clean memory
    del df, texts, embeddings, model
    gc.collect()
    
    elapsed_time = time.time() - start_time
    print(f"\n[SUCCESS] BERT Embeddings Build completed in {elapsed_time:.2f} seconds.")
    print("="*60 + "\n")
    
    return {"status": "success", "file": out_path, "docs_count": total_docs, "dimensions": matrix_shape[1]}
