# BM25 Representation Builder
import os
import gc
import time
import sqlite3
import joblib
import pandas as pd
from rank_bm25 import BM25Okapi
from .config import settings

def build_bm25_representation(dataset_name: str, k1: float = 1.5, b: float = 0.75) -> dict:
    start_time = time.time()
    print("\n" + "="*60)
    print(f"📊 [BM25 BUILDER] Starting build for dataset: {dataset_name.upper()}")
    print(f"   Parameters: k1={k1}, b={b}")
    print("="*60)
    
    # 1. Database Connection
    print("🔌 Connecting to SQLite database...")
    conn = sqlite3.connect(settings.DB_PATH)
    
    # Match both standard dataset_name and with _classical suffix
    print("📥 Loading preprocessed texts from 'processed_documents' table...")
    df = pd.read_sql(
        "SELECT doc_id, processed_text FROM processed_documents WHERE dataset_name IN (?, ?)",
        conn,
        params=(dataset_name, f"{dataset_name}_classical")
    )
    conn.close()
    
    total_docs = len(df)
    print(f"✅ Loaded {total_docs:,} processed documents.")
    
    if df.empty:
        raise ValueError(f"No processed documents found for dataset {dataset_name}")
        
    # 2. Tokenize corpus
    print("\n⚡ Tokenizing preprocessed corpus (splitting by whitespace)...")
    tok_start = time.time()
    tokenized_corpus = [str(text).split() for text in df['processed_text']]
    print(f"✅ Tokenized {len(tokenized_corpus):,} docs in {time.time() - tok_start:.2f} seconds.")
    
    # 3. Fit BM25 Okapi index
    print("\n⏳ Fitting BM25Okapi index (calculating document frequencies and stats)...")
    fit_start = time.time()
    bm25_vectorizer = BM25Okapi(tokenized_corpus, k1=k1, b=b)
    fit_duration = time.time() - fit_start
    print(f"✅ BM25 index fit completed in {fit_duration:.2f} seconds.")
    
    # Print BM25 Stats
    print(f"   - Average Document Length (avgdl): {bm25_vectorizer.avgdl:.2f} tokens")
    print(f"   - Total documents indexed: {bm25_vectorizer.corpus_size:,}")
    
    # 4. Save to Disk
    os.makedirs(settings.MODELS_DIR, exist_ok=True)
    out_path = os.path.join(settings.MODELS_DIR, f"{dataset_name}_bm25.joblib")
    
    print(f"\n💾 Dumping BM25 model, tokenized corpus, and doc mappings to disk at: {out_path}...")
    save_start = time.time()
    joblib.dump({
        "bm25_vectorizer": bm25_vectorizer,
        "doc_ids": df['doc_id'].tolist(),
        "tokenized_corpus": tokenized_corpus
    }, out_path)
    print(f"💾 Model dumped successfully in {time.time() - save_start:.2f} seconds.")
    
    # Clean memory
    del df, tokenized_corpus, bm25_vectorizer
    gc.collect()
    
    elapsed_time = time.time() - start_time
    print(f"\n🎉 [BM25 SUCCESS] Build completed in {elapsed_time:.2f} seconds.")
    print("="*60 + "\n")
    
    return {"status": "success", "file": out_path, "docs_count": total_docs}
