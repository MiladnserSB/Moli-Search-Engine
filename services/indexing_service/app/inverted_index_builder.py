# Inverted Index Builder
import os
import gc
import json
import time
import sqlite3
import pandas as pd
from collections import defaultdict
from .config import settings

def build_inverted_index_representation(dataset_name: str) -> dict:
    start_time = time.time()
    print("\n" + "="*60)
    print(f"[INVERTED INDEX BUILDER] Starting build for dataset: {dataset_name.upper()}")
    print("="*60)
    
    # 1. Database Connection
    print("[DB_CONNECT] Connecting to SQLite database...")
    conn = sqlite3.connect(settings.DB_PATH)
    
    print("[DB_LOAD] Loading preprocessed texts from 'processed_documents' table...")
    df = pd.read_sql(
        "SELECT doc_id, processed_text FROM processed_documents WHERE dataset_name IN (?, ?)",
        conn,
        params=(dataset_name, f"{dataset_name}_classical")
    )
    conn.close()
    
    total_docs = len(df)
    print(f"[OK] Loaded {total_docs:,} processed documents.")
    
    if df.empty:
        raise ValueError(f"No processed documents found for dataset {dataset_name}")
        
    # 2. Build Inverted Index Map
    print("\n[MAPPING] Building standard inverted index (word -> set of doc_ids)...")
    index = defaultdict(set)
    doc_lengths = {}
    
    map_start = time.time()
    for idx, (doc_id, text) in enumerate(zip(df['doc_id'], df['processed_text'])):
        words = str(text).split()
        doc_lengths[doc_id] = len(words)
        for word in words:
            index[word].add(doc_id)
            
        # Print progress checkpoints
        if (idx + 1) % 100000 == 0 or (idx + 1) == total_docs:
            percentage = ((idx + 1) / total_docs) * 100
            print(f"   * Checkpoint: processed {idx+1:,} / {total_docs:,} docs ({percentage:.1f}%)")
            
    print(f"[OK] Inverted index mapping completed in {time.time() - map_start:.2f} seconds.")
    
    # 3. Stats Calculation
    vocab_size = len(index)
    total_postings = sum(len(postings) for postings in index.values())
    avg_postings = total_postings / vocab_size if vocab_size > 0 else 0
    
    print(f"\n[STATS] Inverted Index Stats:")
    print(f"   - Unique Index Vocabulary Terms: {vocab_size:,}")
    print(f"   - Total Postings Count: {total_postings:,}")
    print(f"   - Average Document Frequency per term: {avg_postings:.2f} docs/term")
    
    # 4. Serialize to JSON
    print("\n[CONVERSION] Converting dictionary postings sets to sorted lists for JSON...")
    serialize_start = time.time()
    # Convert sets to sorted lists for JSON serialization
    index_json = {k: sorted(list(v)) for k, v in index.items()}
    print(f"[OK] Conversion completed in {time.time() - serialize_start:.2f} seconds.")
    
    os.makedirs(settings.INDICES_DIR, exist_ok=True)
    out_path = os.path.join(settings.INDICES_DIR, f"{dataset_name}_index.json")
    
    print(f"\n[SAVE] Writing JSON inverted index file to disk at: {out_path}...")
    save_start = time.time()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(index_json, f)
    print(f"[SAVE] File written successfully in {time.time() - save_start:.2f} seconds.")
    
    # Save document lengths for BM25/VSM use
    lengths_path = os.path.join(settings.INDICES_DIR, f"{dataset_name}_lengths.json")
    print(f"[SAVE] Writing document lengths helper JSON file to disk at: {lengths_path}...")
    with open(lengths_path, "w", encoding="utf-8") as f:
        json.dump(doc_lengths, f)
    print("[SAVE] File written successfully.")
    
    # Clean memory
    del df, index, index_json, doc_lengths
    gc.collect()
    
    elapsed_time = time.time() - start_time
    print(f"\n[SUCCESS] Inverted Index Build completed in {elapsed_time:.2f} seconds.")
    print("="*60 + "\n")
    
    return {"status": "success", "file": out_path, "vocab_size": vocab_size}
