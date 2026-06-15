import sqlite3
import os

BASE_DATA_DIR = "data"
DB_PATH = os.path.join(BASE_DATA_DIR, "ir_dataset_store.db")

def get_db_connection() -> sqlite3.Connection:
    """Creates a container-safe SQLite connection with optimized concurrent-read timeout."""
    os.makedirs(BASE_DATA_DIR, exist_ok=True)
    # تمديد الـ timeout إلى 60 ثانية يمنع الحاويات من الانهيار إذا كانت هناك خدمة أخرى تقرأ من الملف
    return sqlite3.connect(DB_PATH, timeout=60.0)

def init_database():
    """Initializes comprehensive storage schematics, relational entities, and key lookup indices."""
    query_create_documents = '''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_name TEXT,
            doc_id TEXT,
            text TEXT
        )
    '''
    
    query_create_queries = '''
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_name TEXT,
            query_id TEXT,
            text TEXT
        )
    '''
    
    query_create_qrels = '''
        CREATE TABLE IF NOT EXISTS qrels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_name TEXT,
            query_id TEXT,
            doc_id TEXT,
            relevance INTEGER
        )
    '''
    
    query_create_processed = '''
        CREATE TABLE IF NOT EXISTS processed_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_name TEXT,
            doc_id TEXT,
            processed_text TEXT
        )
    '''
    
    query_create_embeddings = '''
        CREATE TABLE IF NOT EXISTS document_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_name TEXT,
            doc_id TEXT,
            embedding_model_name TEXT,
            vector BLOB
        )
    '''

    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Build Schema Tables
        cursor.execute(query_create_documents)
        cursor.execute(query_create_queries)
        cursor.execute(query_create_qrels)
        cursor.execute(query_create_processed)
        cursor.execute(query_create_embeddings)
        
        # Build System Acceleration Lookup Indices
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_doc_search ON documents(dataset_name, doc_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_query_search ON queries(dataset_name, query_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_qrel_search ON qrels(dataset_name, query_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_processed_search ON processed_documents(dataset_name, doc_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_emb_search ON document_embeddings(dataset_name, doc_id, embedding_model_name)')
        
        conn.commit()
    print("⚙️ [DATABASE] Relational system schema and query optimization indexes verified/constructed successfully.")