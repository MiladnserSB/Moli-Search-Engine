import os

class Settings:
    DATA_DIR = "/app/data"
    if not os.path.exists(DATA_DIR):
        DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))
        
    DATASETS_DIR = DATA_DIR
    INDICES_DIR = os.getenv("INDICES_DIR", os.path.join(DATA_DIR, "indices"))
    MODELS_DIR = os.getenv("MODELS_DIR", os.path.join(DATA_DIR, "models"))
    
    # SQLite Database Location (Works locally and in Docker container)
    DB_PATH = os.path.join(DATA_DIR, "ir_dataset_store.db")
        
    INDEXING_PORT = int(os.getenv("INDEXING_PORT", "8002"))

settings = Settings()
