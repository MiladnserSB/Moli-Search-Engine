import os

class Settings:
    DATA_DIR = os.getenv("DATA_DIR", "data")
    DATASETS_DIR = os.getenv("DATASETS_DIR", "data")
    INDICES_DIR = os.getenv("INDICES_DIR", os.path.join(DATA_DIR, "indices"))
    MODELS_DIR = os.getenv("MODELS_DIR", os.path.join(DATA_DIR, "models"))
    
    # SQLite Database Location (Works locally and in Docker container)
    DB_PATH = "/app/data/ir_dataset_store.db"
    if not os.path.exists(os.path.dirname(DB_PATH)):
        DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "ir_dataset_store.db"))
        
    INDEXING_PORT = int(os.getenv("INDEXING_PORT", "8002"))

settings = Settings()
