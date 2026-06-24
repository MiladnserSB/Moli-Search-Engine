import os

class Settings:
    DATA_DIR = "/app/data"
    if not os.path.exists(DATA_DIR):
        DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))
        
    MODELS_DIR = os.getenv("MODELS_DIR", os.path.join(DATA_DIR, "models"))
    INDICES_DIR = os.getenv("INDICES_DIR", os.path.join(DATA_DIR, "indices"))
    
    # SQLite Database Location
    DB_PATH = os.path.join(DATA_DIR, "ir_dataset_store.db")
        
    RETRIEVAL_PORT = int(os.getenv("RETRIEVAL_PORT", "8003"))

settings = Settings()
