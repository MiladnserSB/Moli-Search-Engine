import os

class Settings:
    DATA_DIR = os.getenv("DATA_DIR", "data")
    MODELS_DIR = os.getenv("MODELS_DIR", os.path.join(DATA_DIR, "models"))
    INDICES_DIR = os.getenv("INDICES_DIR", os.path.join(DATA_DIR, "indices"))
    
    # SQLite Database Location
    DB_PATH = "/app/data/ir_dataset_store.db"
    if not os.path.exists(os.path.dirname(DB_PATH)):
        DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "ir_dataset_store.db"))
        
    RETRIEVAL_PORT = int(os.getenv("RETRIEVAL_PORT", "8003"))

settings = Settings()
