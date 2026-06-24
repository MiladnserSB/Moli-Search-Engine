import os

class Settings:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    
    INDICES_DIR = os.path.abspath(os.path.join(BASE_DIR, "services", "indexing_service", "data", "indices"))
    if not os.path.exists(INDICES_DIR):
        INDICES_DIR = os.path.abspath(os.path.join(BASE_DIR, "data", "indices"))
        
    DB_PATH = "/app/data/ir_dataset_store.db"
    if not os.path.exists(os.path.dirname(DB_PATH)):
        DB_PATH = os.path.abspath(os.path.join(BASE_DIR, "data", "ir_dataset_store.db"))
        
    REFINEMENT_PORT = int(os.getenv("REFINEMENT_PORT", "8006"))

settings = Settings()
