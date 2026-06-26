import os

class Settings:
    DATA_DIR = os.getenv("DATA_DIR", "data")
    DB_PATH = os.getenv("DB_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "ir_dataset_store.db")))
    CACHE_PATH = os.getenv("CACHE_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "evaluation_results.json")))
    GATEWAY_URL = os.getenv("GATEWAY_URL", "http://127.0.0.1:8000")
    REFINEMENT_SERVICE_URL = os.getenv("REFINEMENT_SERVICE_URL", "http://127.0.0.1:8006")
    EVALUATION_PORT = int(os.getenv("EVALUATION_PORT", "8005"))

settings = Settings()
