import os

class Settings:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    DB_PATH = os.path.join(DATA_DIR, "ir_dataset_store.db")
    CLUSTERING_PORT = int(os.getenv("CLUSTERING_PORT", "8004"))

settings = Settings()
