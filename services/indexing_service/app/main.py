from fastapi import FastAPI, HTTPException, BackgroundTasks
from shared.config import settings
import uvicorn
import os

app = FastAPI(title="Indexing Service", version="1.0.0")

# Simple in-memory tracker for indexing status
index_status = {}

def build_index_task(dataset_name: str):
    index_status[dataset_name] = "building"
    try:
        # Placeholder logic: ensure shared data datasets folder path is accessible
        dataset_path = os.path.join(settings.DATASETS_DIR, dataset_name)
        
        # Simulate index building
        import time
        time.sleep(2) # Mock indexing work
        
        index_status[dataset_name] = "ready"
    except Exception as e:
        index_status[dataset_name] = f"error: {str(e)}"

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "indexing_service"}

@app.post("/index/build/{dataset_name}")
def build_index(dataset_name: str, background_tasks: BackgroundTasks):
    if dataset_name in index_status and index_status[dataset_name] == "building":
        return {"message": "Indexing is already in progress"}
        
    background_tasks.add_task(build_index_task, dataset_name)
    return {"message": f"Index building task started for dataset: {dataset_name}"}

@app.get("/index/status/{dataset_name}")
def get_index_status(dataset_name: str):
    status = index_status.get(dataset_name, "not_built")
    return {"dataset": dataset_name, "status": status}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=settings.INDEXING_PORT, reload=True)
