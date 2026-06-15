from fastapi import FastAPI, HTTPException, BackgroundTasks
from shared.schemas import ClusteringRequest, ClusteringResponse
from shared.config import settings
import uvicorn
import time

app = FastAPI(title="Clustering Service", version="1.0.0")

clustering_jobs = {}

def run_clustering_task(dataset: str, num_clusters: int, algorithm: str):
    clustering_jobs[dataset] = "processing"
    try:
        # Mock clustering work
        time.sleep(3)
        clustering_jobs[dataset] = "completed"
    except Exception as e:
        clustering_jobs[dataset] = f"failed: {str(e)}"

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "clustering_service"}

@app.post("/cluster/build", response_model=ClusteringResponse)
def build_clusters(request: ClusteringRequest, background_tasks: BackgroundTasks):
    dataset = request.dataset
    if dataset in clustering_jobs and clustering_jobs[dataset] == "processing":
        return ClusteringResponse(
            dataset=dataset,
            status="processing",
            clusters_count=0,
            message="Clustering job is already in progress"
        )
        
    background_tasks.add_task(
        run_clustering_task, 
        dataset, 
        request.num_clusters, 
        request.algorithm
    )
    
    return ClusteringResponse(
        dataset=dataset,
        status="started",
        clusters_count=0,
        message=f"Clustering job started asynchronously using {request.algorithm}."
    )

@app.get("/cluster/status/{dataset}")
def get_clustering_status(dataset: str):
    status = clustering_jobs.get(dataset, "idle")
    return {"dataset": dataset, "status": status}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=settings.CLUSTERING_PORT, reload=True)
