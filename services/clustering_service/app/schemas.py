from pydantic import BaseModel
from typing import Optional

class ClusteringRequest(BaseModel):
    dataset: str
    num_clusters: int = 5
    algorithm: Optional[str] = "kmeans"

class ClusteringResponse(BaseModel):
    dataset: str
    status: str
    clusters_count: int
    message: str
