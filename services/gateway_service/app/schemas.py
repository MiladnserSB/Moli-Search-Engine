from pydantic import BaseModel
from typing import List, Optional

class DocumentSchema(BaseModel):
    id: str
    title: Optional[str] = None
    content: str
    score: float

class QueryRequest(BaseModel):
    query: str
    dataset: str
    method: str
    preprocessed_query: Optional[str] = None
    bm25_k1: Optional[float] = 1.5
    bm25_b: Optional[float] = 0.75
    hybrid_alpha: Optional[float] = 0.5
    use_additional_features: Optional[bool] = False
    top_k: Optional[int] = 10
    retrieve_text: Optional[bool] = True

class SearchResponse(BaseModel):
    results: List[DocumentSchema]
    refined_query: Optional[str] = None
    time_taken_ms: float

class BatchQueryRequest(BaseModel):
    queries: List[str]
    dataset: str
    method: str
    preprocessed_queries: Optional[List[str]] = None
    bm25_k1: Optional[float] = 1.5
    bm25_b: Optional[float] = 0.75
    hybrid_alpha: Optional[float] = 0.5
    use_additional_features: Optional[bool] = False
    top_k: Optional[int] = 10
    retrieve_text: Optional[bool] = False

class BatchQueryResult(BaseModel):
    query: str
    results: List[DocumentSchema]

class BatchSearchResponse(BaseModel):
    results: List[BatchQueryResult]
    time_taken_ms: float


class EvaluationRequest(BaseModel):
    dataset: str
    method: str
    use_additional_features: bool = False
    bm25_k1: Optional[float] = 1.5
    bm25_b: Optional[float] = 0.75
    hybrid_alpha: Optional[float] = 0.5

class EvaluationResponse(BaseModel):
    dataset: str
    map_score: float
    recall_score: float
    precision_at_k: float
    ndcg_score: float


class ClusteringRequest(BaseModel):
    dataset: str
    num_clusters: int = 5
    algorithm: Optional[str] = "kmeans"

class ClusteringResponse(BaseModel):
    dataset: str
    status: str
    clusters_count: int
    message: str
