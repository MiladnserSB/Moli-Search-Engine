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
    method: str  # 'bm25', 'vsm', 'embedding', 'hybrid_serial', 'hybrid_parallel'
    bm25_k1: Optional[float] = 1.5
    bm25_b: Optional[float] = 0.75
    use_additional_features: Optional[bool] = False
    top_k: Optional[int] = 10

class SearchResponse(BaseModel):
    results: List[DocumentSchema]
    refined_query: Optional[str] = None
    time_taken_ms: float
