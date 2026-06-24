from pydantic import BaseModel
from typing import Optional

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

