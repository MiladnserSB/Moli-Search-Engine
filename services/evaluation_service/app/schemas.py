from pydantic import BaseModel
from typing import Optional

class EvaluationRequest(BaseModel):
    dataset: str
    method: str
    use_additional_features: bool = False
    bm25_k1: Optional[float] = 1.5
    bm25_b: Optional[float] = 0.75
    hybrid_alpha: Optional[float] = 0.5
    # "offline" = instant from pre-computed cache (all queries)
    # "online"  = live retrieval via Gateway (online_limit queries)
    mode: str = "offline"
    online_limit: int = 500

class EvaluationResponse(BaseModel):
    dataset: str
    map_score: float
    recall_score: float
    precision_at_k: float
    ndcg_score: float
    num_queries_evaluated: Optional[int] = None
    mode: Optional[str] = None
