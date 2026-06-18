from pydantic import BaseModel
from typing import Optional

class BM25BuildRequest(BaseModel):
    k1: Optional[float] = 1.5
    b: Optional[float] = 0.75

class BuildResponse(BaseModel):
    status: str
    message: str
    dataset: str
    details: Optional[dict] = None
