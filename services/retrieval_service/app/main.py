# from fastapi import FastAPI, HTTPException
# from shared.schemas import QueryRequest, SearchResponse, DocumentSchema
# from shared.config import settings
# import uvicorn
# import time

# app = FastAPI(title="Retrieval Service", version="1.0.0")

# @app.get("/health")
# def health_check():
#     return {"status": "healthy", "service": "retrieval_service"}

# @app.post("/search", response_model=SearchResponse)
# def search_documents(request: QueryRequest):
#     start_time = time.time()
#     try:
#         # Mock documents retrieval response
#         mock_results = [
#             DocumentSchema(
#                 id="doc_001",
#                 title="Introduction to Information Retrieval",
#                 content="This document covers basic concepts of IR including indexing and vector space model.",
#                 score=0.92
#             ),
#             DocumentSchema(
#                 id="doc_002",
#                 title="Understanding BM25",
#                 content="BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms.",
#                 score=0.85
#             ),
#             DocumentSchema(
#                 id="doc_003",
#                 title="Dense Retrieval & Vector Stores",
#                 content="Dense passage retrieval uses neural networks to map text into a vector space for similarity search.",
#                 score=0.74
#             )
#         ]
        
#         # Sort mock results by score
#         mock_results = sorted(mock_results, key=lambda x: x.score, reverse=True)[:request.top_k]
        
#         time_taken_ms = (time.time() - start_time) * 1000
        
#         return SearchResponse(
#             results=mock_results,
#             refined_query=request.query,
#             time_taken_ms=time_taken_ms
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=settings.RETRIEVAL_PORT, reload=True)

from fastapi import FastAPI, HTTPException
from shared.schemas import QueryRequest, SearchResponse, DocumentSchema
from shared.config import settings

from app.search_models import BM25Searcher

import uvicorn
import time

app = FastAPI(
    title="Retrieval Service",
    version="1.0.0"
)

# تحميل BM25 مرة واحدة عند تشغيل السيرفر
bm25_searcher = BM25Searcher(
    f"{settings.MODELS_DIR}/quora_dev_bm25.joblib"
)

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "retrieval_service"
    }

@app.post("/search")
def search_documents(request: QueryRequest):
    start_time = time.time()
    try:
        query_tokens = request.query.split()
        results = bm25_searcher.search(
            query_tokens=query_tokens,
            top_k=request.top_k
        )
        time_taken_ms = (time.time() - start_time) * 1000
        return {
            "results": results,
            "refined_query": request.query,
            "time_taken_ms": time_taken_ms
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.RETRIEVAL_PORT,
        reload=True
    )
