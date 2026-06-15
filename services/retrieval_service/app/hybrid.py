# Hybrid Retrieval Orchestration: Serial and Parallel Methods

class HybridSearcher:
    def __init__(self, sparse_scorer, dense_searcher):
        self.sparse_scorer = sparse_scorer
        self.dense_searcher = dense_searcher

    def search_serial(self, query_tokens: list[str], query_emb: list[float], top_k: int = 10) -> list[dict]:
        # Step 1: Retrieve top M candidates using sparse retrieval (BM25/TF-IDF)
        # Step 2: Rerank those candidates using dense embedding similarity
        return []

    def search_parallel(self, query_tokens: list[str], query_emb: list[float], top_k: int = 10) -> list[dict]:
        # Step 1: Query sparse scorer (e.g., BM25) -> sparse results list
        # Step 2: Query dense vector search -> dense results list
        # Step 3: Run fusion method (e.g., RRF) to compute combined scores
        return []
