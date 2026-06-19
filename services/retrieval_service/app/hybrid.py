from .fusion import reciprocal_rank_fusion


class HybridSearcher:

    def __init__(self, sparse_scorer, dense_searcher):
        self.sparse_scorer = sparse_scorer
        self.dense_searcher = dense_searcher

    def search_serial(
        self,
        query_tokens,
        query_embedding,
        top_k=10
    ):

        candidates = self.sparse_scorer.search(
            query_tokens,
            top_k=100
        )

        reranked = self.dense_searcher.rerank(
            query_embedding,
            candidates
        )

        return reranked[:top_k]

    def search_parallel(
        self,
        query_tokens,
        query_embedding,
        top_k=10
    ):

        sparse_results = self.sparse_scorer.search(
            query_tokens,
            top_k=100
        )

        dense_results = self.dense_searcher.search(
            query_embedding,
            top_k=100
        )

        fused = reciprocal_rank_fusion(
            [sparse_results, dense_results]
        )

        return fused[:top_k]