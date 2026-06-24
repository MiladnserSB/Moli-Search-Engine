from .fusion import reciprocal_rank_fusion


class HybridSearcher:

    def __init__(self, sparse_scorer, dense_searcher):
        self.sparse_scorer = sparse_scorer
        self.dense_searcher = dense_searcher

    def search_serial(
        self,
        query_tokens,
        query_embedding,
        top_k=10,
        inverted_index=None
    ):

        candidates = self.sparse_scorer.search(
            query_tokens,
            top_k=100,
            inverted_index=inverted_index
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
        top_k=10,
        inverted_index=None,
        hybrid_alpha=0.5
    ):

        sparse_results = self.sparse_scorer.search(
            query_tokens,
            top_k=100,
            inverted_index=inverted_index
        )

        dense_results = self.dense_searcher.search(
            query_embedding,
            top_k=100
        )

        fused = reciprocal_rank_fusion(
            [sparse_results, dense_results],
            weights=[hybrid_alpha, 1.0 - hybrid_alpha]
        )

        return fused[:top_k]

    def search_batch_serial(
        self,
        queries_tokens,
        query_embeddings,
        top_k=10,
        inverted_index=None
    ):
        sparse_batch_candidates = self.sparse_scorer.search_batch(
            queries_tokens,
            top_k=100,
            inverted_index=inverted_index
        )
        
        batch_results = []
        for q_idx, candidates in enumerate(sparse_batch_candidates):
            reranked = self.dense_searcher.rerank(
                query_embeddings[q_idx],
                candidates
            )
            batch_results.append(reranked[:top_k])
            
        return batch_results

    def search_batch_parallel(
        self,
        queries_tokens,
        query_embeddings,
        top_k=10,
        inverted_index=None,
        hybrid_alpha=0.5
    ):
        sparse_batch_results = self.sparse_scorer.search_batch(
            queries_tokens,
            top_k=100,
            inverted_index=inverted_index
        )
        
        dense_batch_results = self.dense_searcher.search_batch(
            query_embeddings,
            top_k=100
        )
        
        batch_results = []
        for s_res, d_res in zip(sparse_batch_results, dense_batch_results):
            fused = reciprocal_rank_fusion(
                [s_res, d_res],
                weights=[hybrid_alpha, 1.0 - hybrid_alpha]
            )
            batch_results.append(fused[:top_k])
            
        return batch_results