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
        inverted_index=None,
        hybrid_alpha=0.5,  # ✅ تم إضافة هذا الباراميتر
    ):
        # 1. جلب أفضل 100 مرشح من BM25 و Dense (بدلاً من الاعتماد على BM25 فقط)
        sparse_results = self.sparse_scorer.search(
            query_tokens, top_k=100, inverted_index=inverted_index
        )
        dense_results = self.dense_searcher.search(query_embedding, top_k=100)

        # 2. دمج أولي (Pre-Fusion) لتجنب "Garbage In, Garbage Out"
        prefused_candidates = reciprocal_rank_fusion(
            [sparse_results, dense_results], weights=[hybrid_alpha, 1.0 - hybrid_alpha]
        )[:100]  # نأخذ أفضل 100 بعد الدمج

        # 3. إعادة ترتيب المرشحين المُدمَجين باستخدام Dense
        reranked = self.dense_searcher.rerank(query_embedding, prefused_candidates)

        return reranked[:top_k]

    def search_parallel(
        self,
        query_tokens,
        query_embedding,
        top_k=10,
        inverted_index=None,
        hybrid_alpha=0.5,
    ):
        sparse_results = self.sparse_scorer.search(
            query_tokens, top_k=100, inverted_index=inverted_index
        )

        dense_results = self.dense_searcher.search(query_embedding, top_k=100)

        fused = reciprocal_rank_fusion(
            [sparse_results, dense_results], weights=[hybrid_alpha, 1.0 - hybrid_alpha]
        )

        return fused[:top_k]

    def search_batch_serial(
        self,
        queries_tokens,
        query_embeddings,
        top_k=10,
        inverted_index=None,
        hybrid_alpha=0.5,  # ✅ تم إضافة هذا الباراميتر
    ):
        sparse_batch_candidates = self.sparse_scorer.search_batch(
            queries_tokens, top_k=100, inverted_index=inverted_index
        )

        dense_batch_candidates = self.dense_searcher.search_batch(
            query_embeddings, top_k=100
        )

        batch_results = []
        for q_idx, (s_cands, d_cands) in enumerate(
            zip(sparse_batch_candidates, dense_batch_candidates)
        ):
            # دمج أولي (Pre-Fusion) لكل استعلام في الدفعة
            prefused_candidates = reciprocal_rank_fusion(
                [s_cands, d_cands], weights=[hybrid_alpha, 1.0 - hybrid_alpha]
            )[:100]

            reranked = self.dense_searcher.rerank(
                query_embeddings[q_idx], prefused_candidates
            )
            batch_results.append(reranked[:top_k])

        return batch_results

    def search_batch_parallel(
        self,
        queries_tokens,
        query_embeddings,
        top_k=10,
        inverted_index=None,
        hybrid_alpha=0.5,
    ):
        sparse_batch_results = self.sparse_scorer.search_batch(
            queries_tokens, top_k=100, inverted_index=inverted_index
        )

        dense_batch_results = self.dense_searcher.search_batch(
            query_embeddings, top_k=100
        )

        batch_results = []
        for s_res, d_res in zip(sparse_batch_results, dense_batch_results):
            fused = reciprocal_rank_fusion(
                [s_res, d_res], weights=[hybrid_alpha, 1.0 - hybrid_alpha]
            )
            batch_results.append(fused[:top_k])

        return batch_results
