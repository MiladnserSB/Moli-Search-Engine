import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class VectorSearcher:

    def __init__(self):
        self.embeddings = None
        self.doc_ids = None
        self.raw_docs = None

    def load_index(self, embedding_model_path):

        data = joblib.load(embedding_model_path)

        self.embeddings = data["embeddings"]
        self.doc_ids = data["doc_ids"]
        self.raw_docs = data.get("raw_docs")
        # Precompute string mapping for O(1) candidate lookup in reranking
        self.doc_id_to_index = {str(doc_id): i for i, doc_id in enumerate(self.doc_ids)}

    def search(self, query_embedding, top_k=100):

        # Embeddings are L2 normalized, so cosine similarity is the dot product
        scores = np.dot(self.embeddings, query_embedding)

        # Optimize search using argpartition
        idx = np.argpartition(scores, -top_k)[-top_k:]
        idx = idx[np.argsort(scores[idx])[::-1]]

        return [
            {
                "id": self.doc_ids[i],
                "score": float(scores[i])
            }
            for i in idx
        ]

    def rerank(self, query_embedding, candidates):

        # Map candidate IDs to their exact index positions using the precomputed dictionary
        indices = []
        for doc in candidates:
            doc_id_str = str(doc["id"])
            if doc_id_str in self.doc_id_to_index:
                indices.append(self.doc_id_to_index[doc_id_str])

        if not indices:
            return []

        candidate_vectors = self.embeddings[indices]

        # candidate_vectors and query_embedding are L2 normalized, compute cosine similarity via dot product
        scores = np.dot(candidate_vectors, query_embedding)

        reranked = []
        for pos, idx in enumerate(indices):
            reranked.append({
                "id": self.doc_ids[idx],
                "score": float(scores[pos])
            })

        reranked.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return reranked

    def search_batch(self, query_embeddings, top_k=100):
        if len(query_embeddings) == 0:
            return []
        # scores has shape (num_docs, num_queries)
        scores = np.dot(self.embeddings, query_embeddings.T)
        
        batch_results = []
        num_queries = query_embeddings.shape[0]
        for q_idx in range(num_queries):
            q_scores = scores[:, q_idx]
            
            # Optimize batch sorting using argpartition
            idx = np.argpartition(q_scores, -top_k)[-top_k:]
            idx = idx[np.argsort(q_scores[idx])[::-1]]
            
            results = [
                {
                    "id": self.doc_ids[i],
                    "score": float(q_scores[i])
                }
                for i in idx
            ]
            batch_results.append(results)
            
        return batch_results