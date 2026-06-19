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
        self.raw_docs = data["raw_docs"]

    def search(self, query_embedding, top_k=100):

        scores = cosine_similarity(
            [query_embedding],
            self.embeddings
        ).flatten()

        idx = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "id": self.doc_ids[i],
                "score": float(scores[i])
            }
            for i in idx
        ]

    def rerank(self, query_embedding, candidates):

        candidate_ids = {
            str(doc["id"])
            for doc in candidates
        }

        indices = [
            i
            for i, doc_id in enumerate(self.doc_ids)
            if str(doc_id) in candidate_ids
        ]

        candidate_vectors = self.embeddings[indices]

        scores = cosine_similarity(
            [query_embedding],
            candidate_vectors
        ).flatten()

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