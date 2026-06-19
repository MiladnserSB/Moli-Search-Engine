import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFSearcher:
    def __init__(self, model_path):
        data = joblib.load(model_path)

        self.vectorizer = data["vectorizer"]
        self.doc_vectors = data["vectors"]
        self.doc_ids = data["doc_ids"]

    def search(self, query_text, top_k=100):
        q = self.vectorizer.transform([query_text])

        scores = cosine_similarity(q, self.doc_vectors).flatten()

        idx = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "id": self.doc_ids[i],
                "score": float(scores[i])
            }
            for i in idx
        ]


class BM25Searcher:
    def __init__(self, model_path):
        data = joblib.load(model_path)

        self.bm25 = data["bm25_vectorizer"]
        self.doc_ids = data["doc_ids"]

    def search(self, query_tokens, top_k=100):
        scores = self.bm25.get_scores(query_tokens)

        idx = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "id": self.doc_ids[i],
                "score": float(scores[i])
            }
            for i in idx
        ]