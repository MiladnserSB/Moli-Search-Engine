# Matching & Scoring implementations: TF-IDF and BM25
import math

class TFIDFScorer:
    def __init__(self, index):
        self.index = index # Inverted index terms -> list of (doc_id, tf)
        self.doc_lengths = {}
        
    def score(self, query_tokens: list[str], doc_id: str) -> float:
        # Calculate cosine similarity using Term Frequency - Inverse Document Frequency
        return 0.0

class BM25Scorer:
    def __init__(self, index, doc_lengths, avg_doc_len, k1=1.5, b=0.75):
        self.index = index
        self.doc_lengths = doc_lengths
        self.avg_doc_len = avg_doc_len
        self.k1 = k1
        self.b = b
        
    def score(self, query_tokens: list[str], doc_id: str) -> float:
        # Calculate BM25 score for the query in doc_id
        # Score = Sum ( IDF * TF * (k1 + 1) / (TF + k1 * (1 - b + b * (doc_len / avg_doc_len))) )
        return 0.0
