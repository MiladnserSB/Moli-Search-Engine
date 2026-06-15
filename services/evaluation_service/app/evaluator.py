# Evaluator Class: MAP, Recall, Precision@K, nDCG
import math

class IREvaluator:
    def __init__(self, qrels: dict[str, dict[str, int]]):
        # qrels is a dictionary mapping query_id -> {doc_id -> relevance_grade}
        self.qrels = qrels

    def calculate_precision_at_k(self, retrieved: list[str], query_id: str, k: int = 10) -> float:
        # P@K = (Number of relevant docs in top K) / K
        return 0.0

    def calculate_recall(self, retrieved: list[str], query_id: str) -> float:
        # Recall = (Number of relevant retrieved docs) / (Total relevant docs)
        return 0.0

    def calculate_map(self, retrieved_runs: dict[str, list[str]]) -> float:
        # Mean Average Precision across all queries
        return 0.0

    def calculate_ndcg(self, retrieved: list[str], query_id: str, k: int = 10) -> float:
        # Normalized Discounted Cumulative Gain
        # DCG@K = sum ( (2^rel - 1) / log2(i + 1) )
        # NDCG@K = DCG@K / IDCG@K
        return 0.0
