# Evaluator Class: MAP, Recall, Precision@K, nDCG
import math


class IREvaluator:
    def __init__(self, qrels: dict[str, dict[str, int]]):
        # qrels is a dictionary mapping query_id -> {doc_id -> relevance_grade}
        self.qrels = qrels

    def calculate_precision_at_k(
        self, retrieved: list[str], query_id: str, k: int = 10
    ) -> float:
        # P@K = (Number of relevant docs in top K) / K
        query_qrels = self.qrels.get(query_id, {})
        if not query_qrels:
            return 0.0

        # Take top K retrieved
        top_k = retrieved[:k]
        relevant_count = sum(1 for doc in top_k if query_qrels.get(doc, 0) > 0)
        return relevant_count / k

    def calculate_recall(self, retrieved: list[str], query_id: str) -> float:
        # Recall = (Number of relevant retrieved docs) / (Total relevant docs)
        query_qrels = self.qrels.get(query_id, {})
        total_relevant = sum(1 for grade in query_qrels.values() if grade > 0)
        if total_relevant == 0:
            return 0.0

        relevant_retrieved = sum(1 for doc in retrieved if query_qrels.get(doc, 0) > 0)
        return relevant_retrieved / total_relevant

    def calculate_ap(self, retrieved: list[str], query_id: str, k: int = 10) -> float:
        # AP@K = sum_{i=1}^K (P@i * rel(i)) / total_relevant
        query_qrels = self.qrels.get(query_id, {})
        total_relevant = sum(1 for grade in query_qrels.values() if grade > 0)
        if total_relevant == 0:
            return 0.0

        ap_sum = 0.0
        relevant_found = 0

        limit = min(len(retrieved), k)
        for i in range(limit):
            doc = retrieved[i]
            rank = i + 1
            if query_qrels.get(doc, 0) > 0:
                relevant_found += 1
                precision_at_rank = relevant_found / rank
                ap_sum += precision_at_rank

        return ap_sum / total_relevant

    def calculate_map(self, retrieved_runs: dict[str, list[str]], k: int = 10) -> float:
        # Mean Average Precision across all queries in retrieved_runs
        if not retrieved_runs:
            return 0.0
        ap_scores = []
        for qid, retrieved in retrieved_runs.items():
            ap_scores.append(self.calculate_ap(retrieved, qid, k))
        return sum(ap_scores) / len(ap_scores)

    def calculate_ndcg(self, retrieved: list[str], query_id: str, k: int = 10) -> float:
        # Normalized Discounted Cumulative Gain
        # DCG@K = sum ( (2^rel - 1) / log2(rank + 1) )
        # NDCG@K = DCG@K / IDCG@K
        query_qrels = self.qrels.get(query_id, {})
        if not query_qrels:
            return 0.0

        limit = min(len(retrieved), k)
        dcg = 0.0
        for i in range(limit):
            doc = retrieved[i]
            rank = i + 1
            rel = query_qrels.get(doc, 0)
            dcg += (2**rel - 1) / math.log2(rank + 1)

        # Compute Ideal DCG (IDCG)
        # Sort all relevant documents in descending order of relevance
        all_rels = sorted(
            [grade for grade in query_qrels.values() if grade > 0], reverse=True
        )
        if not all_rels:
            return 0.0

        idcg_limit = min(len(all_rels), k)
        idcg = 0.0
        for i in range(idcg_limit):
            rel = all_rels[i]
            rank = i + 1
            idcg += (2**rel - 1) / math.log2(rank + 1)

        if idcg == 0.0:
            return 0.0

        return dcg / idcg
