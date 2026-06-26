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










# import math

# try:
#     import ir_measures
#     from ir_measures import AP, NDCG, Precision, Recall

#     USE_IR_MEASURES = True
# except ImportError:
#     USE_IR_MEASURES = False


# class IREvaluator:
#     def __init__(self, qrels: dict[str, dict[str, int]]):
#         self.qrels = qrels

#         # تحسين: تجهيز مجموعات المستندات ذات الصلة وعددها لكل استعلام
#         self.relevant_docs_set = {}
#         self.total_relevant_counts = {}
#         self.cached_idcg = {}

#         for qid, docs in qrels.items():
#             rel_docs = {doc_id for doc_id, rel in docs.items() if rel > 0}
#             self.relevant_docs_set[qid] = rel_docs
#             self.total_relevant_counts[qid] = len(rel_docs)

#         if USE_IR_MEASURES:
#             self.ir_qrels = []
#             for qid, docs in qrels.items():
#                 for doc_id, rel in docs.items():
#                     self.ir_qrels.append(
#                         ir_measures.Qrel(query_id=qid, doc_id=doc_id, relevance=rel)
#                     )

#         # ذاكرة تخزين مؤقت لكل استعلام: {query_id: {k: (p, r, ap, ndcg)}}
#         self._metric_cache = {}

#     def _convert_to_ir_run(self, retrieved: list[str], query_id: str):
#         """تحويل القائمة إلى صيغة ScoredDoc التي تفهمها ir_measures"""
#         length = len(retrieved)
#         return [
#             ir_measures.ScoredDoc(
#                 query_id=query_id, doc_id=doc_id, score=float(length - i)
#             )
#             for i, doc_id in enumerate(retrieved)
#         ]

#     def _get_cached_idcg(self, query_id: str, k: int) -> float:
#         """حساب وتخزين الـ IDCG لكل استعلام لتجنب التكرار"""
#         cache_key = (query_id, k)
#         if cache_key in self.cached_idcg:
#             return self.cached_idcg[cache_key]

#         query_qrels = self.qrels.get(query_id, {})
#         all_rels = sorted(
#             [grade for grade in query_qrels.values() if grade > 0], reverse=True
#         )
#         if not all_rels:
#             self.cached_idcg[cache_key] = 0.0
#             return 0.0

#         idcg_limit = min(len(all_rels), k)
#         idcg = sum((2 ** all_rels[i] - 1) / math.log2(i + 2) for i in range(idcg_limit))
#         self.cached_idcg[cache_key] = idcg
#         return idcg

#     def _compute_all_metrics(self, retrieved: list[str], query_id: str, k: int):
#         """حساب جميع المقاييس للاستعلام الواحد وتخزينها"""
#         if USE_IR_MEASURES:
#             run = self._convert_to_ir_run(retrieved, query_id)
#             metrics = [
#                 Precision(cutoff=k),
#                 Recall(cutoff=k),
#                 AP(cutoff=k),
#                 NDCG(cutoff=k),
#             ]
#             result = ir_measures.calc_aggregate(metrics, self.ir_qrels, run)
#             p = result.get(Precision(cutoff=k), 0.0)
#             r = result.get(Recall(cutoff=k), 0.0)
#             ap = result.get(AP(cutoff=k), 0.0)
#             ndcg = result.get(NDCG(cutoff=k), 0.0)
#         else:
#             # الحساب اليدوي المجمع (مرور واحد على القائمة)
#             query_rel_set = self.relevant_docs_set.get(query_id, set())
#             total_rel = self.total_relevant_counts.get(query_id, 0)

#             if not retrieved or k <= 0 or not query_rel_set:
#                 p = r = ap = ndcg = 0.0
#             else:
#                 limit = min(len(retrieved), k)
#                 relevant_found = 0
#                 ap_sum = 0.0
#                 dcg = 0.0
#                 idcg = self._get_cached_idcg(query_id, k)

#                 for i in range(limit):
#                     doc = retrieved[i]
#                     rank = i + 1
#                     if doc in query_rel_set:
#                         relevant_found += 1
#                         ap_sum += relevant_found / rank

#                     # جمع معلومات الـ NDCG
#                     rel = self.qrels.get(query_id, {}).get(doc, 0)
#                     if rel > 0:
#                         dcg += (2**rel - 1) / math.log2(rank + 1)

#                 p = relevant_found / k if k > 0 else 0.0
#                 r = relevant_found / total_rel if total_rel > 0 else 0.0
#                 ap = ap_sum / total_rel if total_rel > 0 else 0.0
#                 ndcg = dcg / idcg if idcg > 0 else 0.0

#         return p, r, ap, ndcg

#     def _get_metric(self, retrieved: list[str], query_id: str, k: int, metric_idx: int):
#         """دالة مساعدة لاسترجاع مقياس معين من الكاش"""
#         if not retrieved or k <= 0:
#             return 0.0

#         cache_key = (query_id, k)
#         if cache_key not in self._metric_cache:
#             p, r, ap, ndcg = self._compute_all_metrics(retrieved, query_id, k)
#             self._metric_cache[cache_key] = (p, r, ap, ndcg)

#         return self._metric_cache[cache_key][metric_idx]

#     def calculate_precision_at_k(
#         self, retrieved: list[str], query_id: str, k: int = 10
#     ) -> float:
#         return self._get_metric(retrieved, query_id, k, 0)

#     def calculate_recall(
#         self, retrieved: list[str], query_id: str, k: int = 10
#     ) -> float:
#         return self._get_metric(retrieved, query_id, k, 1)

#     def calculate_ap(self, retrieved: list[str], query_id: str, k: int = 10) -> float:
#         return self._get_metric(retrieved, query_id, k, 2)

#     def calculate_ndcg(self, retrieved: list[str], query_id: str, k: int = 10) -> float:
#         return self._get_metric(retrieved, query_id, k, 3)

#     def calculate_map(self, retrieved_runs: dict[str, list[str]], k: int = 10) -> float:
#         if not retrieved_runs:
#             return 0.0
#         ap_scores = [
#             self.calculate_ap(retrieved, qid, k)
#             for qid, retrieved in retrieved_runs.items()
#         ]
#         return sum(ap_scores) / len(ap_scores)
