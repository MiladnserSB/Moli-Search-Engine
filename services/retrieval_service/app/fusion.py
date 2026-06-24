# Result Fusion Algorithms: Reciprocal Rank Fusion (RRF), CombMNZ

def reciprocal_rank_fusion(runs: list[list[dict]], weights: list[float] = None, k: int = 60) -> list[dict]:
    # Runs is a list of results, each list of dict contains {doc_id, ...} sorted by score
    # Score(d) = Sum_{run in runs} w * 1 / (k + rank(run, d))
    if weights is None:
        weights = [1.0] * len(runs)
        
    rrf_scores = {}
    for run_idx, run in enumerate(runs):
        w = weights[run_idx]
        for rank, doc in enumerate(run, start=1):
            doc_id = doc["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + w * (1.0 / (k + rank))
            
    # Format and sort merged results
    merged_results = [{"id": doc_id, "score": score} for doc_id, score in rrf_scores.items()]
    return sorted(merged_results, key=lambda x: x["score"], reverse=True)

def comb_mnz(runs: list[list[dict]]) -> list[dict]:
    # CombMNZ = sum of normalized scores * count of runs retrieving the document
    return []
