import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFSearcher:
    def __init__(self, model_path):
        data = joblib.load(model_path)

        self.vectorizer = data["vectorizer"]
        self.doc_vectors = data["vectors"]
        self.doc_ids = data["doc_ids"]
        self.doc_id_to_index = {str(doc_id): i for i, doc_id in enumerate(self.doc_ids)}

    def search(self, query_text, top_k=100, inverted_index=None):
        q = self.vectorizer.transform([query_text])
        scores = self.doc_vectors.dot(q.T).toarray().flatten()

        # Optimize search using the inverted index to filter candidate documents
        if inverted_index:
            query_tokens = query_text.split()
            candidate_ids = set()
            corpus_size = len(self.doc_ids)
            # Ignore terms appearing in more than 5% of corpus or 15,000 docs to avoid huge candidate sets
            max_df = min(15000, int(corpus_size * 0.05))
            
            for token in query_tokens:
                if token in inverted_index:
                    postings = inverted_index[token]
                    if len(postings) <= max_df:
                        candidate_ids.update(postings)
            
            if candidate_ids:
                candidate_indices = [self.doc_id_to_index[str(doc_id)] for doc_id in candidate_ids if str(doc_id) in self.doc_id_to_index]
                if candidate_indices:
                    candidate_indices = np.array(candidate_indices, dtype=np.int32)
                    candidate_scores = scores[candidate_indices]
                    idx = np.argsort(candidate_scores)[::-1][:top_k]
                    return [
                        {
                            "id": self.doc_ids[candidate_indices[i]],
                            "score": float(candidate_scores[i])
                        }
                        for i in idx
                    ]

        # Fallback to full collection search if inverted_index is not active or empty
        idx = np.argpartition(scores, -top_k)[-top_k:]
        idx = idx[np.argsort(scores[idx])[::-1]]
        return [
            {
                "id": self.doc_ids[i],
                "score": float(scores[i])
            }
            for i in idx
        ]

    def search_batch(self, queries, top_k=100, inverted_index=None):
        if not queries:
            return []
        # Transform all queries at once
        q = self.vectorizer.transform(queries)
        # Compute dot product
        scores = self.doc_vectors.dot(q.T).toarray()  # (num_docs, num_queries)
        
        batch_results = []
        for q_idx in range(len(queries)):
            q_scores = scores[:, q_idx]
            
            candidate_indices = None
            if inverted_index:
                query_tokens = queries[q_idx].split()
                candidate_ids = set()
                corpus_size = len(self.doc_ids)
                max_df = min(15000, int(corpus_size * 0.05))
                for token in query_tokens:
                    if token in inverted_index:
                        postings = inverted_index[token]
                        if len(postings) <= max_df:
                            candidate_ids.update(postings)
                if candidate_ids:
                    candidate_indices = [self.doc_id_to_index[str(doc_id)] for doc_id in candidate_ids if str(doc_id) in self.doc_id_to_index]
            
            if candidate_indices:
                candidate_indices = np.array(candidate_indices, dtype=np.int32)
                candidate_scores = q_scores[candidate_indices]
                idx = np.argsort(candidate_scores)[::-1][:top_k]
                results = [
                    {
                        "id": self.doc_ids[candidate_indices[i]],
                        "score": float(candidate_scores[i])
                    }
                    for i in idx
                ]
            else:
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


class BM25Searcher:
    def __init__(self, model_path):
        data = joblib.load(model_path)

        self.bm25 = data["bm25_vectorizer"]
        self.doc_ids = data["doc_ids"]
        self.doc_id_to_index = {str(doc_id): i for i, doc_id in enumerate(self.doc_ids)}
        self.mapped_postings_cache = {}
        self.tf_matrix = None
        self.vocab = None

    def _ensure_matrix(self):
        if self.tf_matrix is not None:
            return
        print("  [BM25Searcher] Building vocabulary mapping and CSC matrix for batch search...")
        self.vocab = {token: i for i, token in enumerate(self.bm25.idf.keys())}
        indptr = [0]
        indices = []
        data_values = []
        for doc in self.bm25.doc_freqs:
            for token, freq in doc.items():
                col_idx = self.vocab.get(token)
                if col_idx is not None:
                    indices.append(col_idx)
                    data_values.append(freq)
            indptr.append(len(indices))
        csr = sp.csr_matrix((data_values, indices, indptr), dtype=np.float32, shape=(len(self.doc_ids), len(self.vocab)))
        self.tf_matrix = csr.tocsc()

    def search(self, query_tokens, top_k=100, k1=None, b=None, inverted_index=None):
        if k1 is not None:
            self.bm25.k1 = k1
        if b is not None:
            self.bm25.b = b
            
        corpus_size = len(self.doc_ids)
        # Call the official rank_bm25 library method to compute scores
        scores = np.array(self.bm25.get_scores(query_tokens))
        
        candidate_indices = None
        if inverted_index:
            candidate_ids = set()
            max_df = min(15000, int(corpus_size * 0.05))
            for token in query_tokens:
                if token in inverted_index:
                    postings = inverted_index[token]
                    if len(postings) <= max_df:
                        candidate_ids.update(postings)
            if candidate_ids:
                candidate_indices = []
                for doc_id in candidate_ids:
                    idx = self.doc_id_to_index.get(str(doc_id))
                    if idx is not None:
                        candidate_indices.append(idx)
                if candidate_indices:
                    mask = np.ones(corpus_size, dtype=bool)
                    mask[candidate_indices] = False
                    scores[mask] = -1e9
            
        if candidate_indices:
            candidate_indices = np.array(candidate_indices, dtype=np.int32)
            candidate_scores = scores[candidate_indices]
            sort_idx = np.argsort(candidate_scores)[::-1][:top_k]
            return [
                {
                    "id": self.doc_ids[candidate_indices[i]],
                    "score": float(candidate_scores[i])
                }
                for i in sort_idx
            ]
        else:
            sort_idx = np.argpartition(scores, -top_k)[-top_k:]
            sort_idx = sort_idx[np.argsort(scores[sort_idx])[::-1]]
            return [
                {
                    "id": self.doc_ids[i],
                    "score": float(scores[i])
                }
                for i in sort_idx
            ]

    def search_batch(self, queries_tokens, top_k=100, k1=None, b=None, inverted_index=None):
        if not queries_tokens:
            return []
        self._ensure_matrix()
        if k1 is not None:
            self.bm25.k1 = k1
        if b is not None:
            self.bm25.b = b
            
        corpus_size = len(self.doc_ids)
        k1 = self.bm25.k1
        b = self.bm25.b
        avgdl = self.bm25.avgdl
        doc_len = np.array(self.bm25.doc_len)
        
        batch_results = []
        
        for tokens in queries_tokens:
            scores = np.zeros(corpus_size)
            candidate_indices = None
            candidate_mask = None
            
            if inverted_index:
                candidate_ids = set()
                max_df = min(15000, int(corpus_size * 0.05))
                for token in tokens:
                    if token in inverted_index:
                        postings = inverted_index[token]
                        if len(postings) <= max_df:
                            candidate_ids.update(postings)
                if candidate_ids:
                    candidate_indices = []
                    for doc_id in candidate_ids:
                        idx = self.doc_id_to_index.get(str(doc_id))
                        if idx is not None:
                            candidate_indices.append(idx)
                    if candidate_indices:
                        candidate_mask = np.zeros(corpus_size, dtype=bool)
                        candidate_mask[candidate_indices] = True
                        
            for q in tokens:
                idf = self.bm25.idf.get(q)
                if not idf:
                    continue
                    
                col_idx = self.vocab.get(q)
                if col_idx is None:
                    continue
                    
                col = self.tf_matrix[:, col_idx]
                matching_indices = col.indices
                q_freq = col.data
                
                if len(matching_indices) == 0:
                    continue
                    
                if candidate_mask is not None:
                    in_candidates = candidate_mask[matching_indices]
                    matching_indices = matching_indices[in_candidates]
                    q_freq = q_freq[in_candidates]
                    if len(matching_indices) == 0:
                        continue
                        
                lengths = doc_len[matching_indices]
                denom = q_freq + k1 * (1.0 - b + b * lengths / avgdl)
                contrib = idf * (q_freq * (k1 + 1.0) / denom)
                scores[matching_indices] += contrib
                
            if candidate_indices is not None:
                candidate_indices = np.array(candidate_indices, dtype=np.int32)
                candidate_scores = scores[candidate_indices]
                sort_idx = np.argsort(candidate_scores)[::-1][:top_k]
                results = [
                    {
                        "id": self.doc_ids[candidate_indices[i]],
                        "score": float(candidate_scores[i])
                    }
                    for i in sort_idx
                ]
            else:
                sort_idx = np.argpartition(scores, -top_k)[-top_k:]
                sort_idx = sort_idx[np.argsort(scores[sort_idx])[::-1]]
                results = [
                    {
                        "id": self.doc_ids[i],
                        "score": float(scores[i])
                    }
                    for i in sort_idx
                ]
            batch_results.append(results)
            
        return batch_results