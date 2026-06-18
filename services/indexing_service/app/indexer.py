# Indexer Class Orchestrator
from .vsm_tfidf_builder import build_tfidf_representation
from .bert_embeddings_builder import build_bert_representation
from .bm25_builder import build_bm25_representation
from .hybrid_builder import build_hybrid_representation
from .inverted_index_builder import build_inverted_index_representation

class Indexer:
    """
    Indexer orchestrator. delegating TF-IDF, BERT embeddings, BM25, Hybrid,
    and Inverted Index compilation to optimized modular files.
    """
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        
    def build_tfidf(self) -> dict:
        return build_tfidf_representation(self.dataset_name)
        
    def build_embedding(self) -> dict:
        return build_bert_representation(self.dataset_name)
        
    def build_bm25(self, k1: float = 1.5, b: float = 0.75) -> dict:
        return build_bm25_representation(self.dataset_name, k1, b)
        
    def build_hybrid(self) -> dict:
        return build_hybrid_representation(self.dataset_name)
        
    def build_inverted_index(self) -> dict:
        return build_inverted_index_representation(self.dataset_name)
