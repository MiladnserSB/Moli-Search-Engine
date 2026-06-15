# Indexer Class Skeleton for Inverted Indexing & Vector Database storage

class Indexer:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.inverted_index = {}
        self.doc_lengths = {}

    def build_inverted_index(self, preprocessed_documents: list[dict]):
        # Build standard inverted index: terms -> list of (doc_id, term_frequency)
        pass

    def build_vector_index(self, doc_embeddings):
        # Build local FAISS index or push embeddings to Qdrant
        pass

    def save_index(self, output_path: str):
        # Save indices to disk in settings.INDICES_DIR
        pass

    def load_index(self, input_path: str):
        # Load indices from disk
        pass
