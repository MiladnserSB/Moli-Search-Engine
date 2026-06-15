# Document Clustering: K-Means, Agglomerative Clustering
from sklearn.cluster import KMeans
import numpy as np

class DocumentClusterer:
    def __init__(self, num_clusters: int = 10):
        self.num_clusters = num_clusters
        self.model = None
        self.centroids = None

    def cluster_vectors(self, document_vectors: np.ndarray) -> np.ndarray:
        # Cluster document vectors using K-Means or Hierarchical Clustering
        # Returns an array of labels for each document
        self.model = KMeans(n_clusters=self.num_clusters, random_state=42)
        labels = self.model.fit_predict(document_vectors)
        self.centroids = self.model.cluster_centers_
        return labels

    def predict_cluster(self, vector: np.ndarray) -> int:
        # Predict which cluster a new query/document vector belongs to
        if self.model is None:
            raise ValueError("Clustering model is not trained yet.")
        return int(self.model.predict(vector.reshape(1, -1))[0])
