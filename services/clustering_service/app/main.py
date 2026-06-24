from fastapi import FastAPI, HTTPException, BackgroundTasks
from .schemas import ClusteringRequest, ClusteringResponse
from .config import settings
import uvicorn
import time
import os
import json
import joblib
import sqlite3
import re
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from .clusterer import DocumentClusterer

app = FastAPI(title="Clustering Service", version="1.0.0")

clustering_jobs = {}
embeddings_cache = {}

def run_clustering_task(dataset: str, num_clusters: int, algorithm: str):
    clustering_jobs[dataset] = "processing"
    try:
        emb_path = os.path.join(settings.DATA_DIR, "models", f"{dataset}_embedding.joblib")
        if not os.path.exists(emb_path):
            raise FileNotFoundError(f"Embeddings model file not found at: {emb_path}")
            
        print(f"[Clustering] Loading embeddings from {emb_path}...")
        data = joblib.load(emb_path)
        document_vectors = data["embeddings"]
        doc_ids = data["doc_ids"]
        
        print(f"[Clustering] Training K-Means on {len(doc_ids)} documents with k={num_clusters}...")
        clusterer = DocumentClusterer(num_clusters=num_clusters)
        labels = clusterer.cluster_vectors(document_vectors)
        
        clusters_dir = os.path.join(settings.DATA_DIR, "clusters")
        os.makedirs(clusters_dir, exist_ok=True)
        
        # Create doc_id -> cluster mapping
        cluster_mapping = {str(doc_ids[i]): int(labels[i]) for i in range(len(doc_ids))}
        
        out_path = os.path.join(clusters_dir, f"{dataset}_kmeans_{num_clusters}.json")
        print(f"[Clustering] Writing cluster results to {out_path}...")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(cluster_mapping, f, indent=4)
            
        clustering_jobs[dataset] = "completed"
        print(f"[Clustering] Completed successfully for {dataset}!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        clustering_jobs[dataset] = f"failed: {str(e)}"

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "clustering_service"}

@app.post("/cluster/build", response_model=ClusteringResponse)
def build_clusters(request: ClusteringRequest, background_tasks: BackgroundTasks):
    dataset = request.dataset
    if dataset in clustering_jobs and clustering_jobs[dataset] == "processing":
        return ClusteringResponse(
            dataset=dataset,
            status="processing",
            clusters_count=0,
            message="Clustering job is already in progress"
        )
        
    background_tasks.add_task(
        run_clustering_task, 
        dataset, 
        request.num_clusters, 
        request.algorithm
    )
    
    return ClusteringResponse(
        dataset=dataset,
        status="started",
        clusters_count=0,
        message=f"Clustering job started asynchronously using {request.algorithm}."
    )

@app.get("/cluster/status/{dataset}")
def get_clustering_status(dataset: str):
    status = clustering_jobs.get(dataset, "idle")
    return {"dataset": dataset, "status": status}

@app.get("/cluster/plot/{dataset}/{num_clusters}")
def get_cluster_plot(dataset: str, num_clusters: int):
    global embeddings_cache
    try:
        emb_path = os.path.join(settings.DATA_DIR, "models", f"{dataset}_embedding.joblib")
        if not os.path.exists(emb_path):
            raise HTTPException(status_code=404, detail=f"Embeddings model file not found at: {emb_path}")
        
        # Load dataset embeddings from cache or joblib
        if dataset in embeddings_cache:
            data = embeddings_cache[dataset]
        else:
            print(f"[Clustering] Loading embeddings for plot cache from {emb_path}...")
            data = joblib.load(emb_path)
            embeddings_cache[dataset] = data
            
        document_vectors = data["embeddings"]
        doc_ids = data["doc_ids"]
        
        # Sample 150 documents (deterministic random using RandomState)
        num_docs = len(doc_ids)
        num_samples = min(150, num_docs)
        
        rng = np.random.RandomState(42)
        sample_indices = rng.choice(num_docs, num_samples, replace=False)
        
        sample_vectors = document_vectors[sample_indices]
        sample_doc_ids = [str(doc_ids[i]) for i in sample_indices]
        
        # Perform KMeans on the 150 sampled document vectors
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(sample_vectors)
        centroids = kmeans.cluster_centers_
        
        # Perform PCA to project vectors to 2D
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(sample_vectors)
        centroids_2d = pca.transform(centroids)
        
        # Connect to SQLite to fetch original raw texts
        conn = sqlite3.connect(settings.DB_PATH)
        cursor = conn.cursor()
        
        placeholders = ",".join(["?"] * len(sample_doc_ids))
        query = f"SELECT doc_id, text FROM documents WHERE dataset_name = ? AND doc_id IN ({placeholders})"
        cursor.execute(query, [dataset] + sample_doc_ids)
        rows = cursor.fetchall()
        conn.close()
        
        doc_text_map = {str(row[0]): row[1] for row in rows}
        
        # Generate points
        points = []
        for i in range(num_samples):
            d_id = sample_doc_ids[i]
            text = doc_text_map.get(d_id, "[Document text not found in database]")
            points.append({
                "x": float(coords_2d[i, 0]),
                "y": float(coords_2d[i, 1]),
                "doc_id": d_id,
                "cluster": int(labels[i]),
                "snippet": text[:120] + ("..." if len(text) > 120 else "")
            })
            
        # Generate centroids
        centroids_list = []
        for c in range(num_clusters):
            centroids_list.append({
                "x": float(centroids_2d[c, 0]),
                "y": float(centroids_2d[c, 1]),
                "cluster": c
            })
            
        # Generate dynamic cluster names (keywords extraction)
        cluster_texts = {c: [] for c in range(num_clusters)}
        for i in range(num_samples):
            d_id = sample_doc_ids[i]
            text = doc_text_map.get(d_id, "")
            if text:
                cluster_texts[labels[i]].append(text)
                
        # Simple stopword list
        stopwords = {
            "the", "a", "and", "is", "in", "it", "to", "of", "for", "on", "with", "as", "at", "by", "an",
            "this", "that", "are", "was", "were", "or", "but", "if", "you", "your", "my", "we", "they",
            "he", "she", "it", "i", "do", "does", "did", "have", "has", "had", "what", "which", "who",
            "whom", "how", "why", "where", "when", "about", "there", "their", "more", "some", "any",
            "one", "would", "like", "can", "out", "up", "so", "no", "just", "what", "which", "will", "from",
            "not", "get", "use"
        }
        
        cluster_names = []
        cluster_word_counts = {}
        global_word_docs = {} # word -> set of cluster IDs
        
        for c in range(num_clusters):
            texts = cluster_texts[c]
            word_counts = {}
            for text in texts:
                words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
                for word in words:
                    if word not in stopwords:
                        word_counts[word] = word_counts.get(word, 0) + 1
                        if word not in global_word_docs:
                            global_word_docs[word] = set()
                        global_word_docs[word].add(c)
            cluster_word_counts[c] = word_counts
            
        for c in range(num_clusters):
            word_counts = cluster_word_counts[c]
            word_scores = []
            for word, freq in word_counts.items():
                num_sharing_clusters = len(global_word_docs.get(word, set()))
                score = freq / (1.0 + num_sharing_clusters)
                word_scores.append((word, score))
                
            word_scores.sort(key=lambda x: x[1], reverse=True)
            top_words = [ws[0].capitalize() for ws in word_scores[:3]]
            
            if top_words:
                cluster_names.append(f"Cluster {c+1} ({', '.join(top_words)})")
            else:
                cluster_names.append(f"Cluster {c+1}")
                
        return {
            "points": points,
            "centroids": centroids_list,
            "cluster_names": cluster_names
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=settings.CLUSTERING_PORT, reload=True)
