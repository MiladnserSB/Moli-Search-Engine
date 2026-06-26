<div align="center">
  <h1>🔍 Moli Search Engine</h1>
  <p><strong>A Production-Grade Information Retrieval System built with Service-Oriented Architecture</strong></p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white" />
    <img src="https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=flat&logo=fastapi&logoColor=white" />
    <img src="https://img.shields.io/badge/Architecture-SOA-002456?style=flat" />
    <img src="https://img.shields.io/badge/Datasets-Quora%20%7C%20LoTTE-f89c1e?style=flat" />
  </p>
</div>

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Datasets](#-datasets)
3. [System Architecture](#-system-architecture)
4. [Services Breakdown](#-services-breakdown)
5. [Retrieval Models](#-retrieval-models)
6. [Advanced Features](#-advanced-features)
7. [Evaluation](#-evaluation)
8. [Setup & Installation](#-setup--installation)
9. [Running the System](#-running-the-system)
10. [API Reference](#-api-reference)
11. [Project Structure](#-project-structure)

---

## 📖 Project Overview

**Moli Search Engine** is a full-featured Information Retrieval (IR) system designed and built according to the **Service-Oriented Architecture (SOA)** principle. Each major capability of the system is encapsulated in its own independent microservice, communicating via REST APIs through a central **API Gateway**.

### Key Capabilities
- Multi-model retrieval: **TF-IDF (VSM)**, **BM25**, **Dense Embeddings (BERT)**, **Hybrid (Serial & Parallel)**
- Advanced **Query Refinement**: spell correction, synonym expansion, personalized history-based re-ranking
- **Document Clustering** (K-Means) with PCA visualization
- **Offline & Online Evaluation** with Before/After advanced-feature comparison
- Full **Evaluation Dashboard** with charts (MAP, Recall, P@10, nDCG)
- Rich **Web UI** with live search, autocomplete, and real-time refinement banners

---

## 📊 Datasets

The system operates on **two IR benchmark datasets**, both sourced from [ir-datasets.com](https://ir-datasets.com):

| Dataset | Queries | Documents | Description |
|---------|---------|-----------|-------------|
| **Quora Dev** (`quora_dev`) | 5,000 | 530,809 | Duplicate question pairs — question-answering IR |
| **LoTTE Lifestyle Dev** (`lotte_lifestyle_dev`) | 2,076 | 269,814 | Forum-style lifestyle Q&A — passage retrieval |

Both datasets include **qrels** (relevance judgement files), enabling standard IR evaluation.

> ⚠️ The system stores all document raw texts in a **SQLite database** (`data/ir_dataset_store.db`), retrieved at query time by document ID — as required by the project supervisor.

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MOLI SEARCH ENGINE                           │
│                    Service-Oriented Architecture                     │
└─────────────────────────────────────────────────────────────────────┘

  User Browser
      │
      ▼
┌─────────────────┐
│ Frontend Service│  (Port 8001) — HTML/CSS/JS Web UI
│   (FastAPI +    │
│   Jinja2 +      │
│   Chart.js)     │
└────────┬────────┘
         │ HTTP REST
         ▼
┌─────────────────┐
│  API Gateway    │  (Port 8000) — Single entry point, routing, CORS
│  Service        │
└────────┬────────┘
         │
    ┌────┴──────────────────────────────────────────────┐
    │              Internal Service Mesh                 │
    │                                                    │
    ▼              ▼              ▼              ▼       │
┌────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│Retriev-│  │Preproc.  │  │Query     │  │Clustering│  │
│al Svc  │  │Service   │  │Refinement│  │Service   │  │
│(8002)  │  │(8003)    │  │Svc(8006) │  │(8004)    │  │
└──┬─────┘  └──────────┘  └──────────┘  └──────────┘  │
   │                                                    │
   ▼              ▼                                     │
┌────────┐  ┌──────────┐                               │
│Indexing│  │Evaluation│                               │
│Service │  │Service   │                               │
│(8007)  │  │(8005)    │                               │
└────────┘  └──────────┘                               │
                    └──────────────────────────────────┘

Shared Storage:
  ├── data/ir_dataset_store.db    ← SQLite: raw texts, queries, qrels
  ├── data/models/                ← Trained model artifacts (joblib)
  ├── data/indices/               ← Prebuilt search indices
  ├── data/clusters/              ← K-Means cluster assignments
  └── data/evaluation_results.json ← Pre-computed eval metrics cache
```

---

## 🔧 Services Breakdown

### 1. 🌐 Frontend Service (Port 8001)
**Location**: `services/frontend_service/`

The web-based user interface. Built with FastAPI serving static HTML/CSS/JS files.

**Features**:
- Search bar with live autocomplete
- Sidebar with model selection, BM25 parameter sliders, hybrid alpha control
- Query refinement banner (spell correction, synonyms, personalized suggestions)
- Clustering visualization tab (PCA scatter + cluster cards)
- Evaluation dashboard with Before/After comparison charts (Chart.js)
- Toggle between **Offline (instant)** and **Online (live)** evaluation modes

---

### 2. 🚪 API Gateway Service (Port 8000)
**Location**: `services/gateway_service/`

Central routing hub. All frontend requests go through this gateway.

**Endpoints**:
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/search` | Single-query search |
| POST | `/api/search/batch` | Batch search (evaluation) |
| POST | `/api/evaluate` | Trigger evaluation |
| POST | `/api/cluster` | Trigger clustering |
| GET  | `/api/cluster/plot/{dataset}/{n}` | Get cluster plot data |
| POST | `/api/autocomplete` | Get autocomplete suggestions |
| POST | `/api/refine` | Get query refinements |

---

### 3. 🔍 Retrieval Service (Port 8002)
**Location**: `services/retrieval_service/`

Handles all search operations. Loads pre-trained model artifacts at startup.

**Supported Models**:
- `vsm` — TF-IDF cosine similarity
- `bm25` — Probabilistic BM25 (with tunable k₁, b parameters)
- `embedding` — Dense BERT embeddings (all-MiniLM-L6-v2)
- `hybrid_serial` — BM25 candidates → Dense re-ranking
- `hybrid_parallel` — RRF fusion of BM25 + Dense results

**Key Design**: Uses **pre-built indices** stored as `.joblib` files under `data/models/`. No training at query time — all models are loaded at startup.

---

### 4. ⚙️ Preprocessing Service (Port 8003)
**Location**: `services/preprocessing_service/`

Text normalization pipeline with two modes:
- **Classical** (for VSM/BM25): tokenization, stopword removal, stemming, lemmatization
- **Neural** (for Embeddings): light normalization only (preserve semantic content)

**Endpoints**: `/preprocess` (single), `/preprocess/batch` (bulk)

---

### 5. 🗂️ Indexing Service (Port 8007)
**Location**: `services/indexing_service/`

Manages the creation and persistence of search indices:
- Inverted index for BM25/VSM
- FAISS-style vector index for embeddings
- Exposes `/index/build` endpoint

---

### 6. 🔄 Query Refinement Service (Port 8006)
**Location**: `services/query_refinement_service/`

The **Advanced Feature** service. Improves query quality before retrieval.

**Capabilities**:
- **Spell Correction** — SymSpell-based edit-distance correction
- **Synonym Expansion** — WordNet-based query broadening
- **Personalization** — History-based query enrichment using past user queries
- **Autocomplete** — N-gram trie-based suggestions from the dataset vocabulary
- **Query Logging** — Stores user query history for personalization

---

### 7. 📊 Clustering Service (Port 8004)
**Location**: `services/clustering_service/`

Document clustering using **K-Means** on TF-IDF representations.

**Features**:
- Configurable number of clusters (2–50)
- **PCA** dimensionality reduction for 2D visualization
- Cluster quality metrics (inertia, silhouette)
- Returns cluster assignments + PCA coordinates for frontend visualization

---

### 8. 📈 Evaluation Service (Port 8005)
**Location**: `services/evaluation_service/`

Measures retrieval quality using standard IR metrics.

**Metrics**: MAP, Recall, Precision@10, nDCG@10

**Modes**:
| Mode | Description | Speed |
|------|-------------|-------|
| **Offline** | Returns pre-computed results from `evaluation_results.json` | Instant |
| **Online** | Runs live retrieval for N queries (default: 500) | 2–10 min |

**Before/After Comparison**:
- **BASE**: Standard retrieval without query refinement
- **ENHANCED**: Same model with spell correction + synonym expansion applied

---

## 🤖 Retrieval Models

### TF-IDF (Vector Space Model)
Uses scikit-learn's `TfidfVectorizer` with cosine similarity scoring.
```
score(q, d) = cosine(TF-IDF(q), TF-IDF(d))
```

### BM25 (Probabilistic Model)
Uses `rank-bm25` library with tunable parameters:
- **k₁** (term saturation): default 1.5, range [0.5, 3.0]
- **b** (length normalization): default 0.75, range [0.0, 1.0]

```
BM25(q,d) = Σ IDF(t) × [ tf(t,d)×(k₁+1) / (tf(t,d) + k₁×(1-b+b×dl/avgdl)) ]
```

### Dense Embeddings (BERT)
Uses `sentence-transformers` with `all-MiniLM-L6-v2` (384-dim vectors).
```
score(q, d) = cosine(BERT(q), BERT(d))
```

### Hybrid Serial
BM25 retrieves top-100 candidates → Dense model re-ranks them.
```
1. BM25 → top-100 docs
2. Dense re-rank → top-10 final
```

### Hybrid Parallel (RRF Fusion)
BM25 and Dense run independently → scores fused via Reciprocal Rank Fusion:
```
RRF(d) = α × (1/(k + rank_BM25(d))) + (1-α) × (1/(k + rank_Dense(d)))
```
where α controls BM25 vs Dense balance (configurable from UI).

---

## ✨ Advanced Features

### 1. Document Clustering
- **Algorithm**: K-Means on TF-IDF document vectors
- **Visualization**: PCA 2D scatter plot rendered with Chart.js
- **Integration**: Can be toggled independently; clustered docs shown in a separate tab
- **Testability**: Works independently via the Clustering tab in the UI

### 2. Query Refinement (Additional Feature)
- Spell correction, synonym expansion, personalized re-weighting
- Applied as pre-processing step before retrieval
- **Testability**: Toggle "Enable Advanced Features" in the sidebar

### 3. Before/After Evaluation
- Evaluation with `use_additional_features=False` → BASE metrics
- Evaluation with `use_additional_features=True` → ENHANCED metrics (spell correction applied)
- Both shown side-by-side with % improvement badges and a grouped bar chart

---

## 📏 Evaluation

### Metrics Used
| Metric | Formula | Description |
|--------|---------|-------------|
| **MAP** | Mean of AP scores | Average precision across all queries |
| **Recall** | Rel. retrieved / Total rel. | Fraction of relevant docs found |
| **P@10** | Rel. in top-10 / 10 | Precision in top 10 results |
| **nDCG@10** | DCG / IDCG | Ranking quality with graded relevance |

### Results (Full Dataset Offline)

#### Quora Dev (5,000 queries)

| Model | MAP | Recall | P@10 | nDCG |
|-------|-----|--------|------|------|
| BM25 | 0.6782 | 0.8310 | 0.1116 | 0.7272 |
| VSM (TF-IDF) | 0.6479 | 0.8029 | 0.1075 | 0.6974 |
| **Embedding (BERT)** | **0.8353** | **0.9480** | **0.1323** | **0.8732** |
| Hybrid Serial | 0.8104 | 0.9106 | 0.1258 | 0.8455 |
| Hybrid Parallel | 0.7798 | 0.9224 | 0.1267 | 0.8264 |

#### LoTTE Lifestyle Dev (2,076 queries)

| Model | MAP | Recall | P@10 | nDCG |
|-------|-----|--------|------|------|
| BM25 | 0.1318 | 0.2137 | 0.1153 | 0.2130 |
| VSM (TF-IDF) | 0.0828 | 0.1485 | 0.0787 | 0.1421 |
| **Embedding (BERT)** | **0.2479** | **0.3676** | **0.2022** | **0.3690** |
| Hybrid Serial | 0.2243 | 0.3230 | 0.1775 | 0.3358 |
| Hybrid Parallel | 0.2124 | 0.3297 | 0.1791 | 0.3264 |

> **Note**: Dense Embeddings outperform lexical models on both datasets, demonstrating the value of semantic representations for IR.

---

## 🛠 Setup & Installation

### Prerequisites
- **Python 3.10+**
- **pip** or **conda**
- At least **8 GB RAM** (16 GB recommended for neural models)
- **~10 GB disk space** (models + database)

### 1. Clone the Repository
```bash
git clone https://github.com/MiladnserSB/Moli-Search-Engine.git
cd Moli-Search-Engine
```

### 2. Create a Virtual Environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download and Prepare Data
The system requires pre-trained model files and the SQLite database. These are **NOT included in the repository** due to size constraints.

Run the data loader script to download and prepare the datasets:
```bash
python scripts/data_loader.py
```

> ⚠️ First-time setup will download the Quora and LoTTE datasets and build the SQLite database. This takes approximately **30–60 minutes**.

### 5. Build Indices and Train Models
```bash
# Train all retrieval models (TF-IDF, BM25, Embeddings)
# This step is required only once
python scripts/recompute_real_evaluation.py --method coordinator
```

---

## 🚀 Running the System

### Start All Services (Recommended)
```bash
python scripts/run_all_services.py
```

This launches all 8 services concurrently. Wait for all services to report `Application startup complete.`

### Start Individual Services
```bash
# Windows PowerShell — set PYTHONPATH first
$env:PYTHONPATH = "services/gateway_service"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# Repeat for each service with its port
```

### Service Ports Reference
| Service | Port | URL |
|---------|------|-----|
| API Gateway | 8000 | http://127.0.0.1:8000 |
| Frontend | 8001 | http://127.0.0.1:8001 |
| Retrieval | 8002 | http://127.0.0.1:8002 |
| Preprocessing | 8003 | http://127.0.0.1:8003 |
| Clustering | 8004 | http://127.0.0.1:8004 |
| Evaluation | 8005 | http://127.0.0.1:8005 |
| Query Refinement | 8006 | http://127.0.0.1:8006 |
| Indexing | 8007 | http://127.0.0.1:8007 |

### Access the UI
Open your browser at: **http://127.0.0.1:8001**

---

## 📡 API Reference

### Search
```http
POST http://127.0.0.1:8000/api/search
Content-Type: application/json

{
  "query": "What is machine learning?",
  "dataset": "quora_dev",
  "method": "bm25",
  "use_additional_features": false,
  "bm25_k1": 1.5,
  "bm25_b": 0.75,
  "top_k": 10
}
```

### Evaluate
```http
POST http://127.0.0.1:8000/api/evaluate
Content-Type: application/json

{
  "dataset": "quora_dev",
  "method": "bm25",
  "use_additional_features": false,
  "mode": "offline",
  "online_limit": 500
}
```

**Response**:
```json
{
  "dataset": "quora_dev",
  "map_score": 0.6782,
  "recall_score": 0.8310,
  "precision_at_k": 0.1116,
  "ndcg_score": 0.7272,
  "num_queries_evaluated": 5000,
  "mode": "Base (No Additional Features) -- Offline Full Dataset"
}
```

### Cluster
```http
POST http://127.0.0.1:8000/api/cluster
Content-Type: application/json

{
  "dataset": "quora_dev",
  "num_clusters": 10,
  "algorithm": "kmeans"
}
```

---

## 📁 Project Structure

```
Moli-Search-Engine/
│
├── data/
│   ├── ir_dataset_store.db          # SQLite: raw texts, queries, qrels
│   ├── evaluation_results.json      # Pre-computed eval metrics (base + enhanced)
│   ├── models/                      # Trained model artifacts (.joblib)
│   │   ├── quora_dev_tfidf.joblib
│   │   ├── quora_dev_bm25.joblib
│   │   ├── quora_dev_embedding.joblib
│   │   └── lotte_lifestyle_dev_*.joblib
│   ├── indices/                     # Search indices
│   └── clusters/                    # K-Means cluster data
│
├── services/
│   ├── gateway_service/             # API Gateway (port 8000)
│   │   └── app/
│   │       ├── main.py              # Route definitions
│   │       ├── schemas.py           # Pydantic request/response models
│   │       └── config.py            # Service URLs configuration
│   │
│   ├── frontend_service/            # Web UI (port 8001)
│   │   └── app/
│   │       ├── main.py              # FastAPI static file server
│   │       ├── templates/index.html # Main HTML interface
│   │       └── static/
│   │           ├── css/styles.css   # Premium design system
│   │           └── js/app.js        # Frontend controller logic
│   │
│   ├── retrieval_service/           # Search models (port 8002)
│   │   └── app/
│   │       ├── main.py              # Search endpoints
│   │       ├── retriever.py         # Model loading & scoring
│   │       └── fusion.py            # RRF & serial hybrid logic
│   │
│   ├── preprocessing_service/       # Text pipeline (port 8003)
│   │   └── app/
│   │       ├── main.py              # Preprocess endpoints
│   │       └── preprocessor.py      # Classical & neural pipelines
│   │
│   ├── clustering_service/          # K-Means clustering (port 8004)
│   │   └── app/
│   │       └── main.py              # Cluster build & plot endpoints
│   │
│   ├── evaluation_service/          # IR evaluation (port 8005)
│   │   └── app/
│   │       ├── main.py              # Offline/online eval endpoints
│   │       ├── evaluator.py         # MAP, Recall, P@10, nDCG logic
│   │       ├── schemas.py           # Request/response schemas
│   │       └── config.py            # Service config
│   │
│   ├── query_refinement_service/    # Query enhancement (port 8006)
│   │   └── app/
│   │       └── main.py              # Refine, autocomplete, log, personalize
│   │
│   └── indexing_service/            # Index builder (port 8007)
│       └── app/
│           └── main.py              # Index build endpoints
│
├── scripts/
│   ├── run_all_services.py          # Launch all 8 services
│   ├── data_loader.py               # Download & prepare datasets
│   ├── database.py                  # SQLite schema initialization
│   ├── recompute_real_evaluation.py # Offline evaluation recomputation
│   └── run_evaluation.py            # Standalone evaluation runner
│
├── qrels_files/                     # Official qrels files from ir-datasets
│   ├── quora_dev_qrels.tsv
│   └── lotte_lifestyle_dev_qrels.tsv
│
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── Notes from supervisor.txt        # Supervisor comments & requirements
```

---

## 🔬 Recomputing Evaluation Metrics

To recompute the offline evaluation from scratch (after retraining models):

```bash
# Full evaluation on all models and datasets (takes ~2-4 hours)
python scripts/recompute_real_evaluation.py

# Evaluate a specific model only
python scripts/recompute_real_evaluation.py --method bm25
python scripts/recompute_real_evaluation.py --method vsm
python scripts/recompute_real_evaluation.py --method embedding
python scripts/recompute_real_evaluation.py --method hybrid
```

Results are saved to `data/evaluation_results.json` and are immediately available via the Evaluation Service.

---

## 🧪 Testing Individual Services

Each service can be tested independently:

```bash
# Health check any service
curl http://127.0.0.1:8005/health

# Test evaluation service directly
curl -X POST http://127.0.0.1:8005/evaluate \
  -H "Content-Type: application/json" \
  -d '{"dataset":"quora_dev","method":"bm25","mode":"offline"}'

# Test retrieval service directly
curl -X POST http://127.0.0.1:8002/search \
  -H "Content-Type: application/json" \
  -d '{"query":"machine learning","dataset":"quora_dev","method":"bm25"}'
```

---

## 👥 Team

This project was developed as part of the **Information Retrieval Systems (2026)** course.

---

## 📄 License

This project is for academic purposes. All datasets are credited to their respective sources on [ir-datasets.com](https://ir-datasets.com).
