import joblib
import os

data_dir = r"c:\Users\USER\Desktop\fifth year\وظائف ومشاريع الجامعة\IR\IR-Project\Moli-Search-Engine\data"

for filename in ["lotte_lifestyle_dev_embedding.joblib", "quora_dev_embedding.joblib"]:
    path = os.path.join(data_dir, "models", filename)
    if os.path.exists(path):
        print(f"Loading {filename}...")
        data = joblib.load(path)
        embeddings = data["embeddings"]
        doc_ids = data["doc_ids"]
        print(f"  Embeddings shape: {embeddings.shape}")
        print(f"  Number of doc IDs: {len(doc_ids)}")
    else:
        print(f"Path not found: {path}")
