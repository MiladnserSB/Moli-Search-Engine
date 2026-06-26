import ir_datasets
import logging
from typing import List, Dict
from scripts.database import get_db_connection

logger = logging.getLogger(__name__)

class IRDatasetDownloader:
    """Automated extraction architecture designed to stream raw IR datasets securely into persistent tables."""
    def __init__(self, datasets_config: List[Dict[str, str]]):
        self.datasets = datasets_config
    def is_already_downloaded(self, cursor, db_label: str) -> bool:
        """Validates if target repository cluster already exists within internal storage blocks."""
        cursor.execute(
            "SELECT COUNT(*) FROM documents WHERE dataset_name = ?", (db_label,)
        )
        return cursor.fetchone()[0] > 0
    def save_documents(self, cursor, db_label: str, dataset) -> int:
        """Extracts text documents iteratively, inserting chunks into storage blocks to protect RAM."""
        doc_count = 0
        buffer = []
        batch_size = 50000
        print(ll := f"   -> Extracting raw Documents (Batch Size: {batch_size})...")
        for doc in dataset.docs_iter():
            doc_id = str(doc.doc_id)
            # Robust property validation fallback structure
            text = getattr(doc, "text", "") or ""
            title = getattr(doc, "title", "") or ""
            full_text = f"{title}. {text}" if title and text else (title or text)

            if full_text:
                buffer.append((db_label, doc_id, full_text))
                doc_count += 1

            if len(buffer) >= batch_size:
                cursor.executemany(
                    "INSERT INTO documents (dataset_name, doc_id, text) VALUES (?, ?, ?)",
                    buffer,
                )
                print(f"      * Checkpoint: Saved {doc_count} total documents into database...")
                buffer = []
        if buffer:
            cursor.executemany(
                "INSERT INTO documents (dataset_name, doc_id, text) VALUES (?, ?, ?)",
                buffer,
            )
        print(f"   ✅ Finished documents. Total saved: {doc_count}")
        return doc_count

    def save_queries(self, cursor, db_label: str, dataset) -> int:
        """Parses evaluative test queries mapped to current active repository profile."""
        query_count = 0
        buffer = []
        batch_size = 10000
        if dataset.has_queries():
            print(f"   -> Extracting Evaluation Queries (Batch Size: {batch_size})...")
            for query in dataset.queries_iter():
                query_id = str(query.query_id)
                text = getattr(query, "text", "")
                if not text:
                    text = getattr(query, "query", "")
                if text:
                    buffer.append((db_label, query_id, text))
                    query_count += 1

                if len(buffer) >= batch_size:
                    cursor.executemany(
                        "INSERT INTO queries (dataset_name, query_id, text) VALUES (?, ?, ?)",
                        buffer,
                    )
                    print(f"      * Checkpoint: Saved {query_count} total queries into database...")
                    buffer = []
            if buffer:
                cursor.executemany(
                    "INSERT INTO queries (dataset_name, query_id, text) VALUES (?, ?, ?)",
                    buffer,
                )
            print(f"   ✅ Finished queries. Total saved: {query_count}")
        else:
            print("   ⚠️ Notice: This specific dataset variant does not contain explicit queries configuration.")
        return query_count

    def save_qrels(self, cursor, db_label: str, dataset) -> int:
        """Extracts human-annotated Ground-Truth evaluation matrix (Qrels)."""
        qrel_count = 0
        buffer = []
        batch_size = 50000
        if dataset.has_qrels():
            print(f"   -> Extracting Ground-Truth Judgments Matrix (Qrels Batch Size: {batch_size})...")
            for qrel in dataset.qrels_iter():
                buffer.append((db_label, str(qrel.query_id), str(qrel.doc_id), int(qrel.relevance)))
                qrel_count += 1
                if len(buffer) >= batch_size:
                    cursor.executemany(
                        "INSERT INTO qrels (dataset_name, query_id, doc_id, relevance) VALUES (?, ?, ?, ?)",
                        buffer,
                    )
                    print(f"      * Checkpoint: Saved {qrel_count} total Qrel records into database...")
                    buffer = []
            if buffer:
                cursor.executemany(
                    "INSERT INTO qrels (dataset_name, query_id, doc_id, relevance) VALUES (?, ?, ?, ?)",
                    buffer,
                )
            print(f"   ✅ Finished Qrels matrix. Total saved: {qrel_count}")
        else:
            print("   ⚠️ Notice: This specific dataset variant does not contain an evaluation Qrels matrix.")
        return qrel_count

    def execute_pipeline(self):
        """Orchestrates comprehensive step-by-step extraction workflow for configured targets."""
        conn = get_db_connection()
        cursor = conn.cursor()

        for ds_info in self.datasets:
            ds_name = ds_info["name"]
            ds_label = ds_info["db_label"]

            print(f"\n🌐 Analyzing Target Profile: [{ds_label.upper()}] Reference: {ds_name}")

            if self.is_already_downloaded(cursor, ds_label):
                print(f"ℹ️ Storage Alert: Profile [{ds_label}] is already populated in local DB. Skipping download.")
                continue

            print(f"📥 Connecting to remote cluster streaming server: {ds_name}...")
            try:
                dataset = ir_datasets.load(ds_name)
            except Exception as e:
                print(f"❌ Connection Error: Failed to mount downstream repo reference: {ds_name}. Details: {e}")
                continue

            # Stream & Commit Pipeline Subsections Sequentially
            total_docs = self.save_documents(cursor, ds_label, dataset)
            conn.commit()

            total_queries = self.save_queries(cursor, ds_label, dataset)
            conn.commit()

            total_qrels = self.save_qrels(cursor, ds_label, dataset)
            conn.commit()

            print(f"✨ [SUCCESS ENTRY] Target [{ds_label}] fully integrated to DB storage.")
            print(f"   Summary Statistics -> [Docs: {total_docs}] | [Queries: {total_queries}] | [Qrels: {total_qrels}]")
            print("-" * 60)

        conn.close()