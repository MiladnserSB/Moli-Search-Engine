# Project-Scoped Rules & Constraints

## UI Results Display
- **Original Content Only**: The search results displayed in the UI must show the **original raw text** of the documents (not the preprocessed or cleaned text).
- **Document IDs**: The UI must explicitly print/render the unique document ID (`doc_id` / `id`) next to or inside each document result card (e.g., as `ID: doc_123`).
- **Database Fetch**: Raw texts must be fetched at query time from the SQLite database `ir_dataset_store.db` using the document IDs.

## Evaluation & Qrels
- **Relevance Judgements**: Evaluation scripts or notebooks must run using the official, full `qrels` file.
- **Verification Ready**: Ensure that the `qrels` file structure is easy to verify and check during the evaluation interface.
