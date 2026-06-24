# import os
# import sys
# import sqlite3
# import nltk

# # Reconfigure stdout and stderr to prevent UnicodeEncodeError on Windows
# if sys.stdout.encoding != 'utf-8':
#     try:
#         sys.stdout.reconfigure(encoding='utf-8', errors='replace')
#     except Exception:
#         pass
# if sys.stderr.encoding != 'utf-8':
#     try:
#         sys.stderr.reconfigure(encoding='utf-8', errors='replace')
#     except Exception:
#         pass

# # تحميل حزم الـ NLTK الأساسية لمرة واحدة عند الاستيراد للتأكد من جاهزيتها قبل إنشاء الكائنات
# nltk.download('punkt', quiet=True)
# nltk.download('stopwords', quiet=True)
# nltk.download('wordnet', quiet=True)
# nltk.download('averaged_perceptron_tagger', quiet=True)
# nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# from fastapi import FastAPI, BackgroundTasks, HTTPException
# from pydantic import BaseModel
# from typing import Optional
# from contextlib import asynccontextmanager
# from preprocessor import Preprocessor

# # كاشف آمن لمكان ملف قاعدة البيانات (يعمل في دكر ومحلياً تلقائياً)
# DB_PATH = "/app/data/ir_dataset_store.db"
# if not os.path.exists(os.path.dirname(DB_PATH)):
#     DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "ir_dataset_store.db"))

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     print("🚀 Preprocessing Microservice Ready For Operations.")
#     yield

# app = FastAPI(title="Preprocessing Service", lifespan=lifespan)
# print("Milad was here")
# preprocessor = Preprocessor()

# # نماذج التحقق البسيطة (DTO)
# class SinglePreprocessRequest(BaseModel):
#     text: str
#     dataset_name: Optional[str] = "Query"
#     pipeline_type: Optional[str] = "classical"  # "classical" (TF-IDF/BM25) أو "neural" (BERT/Embeddings)
#     stem: Optional[bool] = False
#     lemmatize: Optional[bool] = True
#     run_semantics: Optional[bool] = False
#     run_spellcheck: Optional[bool] = False

# class DatabaseBatchRequest(BaseModel):
#     dataset_name: str                     # مثل "quora" أو "lotte"
#     pipeline_type: Optional[str] = "classical"  # "classical" أو "neural"
#     batch_size: Optional[int] = 20000

# @app.get("/health")
# def health():
#     return {"status": "healthy"}

# @app.post("/preprocess")
# def preprocess_single_text(request: SinglePreprocessRequest):
#     """معالجة فجائية فورية للنصوص والاستعلامات الحية"""
#     processed_text, tokens, sentiment = preprocessor.preprocess(
#         text=request.text,
#         dataset_name=request.dataset_name,
#         pipeline_type=request.pipeline_type,
#         stem=request.stem,
#         lemmatize=request.lemmatize,
#         verbose=True,
#         run_semantics=request.run_semantics,
#         run_spellcheck=request.run_spellcheck
#     )
#     return {
#         "original_text": request.text, 
#         "processed_text": processed_text, 
#         "tokens": tokens,
#         "pipeline_type": request.pipeline_type,
#         "semantics": sentiment
#     }


# def database_batch_worker(dataset_name: str, pipeline_type: str, batch_size: int):
#     """محرك الـ Batch Processing لمعالجة قاعدة البيانات الضخمة دون استهلاك الذاكرة RAM"""
#     pipeline_type = pipeline_type.lower()
#     # تعديل ذكي: دمج نوع الأنبوب مع اسم المجموعة لمنع تضارب البيانات والسماح بحفظ تمثيلين لنفس المستند
#     db_dataset_identifier = f"{dataset_name}_{pipeline_type}"
    
#     print(f"\n[DATABASE WORKER] Triggered bulk processing for: {db_dataset_identifier.upper()}")
#     try:
#         conn = sqlite3.connect(DB_PATH, timeout=60.0)
#         cursor = conn.cursor()

#         # التحقق من حالة الإكمال الكلي أو الجزئي لتفادي التكرار أو البدء من منتصف دفعات تالفة
#         cursor.execute("SELECT COUNT(*) FROM documents WHERE dataset_name = ?", (dataset_name,))
#         total_docs = cursor.fetchone()[0]

#         cursor.execute("SELECT COUNT(*) FROM processed_documents WHERE dataset_name = ?", (db_dataset_identifier,))
#         processed_docs = cursor.fetchone()[0]

#         if processed_docs > 0:
#             if processed_docs >= total_docs:
#                 print(f"ℹ️ Dataset [{db_dataset_identifier.upper()}] already fully processed inside database ({processed_docs}/{total_docs}). Aborting.")
#                 conn.close()
#                 return
#             else:
#                 print(f"⚠️ Dataset [{db_dataset_identifier.upper()}] was partially processed ({processed_docs}/{total_docs}). Clearing and restarting.")
#                 cursor.execute("DELETE FROM processed_documents WHERE dataset_name = ?", (db_dataset_identifier,))
#                 conn.commit()

#         last_id = 0
#         processed_count = 0
#         while True:
#             # تطبيق مفهوم الـ Keyset Pagination (Seek Method) السريع لتجنب بطء OFFSET في الجداول الضخمة
#             cursor.execute(
#                 "SELECT id, doc_id, text FROM documents WHERE dataset_name = ? AND id > ? ORDER BY id ASC LIMIT ?",
#                 (dataset_name, last_id, batch_size)
#             )
#             rows = cursor.fetchall()
#             if not rows: 
#                 break # عند انتهاء المستندات نخرج من الحلقة

#             # تقسيم الدفعة الكبيرة إلى دفعات فرعية (Mini-batches) بحجم 1000 لتفادي استهلاك الذاكرة والـ CPU الزائد في NLTK
#             mini_batch_size = 1000
#             buffer = []
            
#             for i in range(0, len(rows), mini_batch_size):
#                 chunk = rows[i:i+mini_batch_size]
#                 chunk_texts = [r[2] for r in chunk]
                
#                 # طباعة السجلات للمجموعة الفرعية الأولى فقط للمراقبة المعمارية
#                 show_logs = (processed_count == 0 and i == 0)
                
#                 batch_results = preprocessor.preprocess_batch(
#                     texts=chunk_texts,
#                     dataset_name=f"{db_dataset_identifier} (Batch-Check)",
#                     pipeline_type=pipeline_type,
#                     stem=True,
#                     lemmatize=(pipeline_type == "classical"),
#                     verbose=show_logs,
#                     run_semantics=False,
#                     run_spellcheck=True
#                 )
                
#                 for (doc_id, text), (proc_text, _, _) in zip([(r[1], r[2]) for r in chunk], batch_results):
#                     if proc_text:
#                         buffer.append((db_dataset_identifier, doc_id, proc_text))
            
#             # حفظ الدفعة الحالية بالكامل لضمان سرعة معالجة قصوى وضغط عمليات الكتابة I/O
#             if buffer:
#                 cursor.executemany(
#                     "INSERT INTO processed_documents (dataset_name, doc_id, processed_text) VALUES (?, ?, ?)",
#                     buffer
#                 )
#                 conn.commit()
#                 processed_count += len(buffer)
#                 print(f"📦 [{pipeline_type.upper()} BATCH SAVED] Synchronized chunk up to row ID: {rows[-1][0]} (Processed {len(buffer)} docs)")
            
#             last_id = rows[-1][0]

#         conn.close()
#         print(f"🎉 Offline database preprocessing pipeline finalized for: {db_dataset_identifier.upper()}\n")
#     except Exception as e:
#         print(f"❌ Error during database batch processing: {e}")


# @app.post("/preprocess/database")
# def preprocess_entire_database(request: DatabaseBatchRequest, background_tasks: BackgroundTasks):
#     """تشغيل معالجة قاعدة البيانات الخام في الخلفية بصورة آمنة على دفعات"""
#     if not os.path.exists(DB_PATH):
#         raise HTTPException(status_code=404, detail=f"Database store file missing at expected path: {DB_PATH}")
    
#     if request.pipeline_type.lower() not in ["classical", "neural"]:
#         raise HTTPException(status_code=400, detail="Invalid pipeline_type. Must be 'classical' or 'neural'.")
        
#     background_tasks.add_task(database_batch_worker, request.dataset_name, request.pipeline_type, request.batch_size)
#     return {"message": f"Database pipeline for '{request.dataset_name}' [{request.pipeline_type}] successfully queued as a background process."}



import os
import sys
import sqlite3
import nltk

# Reconfigure stdout and stderr to prevent UnicodeEncodeError on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
if sys.stderr.encoding != 'utf-8':
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# تحميل حزم الـ NLTK الأساسية لمرة واحدة عند الاستيراد للتأكد من جاهزيتها قبل إنشاء الكائنات
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
from .preprocessor import Preprocessor

# كاشف آمن لمكان ملف قاعدة البيانات (يعمل في دكر ومحلياً تلقائياً)
DB_PATH = "/app/data/ir_dataset_store.db"
if not os.path.exists(os.path.dirname(DB_PATH)):
    DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "ir_dataset_store.db"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Preprocessing Microservice Ready For Operations.")
    yield

app = FastAPI(title="Preprocessing Service", lifespan=lifespan)
print("Milad was here")
preprocessor = Preprocessor()

# نماذج التحقق البسيطة (DTO)
class SinglePreprocessRequest(BaseModel):
    text: str
    dataset_name: Optional[str] = "Query"
    pipeline_type: Optional[str] = "classical"  # "classical" (TF-IDF/BM25) أو "neural" (BERT/Embeddings)
    stem: Optional[bool] = False
    lemmatize: Optional[bool] = True
    run_semantics: Optional[bool] = False
    run_spellcheck: Optional[bool] = False

class BatchPreprocessRequest(BaseModel):
    texts: list[str]
    dataset_name: Optional[str] = "Batch"
    pipeline_type: Optional[str] = "classical"
    stem: Optional[bool] = False
    lemmatize: Optional[bool] = True
    run_semantics: Optional[bool] = False
    run_spellcheck: Optional[bool] = False

class DatabaseBatchRequest(BaseModel):
    dataset_name: str                     # مثل "quora" أو "lotte"
    pipeline_type: Optional[str] = "classical"  # "classical" أو "neural"
    batch_size: Optional[int] = 20000

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/preprocess")
def preprocess_single_text(request: SinglePreprocessRequest):
    """معالجة فجائية فورية للنصوص والاستعلامات الحية"""
    processed_text, tokens, sentiment = preprocessor.preprocess(
        text=request.text,
        dataset_name=request.dataset_name,
        pipeline_type=request.pipeline_type,
        stem=request.stem,
        lemmatize=request.lemmatize,
        verbose=True,
        run_semantics=request.run_semantics,
        run_spellcheck=request.run_spellcheck
    )
    return {
        "original_text": request.text, 
        "processed_text": processed_text, 
        "tokens": tokens,
        "pipeline_type": request.pipeline_type,
        "semantics": sentiment
    }

@app.post("/preprocess/batch")
def preprocess_batch_text(request: BatchPreprocessRequest):
    """معالجة دفعية للنصوص والاستعلامات دفعة واحدة لتسريع التقييم"""
    results = preprocessor.preprocess_batch(
        texts=request.texts,
        dataset_name=request.dataset_name,
        pipeline_type=request.pipeline_type,
        stem=request.stem,
        lemmatize=request.lemmatize,
        verbose=True,
        run_semantics=request.run_semantics,
        run_spellcheck=request.run_spellcheck
    )
    return {
        "results": [
            {
                "processed_text": res[0],
                "tokens": res[1],
                "semantics": res[2]
            }
            for res in results
        ]
    }


def database_batch_worker(dataset_name: str, pipeline_type: str, batch_size: int):
    """محرك الـ Batch Processing لمعالجة قاعدة البيانات الضخمة دون استهلاك الذاكرة RAM"""
    pipeline_type = pipeline_type.lower()
    # تعديل ذكي: دمج نوع الأنبوب مع اسم المجموعة لمنع تضارب البيانات والسماح بحفظ تمثيلين لنفس المستند
    db_dataset_identifier = f"{dataset_name}_{pipeline_type}"
    
    print(f"\n[DATABASE WORKER] Triggered bulk processing for: {db_dataset_identifier.upper()}")
    try:
        conn = sqlite3.connect(DB_PATH, timeout=60.0)
        cursor = conn.cursor()

        # التحقق من حالة الإكمال الكلي أو الجزئي لتفادي التكرار أو البدء من منتصف دفعات تالفة
        cursor.execute("SELECT COUNT(*) FROM documents WHERE dataset_name = ?", (dataset_name,))
        total_docs = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM processed_documents WHERE dataset_name = ?", (db_dataset_identifier,))
        processed_docs = cursor.fetchone()[0]

        if processed_docs > 0:
            if processed_docs >= total_docs:
                print(f"ℹ️ Dataset [{db_dataset_identifier.upper()}] already fully processed inside database ({processed_docs}/{total_docs}). Aborting.")
                conn.close()
                return
            else:
                print(f"⚠️ Dataset [{db_dataset_identifier.upper()}] was partially processed ({processed_docs}/{total_docs}). Clearing and restarting.")
                cursor.execute("DELETE FROM processed_documents WHERE dataset_name = ?", (db_dataset_identifier,))
                conn.commit()
        # حساب عدد المستندات المعالجة الفريدة فقط
        # cursor.execute("""
        #     SELECT COUNT(DISTINCT doc_id)
        #     FROM processed_documents
        #     WHERE dataset_name = ?
        # """, (db_dataset_identifier,))
        # processed_docs = cursor.fetchone()[0]

        # # التحقق من الإكمال الحقيقي
        # if processed_docs == total_docs:
        #     print(
        #         f"ℹ️ Dataset [{db_dataset_identifier.upper()}] already fully processed "
        #         f"({processed_docs}/{total_docs})."
        #     )
        #     conn.close()
        #     return

        # # إيجاد آخر نقطة معالجة فعلية للاستكمال
        # cursor.execute("""
        #     SELECT COALESCE(MAX(d.id), 0)
        #     FROM documents d
        #     INNER JOIN processed_documents p
        #         ON CAST(d.doc_id AS TEXT) = p.doc_id
        #     WHERE d.dataset_name = ?
        #       AND p.dataset_name = ?
        # """, (dataset_name, db_dataset_identifier))

        # resume_id = cursor.fetchone()[0]

        # print(
        #     f"▶ Resume Mode Active | "
        #     f"Processed: {processed_docs}/{total_docs} | "
        #     f"Starting from documents.id > {resume_id}"
        # )

        # last_id = resume_id
        last_id = 0
        processed_count = 0
        while True:
            # تطبيق مفهوم الـ Keyset Pagination (Seek Method) السريع لتجنب بطء OFFSET في الجداول الضخمة
            cursor.execute(
                "SELECT id, doc_id, text FROM documents WHERE dataset_name = ? AND id > ? ORDER BY id ASC LIMIT ?",
                (dataset_name, last_id, batch_size)
            )
            rows = cursor.fetchall()
            if not rows:
                break # عند انتهاء المستندات نخرج من الحلقة

            # تقسيم الدفعة الكبيرة إلى دفعات فرعية (Mini-batches) بحجم 1000 لتفادي استهلاك الذاكرة والـ CPU الزائد في NLTK
            mini_batch_size = 1000
            buffer = []

            for i in range(0, len(rows), mini_batch_size):
                chunk = rows[i:i+mini_batch_size]
                chunk_texts = [r[2] for r in chunk]

                # طباعة السجلات للمجموعة الفرعية الأولى فقط للمراقبة المعمارية
                show_logs = (processed_count == 0 and i == 0)

                batch_results = preprocessor.preprocess_batch(
                    texts=chunk_texts,
                    dataset_name=f"{db_dataset_identifier} (Batch-Check)",
                    pipeline_type=pipeline_type,
                    stem=True,
                    lemmatize=(pipeline_type == "classical"),
                    verbose=show_logs,
                    run_semantics=False,
                    run_spellcheck=True 
                )

                for (doc_id, text), (proc_text, _, _) in zip([(r[1], r[2]) for r in chunk], batch_results):
                    if proc_text:
                        buffer.append((db_dataset_identifier, doc_id, proc_text))

            # حفظ الدفعة الحالية بالكامل لضمان سرعة معالجة قصوى وضغط عمليات الكتابة I/O
            if buffer:
                cursor.executemany(
                    "INSERT INTO processed_documents (dataset_name, doc_id, processed_text) VALUES (?, ?, ?)",
                    buffer
                )
                conn.commit()
                processed_count += len(buffer)
                print(f"📦 [{pipeline_type.upper()} BATCH SAVED] Synchronized chunk up to row ID: {rows[-1][0]} (Processed {len(buffer)} docs)")

            last_id = rows[-1][0]

        conn.close()
        print(f"🎉 Offline database preprocessing pipeline finalized for: {db_dataset_identifier.upper()}\n")
    except Exception as e:
        print(f"❌ Error during database batch processing: {e}")


@app.post("/preprocess/database")
def preprocess_entire_database(request: DatabaseBatchRequest, background_tasks: BackgroundTasks):
    """تشغيل معالجة قاعدة البيانات الخام في الخلفية بصورة آمنة على دفعات"""
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=404, detail=f"Database store file missing at expected path: {DB_PATH}")
    
    if request.pipeline_type.lower() not in ["classical", "neural"]:
        raise HTTPException(status_code=400, detail="Invalid pipeline_type. Must be 'classical' or 'neural'.")
        
    background_tasks.add_task(database_batch_worker, request.dataset_name, request.pipeline_type, request.batch_size)
    return {"message": f"Database pipeline for '{request.dataset_name}' [{request.pipeline_type}] successfully queued as a background process."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)