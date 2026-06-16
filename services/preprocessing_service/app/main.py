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
from preprocessor import Preprocessor

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
    stem: Optional[bool] = False
    lemmatize: Optional[bool] = True
    run_semantics: Optional[bool] = False
    run_spellcheck: Optional[bool] = False

class DatabaseBatchRequest(BaseModel):
    dataset_name: str  # مثل "quora_dev" أو "lotte_tech_dev"
    batch_size: Optional[int] = 20000

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/preprocess")
def preprocess_single_text(request: SinglePreprocessRequest):
    """معالجة فجائية فورية للنصوص والاستعلامات الحية"""
    processed_text, tokens = preprocessor.preprocess(
        text=request.text,
        dataset_name=request.dataset_name,
        stem=request.stem,
        lemmatize=request.lemmatize,
        verbose=True,
        run_semantics=request.run_semantics,
        run_spellcheck=request.run_spellcheck
    )
    # FIX: Added 'semantics' payload to capture the new VADER & Transformer calculations smoothly
    return {
        "original_text": request.text, 
        "processed_text": processed_text, 
        "tokens": tokens,
        "semantics": preprocessor.latest_sentiment
    }


def database_batch_worker(dataset_name: str, batch_size: int):
    """محرك الـ Batch Processing لمعالجة قاعدة البيانات الضخمة دون استهلاك الذاكرة RAM"""
    print(f"\n[DATABASE WORKER] Triggered bulk processing for: {dataset_name.upper()}")
    try:
        conn = sqlite3.connect(DB_PATH, timeout=60.0)
        cursor = conn.cursor()

        # تجنب التكرار: إذا كانت المجموعة معالجة مسبقاً، نتوقف
        cursor.execute("SELECT COUNT(*) FROM processed_documents WHERE dataset_name = ?", (dataset_name,))
        if cursor.fetchone()[0] > 0:
            print(f"ℹ️ Dataset [{dataset_name.upper()}] already processed inside database. Aborting.")
            conn.close()
            return

        offset = 0
        while True:
            # تطبيق مفهوم الـ Lazy Loading عبر سحب دفعات محددة الحجم فقط بالتتالي
            cursor.execute(
                "SELECT doc_id, text FROM documents WHERE dataset_name = ? LIMIT ? OFFSET ?",
                (dataset_name, batch_size, offset)
            )
            rows = cursor.fetchall()
            if not rows: break # عند انتهاء المستندات نخرج من الحلقة

            buffer = []
            for idx, (doc_id, text) in enumerate(rows):
                # فكرة ذكية: نطبع تقرير الخطوات خطوة بخطوة للمستند الأول فقط في كل دفعة لمعاينة المهندسة
                show_logs = (idx == 0)
                
                proc_text, _ = preprocessor.preprocess(
                    text=text,
                    dataset_name=f"{dataset_name} (Chunk-Row-Check)",
                    stem=False,
                    lemmatize=True,
                    verbose=show_logs,
                    run_semantics=False,
                    run_spellcheck=False
                )
                if proc_text:
                    buffer.append((dataset_name, doc_id, proc_text))

            # حفظ الدفعة الحالية بالكامل لضمان سرعة معالجة قصوى
            if buffer:
                cursor.executemany(
                    "INSERT INTO processed_documents (dataset_name, doc_id, processed_text) VALUES (?, ?, ?)",
                    buffer
                )
                conn.commit()
                print(f"📦 [BATCH SAVED] Successfully synchronized offset chunk up to row: {offset + len(rows)}")

            offset += batch_size

        conn.close()
        print(f"🎉 Offline database preprocessing pipeline finalized for: {dataset_name.upper()}\n")
    except Exception as e:
        print(f"❌ Error during database batch processing: {e}")


@app.post("/preprocess/database")
def preprocess_entire_database(request: DatabaseBatchRequest, background_tasks: BackgroundTasks):
    """تشغيل معالجة قاعدة البيانات الخام في الخلفية بصورة آمنة على دفعات"""
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=404, detail=f"Database store file missing at expected path: {DB_PATH}")
    
    background_tasks.add_task(database_batch_worker, request.dataset_name, request.batch_size)
    return {"message": f"Database pipeline for '{request.dataset_name}' successfully queued as an offline background process."}