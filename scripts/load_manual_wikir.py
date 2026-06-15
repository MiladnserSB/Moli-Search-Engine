import os
import sys
import csv
import zipfile
import io
from scripts.database import get_db_connection

# 🛠️ حل مشكلة OverflowError الخاصة بـ CSV على أنظمة Windows
_original_field_size_limit = csv.field_size_limit
def _safe_field_size_limit(*args, **kwargs):
    if args and args[0] > 2147483647:
        args = (2147483647,) + args[1:]
    return _original_field_size_limit(*args, **kwargs)
csv.field_size_limit = _safe_field_size_limit

# رفع حد حجم الحقل إلى أقصى درجة آمنة في ويندوز
csv.field_size_limit(2147483647)

DB_LABEL = "wikir_en1k"

def find_file_in_zip(zip_file: zipfile.ZipFile, target_suffix: str) -> str:
    """البحث ديناميكياً عن مسار الملف داخل الـ Zip بغض النظر عن المجلد الأب"""
    for name in zip_file.namelist():
        if name.lower().endswith(target_suffix.lower()):
            return name
    return None

def normalize_row(row: dict, type_: str) -> dict:
    """توحيد مسميات الأعمدة لضمان عدم الانهيار بسبب اختلاف العناوين (Headers)"""
    normalized = {}
    if type_ == 'doc':
        for k in ['id_document', 'doc_id', 'id', 'id_doc']:
            if k in row: normalized['doc_id'] = row[k]; break
        for k in ['text', 'body', 'document_text']:
            if k in row: normalized['text'] = row[k]; break
            
    elif type_ == 'query':
        for k in ['id_query', 'query_id', 'id']:
            if k in row: normalized['query_id'] = row[k]; break
        for k in ['text', 'query', 'query_text']:
            if k in row: normalized['text'] = row[k]; break
            
    elif type_ == 'qrel':
        for k in ['id_query', 'query_id']:
            if k in row: normalized['query_id'] = row[k]; break
        for k in ['id_document', 'doc_id']:
            if k in row: normalized['doc_id'] = row[k]; break
        for k in ['relevance', 'rel', 'label']:
            if k in row: normalized['relevance'] = row[k]; break
            
    return normalized

def stream_and_insert_csv(zip_file: zipfile.ZipFile, internal_path: str, type_: str, cursor, conn):
    """قراءة الـ CSV كتدفق مستمر وحفظه في قاعدة البيانات على دفعات لتقليل استهلاك الذاكرة"""
    print(f"⏳ جاري معالجة وحفظ ملف الـ {type_} [{internal_path}]...")
    
    with zip_file.open(internal_path) as f:
        # تحويل التدفق الثنائي إلى نصي لقرائته كـ CSV
        text_stream = io.TextIOWrapper(f, encoding='utf-8', errors='ignore')
        
        first_line = text_stream.readline()
        text_stream.seek(0)
        
        delimiter = '\t' if '\t' in first_line else ','
        has_header = any(kw in first_line.lower() for kw in ['id', 'text', 'query', 'document', 'relevance'])
        
        buffer = []
        count = 0
        batch_size = 50000 if type_ != 'query' else 10000
        
        if has_header:
            reader = csv.DictReader(text_stream, delimiter=delimiter)
            for row in reader:
                norm = normalize_row(row, type_)
                if type_ == 'doc' and 'doc_id' in norm and 'text' in norm:
                    buffer.append((DB_LABEL, norm['doc_id'], norm['text']))
                elif type_ == 'query' and 'query_id' in norm and 'text' in norm:
                    buffer.append((DB_LABEL, norm['query_id'], norm['text']))
                elif type_ == 'qrel' and 'query_id' in norm and 'doc_id' in norm:
                    buffer.append((DB_LABEL, norm['query_id'], norm['doc_id'], int(norm.get('relevance', 1))))
                
                count += 1
                if len(buffer) >= batch_size:
                    execute_batch_insert(cursor, type_, buffer)
                    conn.commit()
                    buffer = []
        else:
            # في حال عدم وجود عناوين، نعتمد على الترتيب الافتراضي للأعمدة
            reader = csv.reader(text_stream, delimiter=delimiter)
            for row in reader:
                if not row: continue
                if type_ == 'doc' and len(row) >= 2:
                    buffer.append((DB_LABEL, row[0], row[1]))
                elif type_ == 'query' and len(row) >= 2:
                    buffer.append((DB_LABEL, row[0], row[1]))
                elif type_ == 'qrel' and len(row) >= 3:
                    buffer.append((DB_LABEL, row[0], row[1], int(row[2])))
                
                count += 1
                if len(buffer) >= batch_size:
                    execute_batch_insert(cursor, type_, buffer)
                    conn.commit()
                    buffer = []
                    
        if buffer:
            execute_batch_insert(cursor, type_, buffer)
            conn.commit()
            
    print(f"✅ تم بنجاح إدخال {count} سجل في جدول {type_}s.")

def execute_batch_insert(cursor, type_, buffer):
    if type_ == 'doc':
        cursor.executemany("INSERT INTO documents (dataset_name, doc_id, text) VALUES (?, ?, ?)", buffer)
    elif type_ == 'query':
        cursor.executemany("INSERT INTO queries (dataset_name, query_id, text) VALUES (?, ?, ?)", buffer)
    elif type_ == 'qrel':
        cursor.executemany("INSERT INTO qrels (dataset_name, query_id, doc_id, relevance) VALUES (?, ?, ?, ?)", buffer)

def main():
    # الأماكن المتوقع وجود ملف الـ ZIP فيها
    possible_paths = [
        "wikIR1k.zip",
        os.path.join("ir_datasets_cache", "downloads", "wikIR1k.zip"),
        os.path.expanduser("~/Downloads/wikIR1k.zip")
    ]
    
    zip_path = None
    for path in possible_paths:
        if os.path.exists(path):
            zip_path = path
            break
            
    if not zip_path:
        print("❌ لم يتم العثور على ملف 'wikIR1k.zip'.")
        print("💡 من فضلك ضع الملف في المجلد الرئيسي للمشروع أو داخل مجلد الـ Downloads الخاص بجهازك.")
        return

    print(f"📦 تم العثور على داتا سيت يدوية في المسار: {zip_path}")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # تحقق سريع لمنع التكرار
    cursor.execute("SELECT COUNT(*) FROM documents WHERE dataset_name = ?", (DB_LABEL,))
    if cursor.fetchone()[0] > 0:
        print(f"ℹ️ البيانات الخاصة بـ {DB_LABEL} موجودة بالفعل في قاعدة البيانات. تم إلغاء العملية منعاً للتكرار.")
        conn.close()
        return

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_f:
            # 1. البحث عن ملف الوثائق ومعالجته
            doc_file = find_file_in_zip(zip_f, "documents.csv")
            if doc_file:
                stream_and_insert_csv(zip_f, doc_file, 'doc', cursor, conn)
            else:
                print("⚠️ لم يتم العثور على documents.csv داخل ملف الـ Zip!")

            # 2. البحث عن ملف استعلامات التدريب ومعالجته
            query_file = find_file_in_zip(zip_f, "training/queries.csv")
            if query_file:
                stream_and_insert_csv(zip_f, query_file, 'query', cursor, conn)
            else:
                print("⚠️ لم يتم العثور على training/queries.csv داخل ملف الـ Zip!")

            # 3. البحث عن ملف العلاقات (qrels) للتدريب ومعالجته
            qrel_file = find_file_in_zip(zip_f, "training/qrels.csv")
            if qrel_file:
                stream_and_insert_csv(zip_f, qrel_file, 'qrel', cursor, conn)
            else:
                print("⚠️ لم يتم العثور على training/qrels.csv داخل ملف الـ Zip!")
                
        print("\n🎉 🎉 اكتملت عملية المعالجة والـ Ingestion اليدوية بنجاح وبكفاءة Clean Code مطلقة!")
        
    except Exception as e:
        print(f"💥 حدث خطأ غير متوقع أثناء المعالجة: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()