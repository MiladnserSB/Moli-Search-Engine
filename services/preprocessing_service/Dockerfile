FROM python:3.12-slim

# ضبط متغيرات بيئة بايثون لضمان سرعة المخرجات وعدم كتابة ملفات pyc
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# تثبيت المترجمات الأساسية في النظام لبناء المكاتب الرياضية مثل FAISS أو Pytrec
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# نسخ ملف متطلبات الخدمة المحددة وتثبيته
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# نسخ كود تطبيق FastAPI المكتوب داخل مجلد app
COPY ./app /app

# أمر التشغيل الافتراضي للحاوية
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]