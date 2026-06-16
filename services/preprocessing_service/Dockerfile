FROM python:3.12-slim

# منع بايثون من كتابة ملفات الـ .pyc لضمان خفة الحاوية
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# تثبيت المترجمات الأساسية في النظام لبناء المكاتب الرياضية اللاحقة
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# نسخ ملف المتطلبات النظيف وتثبيته
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نسخ محتويات مجلد app بالكامل (بما فيه مجلد core الجديد) إلى داخل الحاوية
COPY ./app /app

# تشغيل الخدمة داخلياً وخارجياً على المنفذ المخصص لها 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]