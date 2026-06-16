import re

class TextNormalizer:
    """Handles highly optimized text cleaning, stripping HTML, URLs, and punctuation while preserving contractions."""
    
    def __init__(self):
        # تجميع الأنماط مسبقاً (Pre-compiling) يمنع إعادة بناء الـ Regex مع كل استدعاء،
        # مما يعطي سرعة معالجة قصوى وضخمة جداً داخل الـ Loops وقاعدة البيانات.
        self.url_pattern = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
        self.html_pattern = re.compile(r"<.*?>")
        
        # إصلاح لغوي حرج: قمنا باستثناء الفاصلة العليا (') من الحذف [^\w\s']
        # هذا يمنع تمزق أو اختفاء الكلمات المختصرة الهامة مثل (we're, don't, it's) قبل وصولها للمصحح الإملائي.
        self.punct_pattern = re.compile(r"[^\w\s']")
        self.space_pattern = re.compile(r"\s+")

    def clean(self, text: str, lowercase: bool = True) -> str:
        """Cleans raw text data through a high-performance, sequence-compiled regex pipeline."""
        if not text:
            return ""
        
        # تحويل النص إلى حروف صغيرة كخطوة أولى لتوحيد المطابقة لاحقاً إذا طلب ذلك
        if lowercase:
            text = text.lower()
        
        # تطبيق عمليات التنظيف المتتالية باستخدام الأنماط الجاهزة في الذاكرة
        text = self.url_pattern.sub("", text)
        text = self.html_pattern.sub("", text)
        text = self.punct_pattern.sub(" ", text)         # استبدال الرموز وعلامات الترقيم الأخرى بمسافات
        text = self.space_pattern.sub(" ", text).strip()  # دمج المسافات المتكررة وقص الأطراف هوامش النص
        
        return text