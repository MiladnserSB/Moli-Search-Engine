import nltk
from nltk import pos_tag, pos_tag_sents
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from typing import List, Dict

class TextLemmatizer:
    """Highly optimized Lemmatizer leveraging dynamic pre-normalized POS tagging with class-level resource guards."""
    
    # متغبر على مستوى الفئة (Class-level flag) لضمان فحص وتحميل حزم NLTK لمرة واحدة فقط
    # طوال فترة تشغيل الخدمة، مما يوفر وقت الـ CPU والذاكرة عند إنشاء كائنات متعددة.
    _resources_loaded = False

    def __init__(self):
        if not TextLemmatizer._resources_loaded:
            for resource in ["averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "wordnet", "omw-1.4"]:
                try:
                    if "tagger" in resource:
                        nltk.data.find(f"taggers/{resource}")
                    else:
                        nltk.data.find(f"corpora/{resource}")
                except LookupError:
                    nltk.download(resource, quiet=True)
            TextLemmatizer._resources_loaded = True

        self.lemmatizer = WordNetLemmatizer()
        
        # تحسين: الاحتفاظ بالمفاتيح كما هي كرموز كبيرة (Uppercase) مباشرة لتفادي استدعاء دالة ()upper. داخل الـ Loops
        self.tag_dict: Dict[str, str] = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV
        }

    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Extracts morphological base lemmas by enforcing case normalization prior to POS tagging."""
        if not tokens:
            return []
            
        # 🚨 تعديل حرج جداً للدقة لغوية: تحويل الكلمات إلى حروف صغيرة (lowercase) قـبـل تمريرها لـ pos_tag
        # إذا دخلت كلمات تبدأ بحروف كبيرة (مثل بداية الجمل)، فإن NLTK يقوم بتصنيفها خاطئاً كـ (Proper Nouns - NNP)
        # مما يؤدي إلى فشل الـ Lemmatizer في إرجاع الأفعال والأسماء إلى جذورها الصحيحة.
        lowercased_tokens = [t.lower() for t in tokens]
        pos_tags = pos_tag(lowercased_tokens)
        
        # تحسين السرعة القصوى: الاستغناء عن الدالة المساعدة _get_wordnet_pos واستبدالها بـ get. مباشرة 
        # مع استخدام الحرف الأول tag[0] كـ Key سريع ومباشر داخل الـ List Comprehension
        return [
            self.lemmatizer.lemmatize(word, pos=self.tag_dict.get(tag[0], wordnet.NOUN))
            for word, tag in pos_tags
        ]

    def lemmatize_tokens_batch(self, token_lists: List[List[str]]) -> List[List[str]]:
        """Extracts morphological base lemmas for a batch of token collections using pos_tag_sents."""
        if not token_lists:
            return []
            
        # Enforce lowercase on all tokens for accurate POS tagging
        lowercased_lists = [[t.lower() for t in tokens] for tokens in token_lists]
        pos_tags_lists = pos_tag_sents(lowercased_lists)
        
        lemmatized_lists = []
        for pos_tags in pos_tags_lists:
            lemmatized_lists.append([
                self.lemmatizer.lemmatize(word, pos=self.tag_dict.get(tag[0], wordnet.NOUN))
                for word, tag in pos_tags
            ])
        return lemmatized_lists