import os
import re
import json
import nltk
from symspellpy import SymSpell, Verbosity
from nltk.corpus import wordnet
import importlib.resources
import sqlite3

# محاولة استيراد مدقق نحوي اختياري
try:
    import language_tool_python

    GRAMMAR_AVAILABLE = True
except ImportError:
    GRAMMAR_AVAILABLE = False


class TrieNode:
    """شجرة Trie بسيطة للاقتراحات التلقائية."""

    __slots__ = ("children", "freq", "term")

    def __init__(self):
        self.children = {}
        self.freq = 0
        self.term = None


class QueryRefiner:
    def __init__(self):
        # 1. المدقق الإملائي
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2)
        # استخدام importlib.resources بدلاً من pkg_resources (مهملة)
        with importlib.resources.path(
            "symspellpy", "frequency_dictionary_en_82_765.txt"
        ) as p:
            dict_path = str(p)
        self.sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)

        # تحميل قاموس الكلمات الصحيحة من NLTK
        try:
            nltk.data.find("corpora/words")
        except LookupError:
            nltk.download("words", quiet=True)
        self.valid_words = set(nltk.corpus.words.words())

        # تحميل WordNet
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)

        # كلمات التوقف
        self.stop_words = {
            "the",
            "a",
            "an",
            "in",
            "on",
            "at",
            "to",
            "for",
            "with",
            "by",
            "of",
            "and",
            "or",
            "but",
            "is",
            "are",
            "was",
            "were",
            "what",
            "how",
            "why",
            "i",
            "me",
            "my",
            "we",
            "our",
            "you",
            "your",
            "he",
            "she",
            "it",
            "they",
            "them",
            "this",
            "that",
            "these",
            "those",
            "do",
            "does",
            "did",
            "not",
        }

        # تخزين مؤقت للمرادفات لتجنب استدعاء WordNet المتكرر
        self.synonyms_cache = {}

        # هيكل Trie لكل مجموعة بيانات (للإكمال التلقائي)
        self.tries = {}

        # مدقق نحوي (اختياري)
        if GRAMMAR_AVAILABLE and self._should_enable_grammar():
            self.lang_tool = language_tool_python.LanguageTool("en-US")
            print("[Refiner] LanguageTool loaded for grammar checking.")
        else:
            self.lang_tool = None
            if not GRAMMAR_AVAILABLE:
                print(
                    "[Refiner] LanguageTool not installed. Grammar checking disabled."
                )
            else:
                print("[Refiner] Grammar checking disabled by config.")
        
        self._init_history_db()

    def _init_history_db(self):
        from .config import settings
        try:
            conn = sqlite3.connect(settings.DB_PATH)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS search_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT DEFAULT 'default_user',
                    dataset_name TEXT,
                    query TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
            conn.close()
            print("[Refiner] Search history table verified/created.")
        except Exception as e:
            print(f"[Refiner] Error initializing history DB: {e}")

    def log_query(self, query: str, dataset: str, user_id: str = 'default_user'):
        from .config import settings
        query = query.strip()
        if not query:
            return
        try:
            conn = sqlite3.connect(settings.DB_PATH)
            cursor = conn.cursor()
            # Prevent duplicates by removing previous identical query for this user/dataset
            cursor.execute('''
                DELETE FROM search_history 
                WHERE user_id = ? AND dataset_name = ? AND LOWER(query) = ?
            ''', (user_id, dataset, query.lower()))
            
            cursor.execute('''
                INSERT INTO search_history (user_id, dataset_name, query) 
                VALUES (?, ?, ?)
            ''', (user_id, dataset, query))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[Refiner] Error logging query: {e}")

    def get_history_suggestions(self, prefix: str, dataset: str, user_id: str = 'default_user') -> list[str]:
        from .config import settings
        prefix_clean = prefix.lower().strip()
        try:
            conn = sqlite3.connect(settings.DB_PATH)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT DISTINCT query FROM search_history 
                WHERE user_id = ? AND dataset_name = ? AND LOWER(query) LIKE ?
                ORDER BY timestamp DESC LIMIT 5
            ''', (user_id, dataset, f"{prefix_clean}%"))
            rows = cursor.fetchall()
            conn.close()
            return [row[0] for row in rows]
        except Exception as e:
            print(f"[Refiner] Error fetching history suggestions: {e}")
            return []

    def get_top_historical_terms(self, current_query: str, dataset: str, user_id: str = 'default_user', limit: int = 5) -> list[str]:
        from .config import settings
        try:
            conn = sqlite3.connect(settings.DB_PATH)
            cursor = conn.cursor()
            
            # استخراج الكلمات المفتاحية من الاستعلام الحالي لاستخدامها في التصفية
            tokens = current_query.lower().split()
            keywords = [re.sub(r'[^a-z]', '', w) for w in tokens if w not in self.stop_words and len(w) > 2]
            
            if not keywords:
                conn.close()
                return []

            # البحث عن الاستعلامات التاريخية التي تحتوي على كلمة مفتاحية واحدة على الأقل من الاستعلام الحالي
            # (لضمان عدم جلب استعلامات غير ذات صلة)
            like_clause = " OR ".join(["query LIKE ?" for _ in keywords])
            params = [user_id, dataset]
            for kw in keywords:
                params.append(f"%{kw}%")
                
            cursor.execute(f'''
                SELECT query FROM search_history 
                WHERE user_id = ? AND dataset_name = ? AND ({like_clause})
            ''', params)
            
            rows = cursor.fetchall()
            conn.close()
            
            word_counts = {}
            for row in rows:
                query = row[0]
                if not query:
                    continue
                words = query.lower().split()
                for w in words:
                    w_clean = re.sub(r'[^a-z]', '', w)
                    # هنا نستبعد الكلمات المفتاحية الموجودة أصلاً في الاستعلام الحالي لكي لا نكررها
                    if w_clean and w_clean not in self.stop_words and len(w_clean) > 2 and w_clean not in keywords:
                        word_counts[w_clean] = word_counts.get(w_clean, 0) + 1
            
            sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
            return [w for w, count in sorted_words[:limit]]
        except Exception as e:
            print(f"[Refiner] Error fetching top historical terms: {e}")
            return []

    def expand_query_with_history(self, query: str, dataset: str, user_id: str = 'default_user') -> str:
        base_expanded = self.expand_query(query)
        # تم تعديل الاستدعاء هنا لتمرير query الحالي كأول باراميتر
        top_history_terms = self.get_top_historical_terms(query, dataset, user_id, limit=3)
        if not top_history_terms:
            return base_expanded
            
        query_words = set(base_expanded.lower().split())
        added_terms = []
        for term in top_history_terms:
            if term not in query_words:
                added_terms.append(term)
                
        if added_terms:
            return f"{base_expanded} {' '.join(added_terms)}"
        return base_expanded

    def _should_enable_grammar(self):
        from .config import settings

        return settings.ENABLE_GRAMMAR_CHECK

    def reduce_repeated_letters(self, word: str) -> str:
        """تقليل الأحرف المتكررة (ثلاثة فأكثر) إلى حرفين."""
        return re.sub(r"(.)\1{2,}", r"\1\1", word)

    def correct_spelling(self, text: str) -> str:
        """تصحيح الأخطاء الإملائية في النص."""
        if not text:
            return ""
        tokens = text.lower().split()
        corrected_tokens = []

        for token in tokens:
            reduced = self.reduce_repeated_letters(token)
            # إذا كانت الكلمة صحيحة بعد التعديل، لا نصححها
            if reduced in self.valid_words:
                corrected_tokens.append(reduced)
                continue

            suggestions = self.sym_spell.lookup(
                reduced, Verbosity.CLOSEST, max_edit_distance=2
            )
            corrected = suggestions[0].term if suggestions else token
            corrected_tokens.append(corrected)

        return " ".join(corrected_tokens)

    def get_synonyms(self, word: str, max_syns: int = 2) -> list[str]:
        """استرجاع مرادفات الكلمة (مع تخزين مؤقت)."""
        if word in self.synonyms_cache:
            return self.synonyms_cache[word][:max_syns]

        syns = []
        # نقيد البحث بأول 3 معاني فقط (الأكثر شيوعاً) لتجنب المرادفات النادرة وغير المناسبة للسياق (مثل آلة -> سيارة)
        for synset in wordnet.synsets(word)[:3]:
            for lemma in synset.lemmas():
                name = lemma.name().replace("_", " ").lower()
                if name != word and name.isalpha() and len(name) > 2:
                    syns.append(name)
        # إزالة التكرار والحفاظ على الترتيب
        seen = set()
        unique = []
        for s in syns:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        self.synonyms_cache[word] = unique
        return unique[:max_syns]

    def expand_query(self, query: str) -> str:
        """توسيع الاستعلام بالمرادفات (مع تصحيح إملائي)."""
        if not query:
            return ""

        corrected = self.correct_spelling(query)
        tokens = corrected.split()
        expanded = list(tokens)

        for token in tokens:
            if token in self.stop_words:
                continue
            synonyms = self.get_synonyms(token, max_syns=2)
            for syn in synonyms:
                if syn not in expanded:
                    expanded.append(syn)

        return " ".join(expanded)

    def check_grammar(self, text: str) -> str:
        """تصحيح الأخطاء النحوية (إن وجد المدقق)."""
        if not self.lang_tool or not text:
            return text
        try:
            matches = self.lang_tool.check(text)
            return language_tool_python.utils.correct(text, matches)
        except Exception as e:
            print(f"Grammar check error: {e}")
            return text

    def generate_alternative_queries(
        self, original_query: str, max_alternatives: int = 3
    ) -> list[str]:
        """
        توليد استعلامات بديلة عن طريق استبدال كل كلمة محتوى بمرادفها الأول.
        ينتج عدة اقتراحات مختلفة يمكن للمستخدم تجربتها.
        """
        if not original_query:
            return []

        # تصحيح الإملاء أولاً
        base_query = self.correct_spelling(original_query)
        tokens = base_query.split()
        alternatives = []

        # لكل كلمة غير توقف، أنشئ استعلاماً بديلاً باستبدالها بمرادف
        for i, token in enumerate(tokens):
            if token in self.stop_words:
                continue
            syns = self.get_synonyms(token, max_syns=1)
            if syns:
                new_tokens = tokens.copy()
                new_tokens[i] = syns[0]
                alt_query = " ".join(new_tokens)
                if alt_query != base_query and alt_query not in alternatives:
                    alternatives.append(alt_query)

        # إضافة تصحيح نحوي للاستعلام الأساسي إذا اختلف
        grammar_corrected = self.check_grammar(base_query)
        if grammar_corrected != base_query and grammar_corrected not in alternatives:
            alternatives.insert(0, grammar_corrected)

        # قد نضيف أيضاً نسخة مقلوبة بسيطة (قلب ترتيب الكلمات) لإثراء الخيارات
        # لكن لتبسيط نكتفي بالاستبدالات
        return alternatives[:max_alternatives]

    def suggest_queries(self, prefix: str, dataset: str, user_id: str = 'default_user') -> list[dict]:
        """اقتراحات تلقائية بناءً على التاريخ وفهرس Trie الخاص بالجمل كاملة."""
        if not prefix or not dataset:
            return []

        prefix = prefix.lower().strip()
        if len(prefix) < 2:
            return []

        # 1. Get suggestions from user history
        history_queries = self.get_history_suggestions(prefix, dataset, user_id)

        # 2. Get suggestions from Trie (dataset queries)
        if dataset not in self.tries:
            self._load_trie_for_dataset(dataset)

        trie_root = self.tries.get(dataset)
        trie_suggestions = []
        if trie_root:
            node = trie_root
            possible = True
            for ch in prefix:
                if ch not in node.children:
                    possible = False
                    break
                node = node.children[ch]

            if possible:
                results = []
                self._collect_terms(node, prefix, results)
                # Sort by frequency (in this case, all have frequency 1 or based on occurrence)
                results.sort(key=lambda x: x[1], reverse=True)
                trie_suggestions = [term for term, _ in results[:10]]

        # Merge suggestions (prioritize history, prevent duplicates)
        seen = set()
        combined = []

        for q in history_queries:
            q_lower = q.lower().strip()
            if q_lower not in seen:
                seen.add(q_lower)
                combined.append({"text": q, "is_history": True})

        for q in trie_suggestions:
            q_lower = q.lower().strip()
            if q_lower not in seen:
                seen.add(q_lower)
                combined.append({"text": q, "is_history": False})

        return combined[:5]

    def _load_trie_for_dataset(self, dataset: str):
        from .config import settings

        db_path = settings.DB_PATH
        if not os.path.exists(db_path):
            print(f"[QueryRefiner] DB not found at {db_path}, cannot load dataset queries for Trie.")
            self.tries[dataset] = None
            return

        print(f"[QueryRefiner] Building Trie from database queries for {dataset}...")
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT text FROM queries WHERE dataset_name = ?", (dataset,))
            rows = cursor.fetchall()
            queries = [row[0] for row in rows if row[0]]
            conn.close()
        except Exception as e:
            print(f"[QueryRefiner] Error loading queries from database: {e}")
            self.tries[dataset] = None
            return

        trie_root = TrieNode()
        for q in queries:
            q_clean = q.lower().strip()
            if not q_clean:
                continue
            self._insert_trie(trie_root, q_clean, 1)

        self.tries[dataset] = trie_root
        print(f"[QueryRefiner] Trie built with {len(queries)} sentences for {dataset}.")

    def _insert_trie(self, root: TrieNode, word: str, freq: int):
        node = root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.freq = freq
        node.term = word

    def _collect_terms(self, node: TrieNode, prefix: str, results: list):
        """جمع كل الجمل المخزنة تحت العقدة."""
        if node.term:
            results.append((node.term, node.freq))
        for ch, child in node.children.items():
            self._collect_terms(child, prefix + ch, results)

    def _count_nodes(self, root: TrieNode) -> int:
        cnt = 1
        for child in root.children.values():
            cnt += self._count_nodes(child)
        return cnt