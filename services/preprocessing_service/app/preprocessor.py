import os
import sys
import time

# Ensure core directory is discoverable by adding the app directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from core.normalizer import TextNormalizer
from core.tokenizer import TextTokenizer
from core.spell_checker import TextSpellChecker
from core.lemmatizer import TextLemmatizer
from core.stemmer import TextStemmer
from core.semantic_analyzer import TextSemanticAnalyzer

class Preprocessor:
    def __init__(self):
        self.normalizer = TextNormalizer()
        self.tokenizer = TextTokenizer()
        self.spell_checker = TextSpellChecker()
        self.lemmatizer = TextLemmatizer()
        self.stemmer = TextStemmer()
        self.semantic_analyzer = TextSemanticAnalyzer()

        # GLOBAL spell cache shared across the entire batch session
        #   token → corrected_token
        self.spell_cache = {}

    def preprocess(self,
                   text: str,
                   dataset_name: str = "Query",
                   pipeline_type: str = "classical",
                   stem: bool = False,
                   lemmatize: bool = True,
                   verbose: bool = True,
                   run_semantics: bool = False,
                   run_spellcheck: bool = False) -> tuple[str, list[str], dict]:

        if not text:
            return "", [], {}

        start_total = time.time()
        if verbose:
            print(f"\n--- [PIPELINE ACTIVE | MODE: {pipeline_type.upper()} | "
                  f"DATASET: {dataset_name.upper()}] ---")

        # ======================== NEURAL PIPELINE ========================
        if pipeline_type.lower() == "neural":
            if verbose:
                print("-> Processing Neural Path: Preserving syntax, casing, and stopwords...")
            cleaned = self.normalizer.clean(text, lowercase=False)
            tokens = cleaned.split()
            if verbose:
                print(f"   [NEURAL BYPASS COMPLETE] Token count: {len(tokens)}")
                print(f"   TOTAL ELAPSED TIME: {time.time() - start_total:.4f} seconds\n" + "-"*50)
            return cleaned, tokens, {}

        # ======================= CLASSICAL PIPELINE ======================
        # Step 1 – Normalization
        t1 = time.time()
        cleaned = self.normalizer.clean(text, lowercase=True)
        dur1 = time.time() - t1
        if verbose:
            print(f"   Step 1 (Normalization) Duration: {dur1:.4f} seconds")

        # Step 2 – Tokenization (ALWAYS remove stopwords; negations are preserved)
        t2 = time.time()
        tokens = self.tokenizer.tokenize_and_filter(
            cleaned, remove_stops=True, is_lowercased=True
        )
        dur2 = time.time() - t2
        if verbose:
            print(f"   Step 2 (Tokenization) Duration: {dur2:.4f} seconds")

        # Step 3 – Spell check (only if requested)
        dur3 = 0.0
        if run_spellcheck:
            if verbose:
                print("-> Step 3: Correcting vocabulary spelling errors...")
            t3 = time.time()
            tokens = self._correct_tokens_with_cache(tokens)
            dur3 = time.time() - t3
            if verbose:
                print(f"   Step 3 (Spell Check) Duration: {dur3:.4f} seconds")

        # Step 4 – Lexical reduction (stemming or lemmatisation)
        dur4 = 0.0
        if stem:
            if verbose:
                print("-> Step 4: Compiling Optimized Snowball Stemming...")
            t4 = time.time()
            tokens = self.stemmer.stem_tokens(tokens)
            dur4 = time.time() - t4
            if verbose:
                print(f"   Step 4 (Stemming) Duration: {dur4:.4f} seconds")
        elif lemmatize:
            if verbose:
                print("-> Step 4: Compiling Dynamic POS Lemmatization...")
            t4 = time.time()
            tokens = self.lemmatizer.lemmatize_tokens(tokens)
            dur4 = time.time() - t4
            if verbose:
                print(f"   Step 4 (Lemmatization) Duration: {dur4:.4f} seconds")

        processed_text = " ".join(tokens)

        # Step 5 – Optional semantics
        sentiment = {}
        if run_semantics:
            t5 = time.time()
            sentiment = {
                "lightweight": self.semantic_analyzer.analyze_lightweight(processed_text),
                "deep_learning": self.semantic_analyzer.analyze_deep_learning(processed_text)
            }
            if verbose:
                print(f"   Step 5 (Semantics) Duration: {time.time() - t5:.4f} seconds")

        total = time.time() - start_total
        if verbose:
            print(f"\n   [PERFORMANCE SUMMARY FOR DATASET: {dataset_name.upper()}]")
            print(f"   BEFORE: '{text[:75]}...'")
            print(f"   AFTER : '{processed_text[:75]}...'")
            print(f"   TOTAL ELAPSED TIME: {total:.4f} seconds")
            print("-" * 50)

        return processed_text, tokens, sentiment

    def preprocess_batch(self,
                         texts: list[str],
                         dataset_name: str = "Batch",
                         pipeline_type: str = "classical",
                         stem: bool = False,
                         lemmatize: bool = True,
                         verbose: bool = True,
                         run_semantics: bool = False,
                         run_spellcheck: bool = False) -> list[tuple[str, list[str], dict]]:
        if not texts:
            return []

        start_total = time.time()
        pipeline_type_lower = pipeline_type.lower()

        if verbose:
            print(f"\n--- [PIPELINE ACTIVE | MODE: {pipeline_type.upper()} | "
                  f"DATASET: {dataset_name.upper()} | BATCH SIZE: {len(texts)}] ---")

        # ======================== NEURAL PIPELINE ========================
        if pipeline_type_lower == "neural":
            if verbose:
                print("-> Processing Neural Path: Preserving syntax, casing, and stopwords...")
            results = []
            for text in texts:
                if not text:
                    results.append(("", [], {}))
                    continue
                cleaned = self.normalizer.clean(text, lowercase=False)
                tokens = cleaned.split()
                results.append((cleaned, tokens, {}))
            if verbose:
                print(f"   [NEURAL BYPASS COMPLETE] Elapsed: {time.time() - start_total:.4f} seconds")
            return results

        # ======================= CLASSICAL PIPELINE ======================
        # Step 1 – Normalisation
        cleaned_texts = [
            self.normalizer.clean(t, lowercase=True) if t else ""
            for t in texts
        ]

        # Step 2 – Tokenisation (ALWAYS remove stopwords)
        token_lists = [
            self.tokenizer.tokenize_and_filter(c, remove_stops=True, is_lowercased=True) if c else []
            for c in cleaned_texts
        ]

        # Step 3 – Spell check using the global cache
        if run_spellcheck:
            for i in range(len(token_lists)):
                token_lists[i] = self._correct_tokens_with_cache(token_lists[i])

        # Step 4 – Lexical reduction
        if stem:
            token_lists = [self.stemmer.stem_tokens(tokens) for tokens in token_lists]
        elif lemmatize:
            token_lists = self.lemmatizer.lemmatize_tokens_batch(token_lists)

        processed_texts = [" ".join(tokens) for tokens in token_lists]

        # Step 5 – Semantics
        results = []
        for p_text, tokens in zip(processed_texts, token_lists):
            sentiment = {}
            if run_semantics and p_text:
                sentiment = {
                    "lightweight": self.semantic_analyzer.analyze_lightweight(p_text),
                    "deep_learning": self.semantic_analyzer.analyze_deep_learning(p_text)
                }
            results.append((p_text, tokens, sentiment))

        if verbose:
            print(f"   [CLASSICAL BATCH COMPLETE] Elapsed: {time.time() - start_total:.4f} seconds")

        return results

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------
    def _correct_tokens_with_cache(self, tokens: list) -> list:
        """
        Applies spell correction using the Preprocessor‑level cache.
        Tokens that have already been corrected in this session are
        returned instantly without calling the spell‑checker engine.
        """
        cache = self.spell_cache
        corrected = []
        for t in tokens:
            if t not in cache:
                cache[t] = self.spell_checker.correct_single(t)
            corrected.append(cache[t])
        return corrected