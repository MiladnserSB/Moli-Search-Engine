import time
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

    def preprocess(self, 
                   text: str, 
                   dataset_name: str = "Query", 
                   pipeline_type: str = "classical",  # Options: "classical" (TF-IDF/BM25) OR "neural" (BERT/Embeddings)
                   stem: bool = False, 
                   lemmatize: bool = True, 
                   verbose: bool = True, 
                   run_semantics: bool = False, 
                   run_spellcheck: bool = False) -> tuple[str, list[str], dict]:
        
        if not text: 
            return "", [], {}

        start_total = time.time()

        if verbose:
            print(f"\n--- [PIPELINE ACTIVE | MODE: {pipeline_type.upper()} | DATASET: {dataset_name.upper()}] ---")

        # =========================================================================
        # PATH A: NEURAL PIPELINE (CRITICAL FIX FOR BERT / EMBEDDINGS)
        # =========================================================================
        if pipeline_type.lower() == "neural":
            if verbose: print("-> Processing Neural Path: Preserving syntax, casing, and stopwords...")
            
            # Clean HTML strings and structural URLs but avoid transforming character casing
            cleaned = self.normalizer.clean(text, lowercase=False)
            tokens = cleaned.split() # Base splitting to maintain dimensional indices if needed
            
            if verbose:
                print(f"   [NEURAL BYPASS COMPLETE] Token count: {len(tokens)}")
                print(f"   TOTAL ELAPSED TIME: {time.time() - start_total:.4f} seconds\n" + "-"*50)
            return cleaned, tokens, {}

        # =========================================================================
        # PATH B: CLASSICAL PIPELINE (OPTIMIZED FOR TF-IDF / BM25)
        # =========================================================================
        
        # Step 1: Structural Cleaning
        start_step = time.time()
        cleaned = self.normalizer.clean(text, lowercase=True)
        dur_step1 = time.time() - start_step
        if verbose: print(f"   Step 1 (Normalization) Duration: {dur_step1:.4f} seconds")

        # Step 2: Linguistic Split
        start_step = time.time()
        # Keep stopwords temporarily if spell check is enabled to avoid breaking context strings
        remove_stops_initially = not run_spellcheck
        tokens = self.tokenizer.tokenize_and_filter(cleaned, remove_stops=remove_stops_initially, is_lowercased=True)
        dur_step2 = time.time() - start_step
        if verbose: print(f"   Step 2 (Tokenization) Duration: {dur_step2:.4f} seconds")

        # Step 3: Symmetric Error Correction 
        dur_step3 = 0.0
        if run_spellcheck:
            if verbose: print("-> Step 3: Correcting vocabulary spelling errors...")
            start_step = time.time()
            tokens = self.spell_checker.correct_tokens(tokens)
            dur_step3 = time.time() - start_step
            if verbose: print(f"   Step 3 (Spell Check) Duration: {dur_step3:.4f} seconds")

            # Post Spell-Check Clean: Safely drop stopwords now that original typographies are fixed
            if hasattr(self.tokenizer, 'stop_words') and self.tokenizer.stop_words:
                tokens = [t for t in tokens if t not in self.tokenizer.stop_words]

        # Step 4: Lexical Reduction (Stemming vs Lemmatization)
        dur_step5 = 0.0
        if stem:
            if verbose: print("-> Step 4: Compiling Optimized Snowball Stemming...")
            start_step = time.time()
            tokens = self.stemmer.stem_tokens(tokens)
            dur_step5 = time.time() - start_step
            if verbose: print(f"   Step 4 (Stemming) Duration: {dur_step5:.4f} seconds")
        elif lemmatize:
            if verbose: print("-> Step 4: Compiling Dynamic POS Lemmatization...")
            start_step = time.time()
            tokens = self.lemmatizer.lemmatize_tokens(tokens)
            dur_step5 = time.time() - start_step
            if verbose: print(f"   Step 4 (Lemmatization) Duration: {dur_step5:.4f} seconds")

        processed_text = " ".join(tokens)

        # Step 5: Optional Semantic/Sentiment Extraction
        sentiment = {}
        if run_semantics:
            start_step = time.time()
            sentiment = {
                "lightweight": self.semantic_analyzer.analyze_lightweight(processed_text),
                "deep_learning": self.semantic_analyzer.analyze_deep_learning(processed_text)
            }
            if verbose: print(f"   Step 5 (Semantics) Duration: {time.time() - start_step:.4f} seconds")

        total_duration = time.time() - start_total

        if verbose:
            print(f"\n   [PERFORMANCE SUMMARY FOR DATASET: {dataset_name.upper()}]")
            print(f"   BEFORE: '{text[:75]}...'")
            print(f"   AFTER : '{processed_text[:75]}...'")
            print(f"   TOTAL ELAPSED TIME: {total_duration:.4f} seconds")
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
            print(f"\n--- [PIPELINE ACTIVE | MODE: {pipeline_type.upper()} | DATASET: {dataset_name.upper()} | BATCH SIZE: {len(texts)}] ---")

        # Neural pipeline path
        if pipeline_type_lower == "neural":
            if verbose: print("-> Processing Neural Path: Preserving syntax, casing, and stopwords...")
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

        # Classical pipeline path
        # Step 1: Normalization
        cleaned_texts = [self.normalizer.clean(t, lowercase=True) if t else "" for t in texts]
        
        # Step 2: Tokenization
        remove_stops_initially = not run_spellcheck
        token_lists = [
            self.tokenizer.tokenize_and_filter(c, remove_stops=remove_stops_initially, is_lowercased=True) if c else []
            for c in cleaned_texts
        ]

        # Step 3: Spell Check
        if run_spellcheck:
            token_lists = self.spell_checker.correct_tokens_batch(token_lists)
            # Post Spell-Check Clean
            if hasattr(self.tokenizer, 'stop_words') and self.tokenizer.stop_words:
                token_lists = [
                    [t for t in tokens if t not in self.tokenizer.stop_words]
                    for tokens in token_lists
                ]

        # Step 4: Lexical Reduction
        if stem:
            token_lists = [self.stemmer.stem_tokens(tokens) for tokens in token_lists]
        elif lemmatize:
            token_lists = self.lemmatizer.lemmatize_tokens_batch(token_lists)

        # Reconstruct processed texts
        processed_texts = [" ".join(tokens) for tokens in token_lists]

        # Step 5: Optional Semantics/Sentiment
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