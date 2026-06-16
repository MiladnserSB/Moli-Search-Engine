from core.normalizer import TextNormalizer
from core.tokenizer import TextTokenizer
from core.spell_checker import TextSpellChecker  # Added
from core.lemmatizer import TextLemmatizer
from core.stemmer import TextStemmer
from core.semantic_analyzer import TextSemanticAnalyzer  # Added

class Preprocessor:
    def __init__(self):
        self.normalizer = TextNormalizer()
        self.tokenizer = TextTokenizer()
        self.spell_checker = TextSpellChecker()  # Added
        self.lemmatizer = TextLemmatizer()
        self.stemmer = TextStemmer()
        self.semantic_analyzer = TextSemanticAnalyzer()  # Added
        self.latest_sentiment = {}  # Added to safely store semantic metrics

    def preprocess(self, text: str, dataset_name: str = "Query", stem: bool = False, lemmatize: bool = True, verbose: bool = True, run_semantics: bool = False, run_spellcheck: bool = False) -> tuple[str, list[str]]:
        if not text: return "", []

        if verbose:
            print(f"\n--- [⚙️ PIPELINE ACTIVE ON: {dataset_name.upper()}] ---")
            print("-> Step 1: Normalizing text & stripping punctuation...")
        cleaned = self.normalizer.clean(text)

        if verbose: print("-> Step 2: Tokenizing text & removing stopwords...")
        tokens = self.tokenizer.tokenize_and_filter(cleaned)

        # Step 3 - Spell Checking (Runs on queries by default, or if explicitly requested)
        should_spellcheck = run_spellcheck or (dataset_name.lower() == "query")
        if should_spellcheck:
            if verbose: print("-> Step 3: Correcting spelling errors...")
            tokens = self.spell_checker.correct_tokens(tokens)

        # Renumbered old step 3 to Step 4 to keep the print sequence clean
        if stem:
            if verbose: print("-> Step 4: Compiling Porter Stemming...")
            tokens = self.stemmer.stem_tokens(tokens)
        elif lemmatize:
            if verbose: print("-> Step 4: Compiling Contextual POS Lemmatization...")
            tokens = self.lemmatizer.lemmatize_tokens(tokens)

        processed_text = " ".join(tokens)

        # Step 5 - Semantic Sentiment Analysis (Optional)
        if run_semantics:
            if verbose: print("-> Step 5: Executing Semantic Sentiment Analysis...")
            self.latest_sentiment = {
                "lightweight": self.semantic_analyzer.analyze_lightweight(processed_text),
                "deep_learning": self.semantic_analyzer.analyze_deep_learning(processed_text)
            }
            
            # Display the semantic output inside the logs if verbose is active
            if verbose:
                print(f"    - VADER Compound Score: {self.latest_sentiment['lightweight'].get('compound', 0.0)}")
                if "label" in self.latest_sentiment['deep_learning']:
                    print(f"    - Transformer Model:   {self.latest_sentiment['deep_learning']['label']} ({self.latest_sentiment['deep_learning']['score']})")
        else:
            self.latest_sentiment = {}

        if verbose:
            print(f" 📝 BEFORE: '{text[:80]}'")
            print(f" ✨ AFTER : '{processed_text[:80]}'")
            print("-" * 50)

        return processed_text, tokens