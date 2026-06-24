import os
import re
import pkg_resources
import nltk
from symspellpy import SymSpell, Verbosity
from nltk.corpus import wordnet


class QueryRefiner:
    def __init__(self):
        # 1. Initialize spelling checker
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2)
        dict_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_dictionary_en_82_765.txt"
        )
        self.sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)
        # Load validation words
        try:
            nltk.data.find("corpora/words")
        except LookupError:
            nltk.download("words", quiet=True)
        self.valid_words = set(nltk.corpus.words.words())
        # Load wordnet
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
        self.autocomplete_cache = {}

    def reduce_repeated_letters(self, word: str) -> str:
        # Convert repeated letters (3 or more) to 2 (e.g., coooool -> cool)
        return re.sub(r"(.)\1{2,}", r"\1\1", word)

    def correct_spelling(self, query: str) -> str:
        if not query:
            return ""
        tokens = query.lower().split()
        corrected_tokens = []

        for token in tokens:
            # First reduce letter repetitions
            reduced = self.reduce_repeated_letters(token)

            # If valid word, keep it
            if reduced in self.valid_words:
                corrected_tokens.append(reduced)
                continue

            # Otherwise use SymSpell
            suggestions = self.sym_spell.lookup(
                reduced, Verbosity.CLOSEST, max_edit_distance=2
            )
            corrected = suggestions[0].term if suggestions else token
            corrected_tokens.append(corrected)

        return " ".join(corrected_tokens)

    def expand_query(self, query: str) -> str:
        if not query:
            return ""

        # Correct spelling first
        corrected = self.correct_spelling(query)
        tokens = corrected.split()

        expanded_tokens = list(tokens)  # start with original corrected tokens

        # Simple stop words to avoid expanding common words
        stop_words = {
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
        }

        for token in tokens:
            if token in stop_words:
                continue
            # Get up to 2 synonyms per word to keep expansion clean and uncomplicated
            syns = []
            for syn in wordnet.synsets(token):
                for l in syn.lemmas():
                    name = l.name().replace("_", " ").lower()
                    if (
                        name != token
                        and name.isalpha()
                        and len(name) > 2
                        and name not in expanded_tokens
                    ):
                        syns.append(name)
                        if len(syns) >= 2:
                            break
                if len(syns) >= 2:
                    break
            expanded_tokens.extend(syns)

        return " ".join(expanded_tokens)

    def suggest_queries(self, prefix: str, dataset: str) -> list[str]:
        if not prefix or not dataset:
            return []

        prefix = prefix.lower().strip()

        # Load autocomplete index for dataset if not cached
        if dataset not in self.autocomplete_cache:
            from .config import settings

            index_path = os.path.join(settings.INDICES_DIR, f"{dataset}_index.json")
            if os.path.exists(index_path):
                print(f"[QueryRefiner] Loading autocomplete terms for {dataset}...")
                import json

                try:
                    with open(index_path, "r", encoding="utf-8") as f:
                        index = json.load(f)

                    # Convert to word list sorted by document frequency
                    term_freqs = [
                        (term, len(doc_ids))
                        for term, doc_ids in index.items()
                        if term.isalpha()
                    ]
                    term_freqs.sort(key=lambda x: x[1], reverse=True)

                    # Keep top 30,000 words to save RAM
                    self.autocomplete_cache[dataset] = term_freqs[:30000]
                except Exception as e:
                    print(f"Error loading index for autocomplete: {e}")
                    self.autocomplete_cache[dataset] = []
            else:
                self.autocomplete_cache[dataset] = []

        # Find matches starting with prefix
        matches = []
        for term, freq in self.autocomplete_cache[dataset]:
            if term.startswith(prefix):
                matches.append(term)
                if len(matches) >= 5:
                    break

        return matches
