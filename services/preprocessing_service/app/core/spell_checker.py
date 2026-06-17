import pkg_resources
import nltk
from symspellpy import SymSpell, Verbosity
from typing import List, Set

class TextSpellChecker:
    """Fast & accurate spell checker using SymSpell + validation dictionary."""

    _sym_spell = None
    _valid_words: Set[str] = set()

    def __init__(self, max_edit_distance: int = 2):
        # ---------- load SymSpell once (class level) ----------
        if TextSpellChecker._sym_spell is None:
            TextSpellChecker._sym_spell = SymSpell(
                max_dictionary_edit_distance=max_edit_distance
            )
            dict_path = pkg_resources.resource_filename(
                "symspellpy", "frequency_dictionary_en_82_765.txt"
            )
            TextSpellChecker._sym_spell.load_dictionary(
                dict_path, term_index=0, count_index=1
            )

        # ---------- load large English word list for validation ----------
        if not TextSpellChecker._valid_words:
            try:
                nltk.data.find("corpora/words")
            except LookupError:
                nltk.download("words", quiet=True)
            TextSpellChecker._valid_words = set(nltk.corpus.words.words())


        self.sym_spell = TextSpellChecker._sym_spell
        self.valid_words = TextSpellChecker._valid_words
        self.cache = {}   # per‑instance cache (still useful for repeated misspellings)

    def correct_single(self, token: str) -> str:
        """Correct one token – respects vocabulary to avoid false positives."""
        if token in self.cache:
            return self.cache[token]

        # If the word is already a valid English word, assume it's correct
        if token.lower() in self.valid_words:
            self.cache[token] = token
            return token

        # Otherwise, ask SymSpell for the best suggestion
        suggestions = self.sym_spell.lookup(
            token, Verbosity.CLOSEST, max_edit_distance=2
        )
        corrected = suggestions[0].term if suggestions else token
        self.cache[token] = corrected
        return corrected

    def correct_tokens(self, tokens: List[str]) -> List[str]:
        if not tokens:
            return []
        return [self.correct_single(t) for t in tokens]

    def correct_tokens_batch(self, token_lists: List[List[str]]) -> List[List[str]]:
        if not token_lists:
            return []
        return [self.correct_tokens(tokens) for tokens in token_lists]