import nltk
from spellchecker import SpellChecker
from typing import List, Set

class TextSpellChecker:
    """Fast & accurate spell checker using pyspellchecker + validation dictionary."""

    _spell = None
    _valid_words: Set[str] = set()

    def __init__(self):
        # ---------- load SpellChecker once (class level) ----------
        if TextSpellChecker._spell is None:
            TextSpellChecker._spell = SpellChecker()

        # ---------- load large English word list for validation ----------
        if not TextSpellChecker._valid_words:
            try:
                nltk.data.find("corpora/words")
            except LookupError:
                nltk.download("words", quiet=True)
            TextSpellChecker._valid_words = set(nltk.corpus.words.words())

        self.spell = TextSpellChecker._spell
        self.valid_words = TextSpellChecker._valid_words
        self.cache = {}   # per‑instance cache (still useful for repeated misspellings)

    def correct_single(self, token: str) -> str:
        """Correct one token – respects vocabulary to avoid false positives."""
        if not token:
            return ""
            
        if token in self.cache:
            return self.cache[token]

        # If the word is already a valid English word, assume it's correct
        if token.lower() in self.valid_words:
            self.cache[token] = token
            return token

        # Otherwise, ask SpellChecker for the best suggestion
        corrected = self.spell.correction(token)
        # If no suggestion, keep the original token
        if corrected is None:
            corrected = token
            
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