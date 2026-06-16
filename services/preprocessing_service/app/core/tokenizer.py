import nltk
from nltk.corpus import stopwords
from typing import List, Set

class TextTokenizer:
    """Handles linguistic tokenization without dropping critical contextual codes, standalone digits, or version numbers."""
    
    def __init__(self):
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords", quiet=True)

        try:
            base_stopwords = set(stopwords.words("english"))
            # Preserve negative semantic words to shield sentiment structures from flipping to true affirmatives
            negation_words = {
                "no", "not", "nor", "neither", "never", "cannot", 
                "don't", "can't", "won't", "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't"
            }
            self.stop_words: Set[str] = base_stopwords - negation_words
        except Exception:
            self.stop_words = set()
            
        self.valid_singles = {"c", "r", "g", "x", "y"}

    def _is_valid_token(self, token: str) -> bool:
        """Validates if a word consists of standard alphanumeric elements, respecting valid internal punctuation loops."""
        if token.isalnum():
            return True
        clean_t = token.replace("'", "").replace("-", "")
        return clean_t.isalnum()

    def tokenize_and_filter(self, cleaned_text: str, remove_stops: bool = True, is_lowercased: bool = False) -> List[str]:
        """Splits texts into logical tokens while filtering out vocabulary noise."""
        if not cleaned_text:
            return []

        tokens = cleaned_text.split()
        filtered_tokens = []
        
        for t in tokens:
            if self._is_valid_token(t):
                t_lower = t if is_lowercased else t.lower()
                
                # CRITICAL FIX: Ensure digits/numbers aren't discarded by a blanket string-length evaluation rule
                if len(t) > 1 or t_lower.isdigit() or t_lower in self.valid_singles:
                    if remove_stops and self.stop_words:
                        if t_lower not in self.stop_words:
                            filtered_tokens.append(t)
                    else:
                        filtered_tokens.append(t)
                        
        return filtered_tokens