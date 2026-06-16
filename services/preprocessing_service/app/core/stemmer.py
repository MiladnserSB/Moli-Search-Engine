import nltk
from nltk.stem import SnowballStemmer
from typing import List

class TextStemmer:
    """Applies the optimized Snowball Stemming (Porter2) algorithm with case normalization."""
    
    def __init__(self, language: str = "english"):
        # Ensure NLTK data assets are present at runtime
        try:
            nltk.data.find("corpora/snowball_data")
        except LookupError:
            nltk.download("snowball_data", quiet=True)
            
        # Snowball is computationally crisper and more accurate than legacy Porter
        self.stemmer = SnowballStemmer(language=language)

    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """Stems a list of tokens efficiently, handling case normalization for accuracy."""
        # Fast-fail for empty or None collections
        if not tokens:
            return []
            
        # Stemmers are strictly case-sensitive. Lowercasing inline prevents 
        # mismatched roots (e.g., 'Running' vs 'running').
        return [self.stemmer.stem(t.lower()) for t in tokens]