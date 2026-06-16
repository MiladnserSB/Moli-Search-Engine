from spellchecker import SpellChecker
from typing import List

class TextSpellChecker:
    """Detects and automatically corrects spelling errors in a tokenized list."""
    
    def __init__(self):
        self.spell = SpellChecker()
        self.cache = {}

    def correct_tokens(self, tokens: List[str]) -> List[str]:
        """Corrects a collection of tokens utilizing optimized O(1) subset lookups and caching."""
        if not tokens:
            return []
            
        misspelled = self.spell.unknown(tokens)
        corrected_tokens = []
        
        for token in tokens:
            if token in misspelled:
                if token not in self.cache:
                    corrected = self.spell.correction(token)
                    self.cache[token] = corrected if corrected is not None else token
                corrected_tokens.append(self.cache[token])
            else:
                corrected_tokens.append(token)
                
        return corrected_tokens