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

    def correct_tokens_batch(self, token_lists: List[List[str]]) -> List[List[str]]:
        """Corrects a batch of token collections, optimizing unknown lookups and caching."""
        if not token_lists:
            return []

        # Find all unique tokens across the entire batch to query pyspellchecker in one go
        all_tokens = list(set(t for tokens in token_lists for t in tokens))
        misspelled = self.spell.unknown(all_tokens)

        # Batch fill cache for any unknown tokens not already cached
        for token in misspelled:
            if token not in self.cache:
                corrected = self.spell.correction(token)
                self.cache[token] = corrected if corrected is not None else token

        # Reconstruct corrected lists using the cache
        corrected_lists = []
        for tokens in token_lists:
            corrected_lists.append([
                self.cache[t] if t in misspelled else t
                for t in tokens
            ])
        return corrected_lists