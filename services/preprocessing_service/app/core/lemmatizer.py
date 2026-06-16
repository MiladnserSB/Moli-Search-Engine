import nltk
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from typing import List

class TextLemmatizer:
    """Optimized Lemmatizer leveraging dynamic POS tagging with cached dictionary lookups."""
    
    def __init__(self):
        # Automatically verify and download required NLTK assets at runtime
        for resource in ["averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "wordnet", "omw-1.4"]:
            try:
                if "tagger" in resource:
                    nltk.data.find(f"taggers/{resource}")
                else:
                    nltk.data.find(f"corpora/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)

        self.lemmatizer = WordNetLemmatizer()
        
        # OPTIMIZATION: Cache the dictionary mapping at initialization.
        # This prevents Python from rebuilding the dictionary on every single token loop.
        self.tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV
        }

    def _get_wordnet_pos(self, tag: str) -> str:
        """Maps NLTK POS tags to WordNet constants (Noun, Verb, Adj, Adv) via cache lookup."""
        if not tag:
            return wordnet.NOUN
        return self.tag_dict.get(tag[0].upper(), wordnet.NOUN)

    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Lemmatizes each word according to its true contextual grammatical role."""
        # Fast-fail guard clause for empty lists
        if not tokens:
            return []
            
        # Generate Part-of-Speech tags for tokens (returns a list of tuples)
        pos_tags = pos_tag(tokens)
        
        # OPTIMIZATION: Apply .lower() to the word inline during lemmatization.
        # NLTK's WordNet lexicon is lowercase; passing capitalized words breaks lookup accuracy.
        return [
            self.lemmatizer.lemmatize(word.lower(), pos=self._get_wordnet_pos(tag))
            for word, tag in pos_tags
        ]