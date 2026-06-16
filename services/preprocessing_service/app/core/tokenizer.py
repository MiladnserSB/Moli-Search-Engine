from nltk.corpus import stopwords

class TextTokenizer:
    """Handles splitting text into clear tokens and stripping stopwords."""
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words("english"))
        except Exception:
            self.stop_words = set()

    def tokenize_and_filter(self, cleaned_text: str, remove_stops: bool = True) -> list[str]:
        # Split by whitespace
        tokens = cleaned_text.split()
        # Keep alphanumeric tokens (to preserve terms like 'python3', 'v2')
        tokens = [t for t in tokens if t.isalnum()]
        
        if remove_stops:
            if not self.stop_words:
                try:
                    self.stop_words = set(stopwords.words("english"))
                except Exception:
                    pass
            tokens = [t for t in tokens if t not in self.stop_words and (len(t) > 1 or t in {"c", "r", "g", "x", "y"})]
        else:
            tokens = [t for t in tokens if len(t) > 1 or t in {"c", "r", "g", "x", "y"}]
        return tokens