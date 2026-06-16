import re

class TextNormalizer:
    """Handles basic text cleaning, stripping HTML, URLs, and punctuation."""
    def clean(self, text: str) -> str:
        if not text:
            return ""
        # Convert to lowercase
        text = text.lower()
        # Remove URLs and hyperlinks
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        # Remove HTML tags
        text = re.sub(r"<.*?>", "", text)
        # Keep only alphanumeric characters and spaces (removes punctuation cleanly)
        text = re.sub(r"[^\w\s]", " ", text)
        # Collapse multiple spaces into a single space
        text = re.sub(r"\s+", " ", text).strip()
        return text