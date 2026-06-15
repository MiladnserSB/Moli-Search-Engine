# Preprocessor Class with NLTK/SpaCy integration for IR Text Cleaning
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

class Preprocessor:
    def __init__(self):
        # Initialize NLTK components (downloaded via setup script)
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words("english"))
        except Exception:
            # Fallback in case stopwords aren't loaded properly
            self.stop_words = set()

    def clean_text(self, text: str) -> str:
        """
        Normalize text: lowercase, remove URLs, HTML tags, and non-alphanumeric characters.
        """
        if not text:
            return ""
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        # Remove HTML tags
        text = re.sub(r"<.*?>", "", text)
        # Keep only alphanumeric words and spaces
        text = re.sub(r"[^\w\s]", " ", text)
        # Collapse multiple spaces into one
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> list[str]:
        """
        Split normalized text into word tokens, filtering out numbers.
        """
        cleaned = self.clean_text(text)
        # Simple tokenization by whitespace
        tokens = cleaned.split()
        # Filter out numbers and keep only words longer than 1 character
        tokens = [t for t in tokens if t.isalpha() and len(t) > 1]
        return tokens

    def remove_stopwords(self, tokens: list[str]) -> list[str]:
        """
        Remove common stop words (e.g., 'the', 'is', 'at').
        """
        return [t for t in tokens if t not in self.stop_words]

    def stem(self, tokens: list[str]) -> list[str]:
        """
        Reduce tokens to their word stems (e.g., 'running' -> 'run').
        """
        return [self.stemmer.stem(t) for t in tokens]

    def lemmatize(self, tokens: list[str]) -> list[str]:
        """
        Reduce tokens to their base dictionary form (e.g., 'better' -> 'good').
        """
        # We assume noun POS tag for lemmatization by default (common in IR)
        return [self.lemmatizer.lemmatize(t, pos="v") for t in tokens]

    def preprocess(self, text: str, lowercase: bool = True, remove_punct: bool = True, 
                   remove_stops: bool = True, stem: bool = False, lemmatize: bool = True) -> tuple[str, list[str]]:
        """
        Full pipeline orchestration. Returns (processed_text_string, tokens_list)
        """
        if not text:
            return "", []
            
        # 1. Clean and normalize
        if lowercase or remove_punct:
            # Use clean_text which lowercases and removes punctuation
            tokens = self.tokenize(text)
        else:
            tokens = text.split()
            
        # 2. Stopwords removal
        if remove_stops:
            tokens = self.remove_stopwords(tokens)
            
        # 3. Stemming
        if stem:
            tokens = self.stem(tokens)
            
        # 4. Lemmatization (preferred over stemming for search accuracy)
        if lemmatize and not stem:
            tokens = self.lemmatize(tokens)
            
        processed_text = " ".join(tokens)
        return processed_text, tokens

