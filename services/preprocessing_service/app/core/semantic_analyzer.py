from typing import Dict, Any, Optional
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class TextSemanticAnalyzer:
    """Handles semantic sentiment analysis using optimized industry-standard tools."""
    
    def __init__(self):
        # VADER is lightweight; load instantly
        self.vader_analyzer = SentimentIntensityAnalyzer()
        # Transformers are heavy; lazy-load on call to prevent initialization freezes
        self._transformer_classifier: Optional[Any] = None

    def analyze_lightweight(self, text: str) -> Dict[str, float]:
        """Runs a high-speed, rules-based semantic analysis using VADER."""
        if not text or not text.strip():
            return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
        return self.vader_analyzer.polarity_scores(text)

    def analyze_deep_learning(self, text: str) -> Dict[str, Any]:
        """Runs a high-accuracy semantic analysis using a Transformer model."""
        if not text or not text.strip():
            return {"label": "NEUTRAL", "score": 1.0}

        if self._transformer_classifier is None:
            from transformers import pipeline
            self._transformer_classifier = pipeline(
                "sentiment-analysis", 
                model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                revision="714eb0f"
            )

        try:
            prediction = self._transformer_classifier(text)[0]
            return {
                "label": prediction["label"],
                "score": round(prediction["score"], 4)
            }
        except Exception as e:
            return {"error": f"Transformer inference failed: {str(e)}"}