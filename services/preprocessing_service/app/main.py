from fastapi import FastAPI, HTTPException
from shared.schemas import PreprocessRequest, PreprocessResponse
from shared.config import settings
from preprocessor import Preprocessor
import uvicorn

app = FastAPI(title="Preprocessing Service", version="1.0.0")
preprocessor = Preprocessor()

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "preprocessing_service"}

@app.post("/preprocess", response_model=PreprocessResponse)
def preprocess_text(request: PreprocessRequest):
    try:
        # Execute the NLTK pipeline
        processed_text, tokens = preprocessor.preprocess(
            text=request.text,
            lowercase=request.lowercase,
            remove_punct=request.remove_punctuation,
            remove_stops=True,
            stem=request.stem,
            lemmatize=request.lemmatize
        )
        
        return PreprocessResponse(
            original_text=request.text,
            processed_text=processed_text,
            tokens=tokens
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=settings.PREPROCESSING_PORT, reload=True)
