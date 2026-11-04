
from pathlib import Path
import pickle
import re
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "sentiment_analysis_model.keras"
VECTORIZER_PATH = BASE_DIR / "model" / "tfidf_vectorizer.pkl"

app = FastAPI(
    title="Movie Review Sentiment API",
    description="Predict movie review sentiment",
    version="1.0.0"
)

model = None
vectorizer = None

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()

@app.on_event("startup")
def load_artifacts():
    global model, vectorizer

    if MODEL_PATH.exists() and VECTORIZER_PATH.exists():
        model = load_model(MODEL_PATH)
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)

class ReviewRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)

@app.get("/")
def home():
    return {"message": "Movie Review Sentiment API"}

@app.get("/health")
def health():
    return {
        "status": "ready" if model is not None else "artifacts_missing"
    }

@app.post("/predict")
def predict(data: ReviewRequest):

    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=503,
            detail="Trained model files are missing"
        )

    features = vectorizer.transform(
        [clean_text(data.text)]
    ).toarray().astype("float32")

    probability = float(
        model.predict(features, verbose=0)[0][0]
    )

    sentiment = "Positive" if probability >= 0.5 else "Negative"

    confidence = (
        probability if probability >= 0.5
        else 1 - probability
    ) * 100

    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 2),
        "positive_probability": round(probability, 4)
    }
