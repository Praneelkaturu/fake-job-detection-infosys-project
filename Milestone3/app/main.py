# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

from app.schemas import PredictRequest, PredictResponse

app = FastAPI(title="Job Post Fake Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # 👇 import here to avoid model loading during /health
    from app.utils import predict_job
    return predict_job(request.text)
