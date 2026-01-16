# app/schemas.py
from pydantic import BaseModel

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    processing_time: float
    timestamp: str
