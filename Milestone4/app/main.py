# app/main.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

from app.schemas import PredictRequest, PredictResponse
from app.auth import fake_users, verify_password, create_access_token
from app.deps import admin_required
from app.utils import predict_job

# Create FastAPI app
app = FastAPI(title="Job Post Fake Detection API")

# Allow CORS (for frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

# Login endpoint
@app.post("/login")
def login(username: str, password: str):
    user = next((u for u in fake_users if u["username"] == username), None)
    if not user or not verify_password(password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = create_access_token({"sub": username, "role": user["role"]})
    return {"access_token": token, "token_type": "bearer"}

# Protected prediction endpoint (admin only)
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, user: dict = Depends(admin_required)):
    return predict_job(request.text)
