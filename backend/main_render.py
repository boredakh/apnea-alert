"""
ApneaAlert API - Simplified for Render Deployment
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import joblib
import numpy as np
import os
import sys

print("🚀 Starting ApneaAlert API for Render...")
print(f"Python: {sys.version}")
print(f"Directory: {os.getcwd()}")

# Render provides PORT environment variable
PORT = int(os.getenv("PORT", "8000"))

# Create FastAPI app
app = FastAPI(
    title="ApneaAlert API",
    description="Sleep Apnea Detection using Machine Learning",
    version="1.0.0",
    docs_url="/docs"
)

# Try to load model, fallback to mock if not available
try:
    print("📦 Attempting to load ML model...")
    
    # Try multiple possible paths
    model_paths = [
        "trained_models_retrained/apnea_random_forest_20260121_140109.pkl",
        "/opt/render/project/src/backend/trained_models_retrained/apnea_random_forest_20260121_140109.pkl",
        "../trained_models_retrained/apnea_random_forest_20260121_140109.pkl"
    ]
    
    model = None
    scaler = None
    
    for path in model_paths:
        try:
            if os.path.exists(path):
                print(f"🔍 Found model at: {path}")
                model = joblib.load(path)
                
                # Try to load scaler
                scaler_path = path.replace(".pkl", "_scaler.pkl")
                if os.path.exists(scaler_path):
                    scaler = joblib.load(scaler_path)
                
                print(f"✅ Model loaded: {type(model).__name__}")
                if hasattr(model, 'n_estimators'):
                    print(f"   Trees: {model.n_estimators}")
                break
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            continue
    
    if model is None:
        print("⚠️ Could not load model. Using simulation mode.")
        
except Exception as e:
    print(f"❌ Model loading error: {e}")
    model = None
    scaler = None

# Feature names
FEATURE_NAMES = [
    'mean', 'std', 'min', 'max', 'range', 'rms', 'skewness', 'kurtosis',
    'hr_mean', 'hr_std', 'rr_mean', 'rr_std', 'lf_power', 'hf_power', 'lf_hf_ratio'
]

# Request model
class ApneaFeatures(BaseModel):
    mean: float = -0.0003
    std: float = 0.35
    min: float = -1.2
    max: float = 2.1
    range: float = 3.3
    rms: float = 0.32
    skewness: float = 3.5
    kurtosis: float = 18.5
    hr_mean: float = 68.5
    hr_std: float = 8.2
    rr_mean: float = 0.88
    rr_std: float = 0.12
    lf_power: float = 0.00015
    hf_power: float = 0.00008
    lf_hf_ratio: float = 1.87

@app.get("/")
async def root():
    return {
        "service": "ApneaAlert API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None,
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "POST /predict"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "environment": "production"
    }

@app.post("/predict")
async def predict(features: ApneaFeatures):
    if model is None:
        # Simulation mode
        apnea_prob = 0.85 if features.lf_hf_ratio > 1.5 else 0.25
    else:
        try:
            # Real prediction
            feature_values = [getattr(features, f) for f in FEATURE_NAMES]
            feature_array = np.array([feature_values])
            
            if scaler is not None:
                feature_array = scaler.transform(feature_array)
            
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(feature_array)[0]
                apnea_prob = float(probabilities[1]) if len(probabilities) > 1 else 0.5
            else:
                apnea_prob = 0.5
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    # Prepare response
    is_apnea = apnea_prob >= 0.5
    risk = "High" if apnea_prob > 0.7 else "Moderate" if apnea_prob > 0.3 else "Low"
    
    return {
        "success": True,
        "apnea_probability": apnea_prob,
        "prediction": 1 if is_apnea else 0,
        "prediction_label": "Apnea" if is_apnea else "Normal",
        "risk_level": risk,
        "confidence": max(apnea_prob, 1 - apnea_prob),
        "note": "Simulation" if model is None else "ML Prediction"
    }

if __name__ == "__main__":
    print(f"🌐 Server starting on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)