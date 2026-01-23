"""
Optimized version for production
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import joblib
import numpy as np
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Configuration
PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "0.0.0.0")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Pre-load model to avoid loading on each request
print("🚀 Loading ApneaAlert API...")

# Load model once at startup
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "trained_models_retrained", "apnea_random_forest_20260121_140109.pkl")
    scaler_path = os.path.join(project_root, "trained_models_retrained", "apnea_random_forest_20260121_140109_scaler.pkl")
    
    print(f"📁 Loading model: {os.path.basename(model_path)}")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    print(f"✅ Model loaded: {type(model).__name__} with {model.n_estimators} trees")
    print(f"✅ Scaler loaded")
    
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None
    scaler = None

# Feature names (must match training)
FEATURE_NAMES = [
    'mean', 'std', 'min', 'max', 'range', 'rms', 'skewness', 'kurtosis',
    'hr_mean', 'hr_std', 'rr_mean', 'rr_std', 'lf_power', 'hf_power', 'lf_hf_ratio'
]

# Create FastAPI app
app = FastAPI(
    title="ApneaAlert API",
    description="Machine Learning API for Sleep Apnea Detection",
    version="2.0.0",
    docs_url="/docs",
    redoc_url=None
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    """API information"""
    return {
        "service": "ApneaAlert API",
        "version": "2.0.0",
        "status": "operational",
        "model_loaded": model is not None,
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "POST /predict",
            "model_info": "/model/info"
        }
    }

@app.get("/health")
async def health():
    """Health check with performance metrics"""
    return {
        "status": "healthy",
        "service": "ApneaAlert API",
        "model_loaded": model is not None,
        "timestamp": time.time()
    }

@app.get("/model/info")
async def model_info():
    """Get model information"""
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "n_estimators": getattr(model, 'n_estimators', 'N/A'),
        "n_features": len(FEATURE_NAMES),
        "features": FEATURE_NAMES,
        "status": "loaded"
    }

@app.post("/predict")
async def predict(features: ApneaFeatures):
    """Make apnea prediction"""
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Convert to array
        feature_values = [getattr(features, f) for f in FEATURE_NAMES]
        feature_array = np.array([feature_values])
        
        # Scale
        if scaler:
            feature_array = scaler.transform(feature_array)
        
        # Predict
        probabilities = model.predict_proba(feature_array)[0]
        apnea_prob = float(probabilities[1])
        
        # Prepare response
        response = {
            "success": True,
            "apnea_probability": apnea_prob,
            "normal_probability": float(probabilities[0]),
            "prediction": 1 if apnea_prob >= 0.5 else 0,
            "prediction_label": "Apnea" if apnea_prob >= 0.5 else "Normal",
            "confidence": max(probabilities),
            "risk_level": "High" if apnea_prob > 0.7 else "Moderate" if apnea_prob > 0.3 else "Low",
            "processing_time_ms": round((time.time() - start_time) * 1000, 2)
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    print(f"🌐 Starting server on {HOST}:{PORT}")
    print(f"📚 API docs: http://{HOST}:{PORT}/docs")
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        access_log=False,  # Disable access logs for better performance
        loop="asyncio"
    )