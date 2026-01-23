"""
ApneaAlert API - Production Ready Version
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
import uvicorn
import joblib
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration from environment
PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "0.0.0.0")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
MODEL_PATH = os.getenv("MODEL_PATH", "trained_models_retrained/apnea_random_forest_20260121_140109.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "trained_models_retrained/apnea_random_forest_20260121_140109_scaler.pkl")

# Create the app
app = FastAPI(
    title="ApneaAlert API",
    description="Machine Learning API for Sleep Apnea Detection",
    version="1.0.0",
    docs_url="/docs" if DEBUG else None,  # Hide docs in production if needed
    redoc_url="/redoc" if DEBUG else None
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
model = None
scaler = None
feature_names = [
    'mean', 'std', 'min', 'max', 'range', 'rms', 'skewness', 'kurtosis',
    'hr_mean', 'hr_std', 'rr_mean', 'rr_std', 'lf_power', 'hf_power', 'lf_hf_ratio'
]

# Pydantic model for input
class ApneaFeatures(BaseModel):
    mean: float
    std: float
    min: float
    max: float
    range: float
    rms: float
    skewness: float
    kurtosis: float
    hr_mean: float
    hr_std: float
    rr_mean: float
    rr_std: float
    lf_power: float
    hf_power: float
    lf_hf_ratio: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "mean": -0.0003,
                "std": 0.35,
                "min": -1.2,
                "max": 2.1,
                "range": 3.3,
                "rms": 0.32,
                "skewness": 3.5,
                "kurtosis": 18.5,
                "hr_mean": 68.5,
                "hr_std": 8.2,
                "rr_mean": 0.88,
                "rr_std": 0.12,
                "lf_power": 0.00015,
                "hf_power": 0.00008,
                "lf_hf_ratio": 1.87
            }
        }

def load_model():
    """Load the ML model and scaler"""
    global model, scaler
    
    try:
        # Get absolute paths
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_abs_path = os.path.join(project_root, MODEL_PATH)
        scaler_abs_path = os.path.join(project_root, SCALER_PATH)
        
        print(f"📁 Loading model from: {model_abs_path}")
        
        if not os.path.exists(model_abs_path):
            raise FileNotFoundError(f"Model file not found: {model_abs_path}")
        
        # Load model
        model = joblib.load(model_abs_path)
        print(f"✅ Model loaded: {type(model).__name__}")
        
        # Load scaler
        if os.path.exists(scaler_abs_path):
            scaler = joblib.load(scaler_abs_path)
            print(f"✅ Scaler loaded")
        else:
            print("⚠️ Scaler not found, using unscaled features")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def predict_apnea(features_dict: Dict) -> Dict:
    """Make prediction using the loaded model"""
    global model, scaler, feature_names
    
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        # Convert features to array in correct order
        feature_values = []
        for feature in feature_names:
            value = features_dict.get(feature)
            if value is None:
                return {"error": f"Missing feature: {feature}"}
            feature_values.append(value)
        
        # Create numpy array
        features_array = np.array([feature_values])
        
        # Scale features if scaler exists
        if scaler is not None:
            features_array = scaler.transform(features_array)
        
        # Make prediction
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_array)[0]
            apnea_prob = float(probabilities[1])  # Probability of apnea
            normal_prob = float(probabilities[0])
        else:
            prediction = model.predict(features_array)[0]
            apnea_prob = 1.0 if prediction == 1 else 0.0
            normal_prob = 1.0 - apnea_prob
        
        # Determine result
        is_apnea = apnea_prob >= 0.5
        confidence = max(apnea_prob, normal_prob)
        
        # Risk level
        if apnea_prob < 0.3:
            risk = "Low"
        elif apnea_prob < 0.7:
            risk = "Moderate"
        else:
            risk = "High"
        
        return {
            "apnea_probability": apnea_prob,
            "normal_probability": normal_prob,
            "prediction": 1 if is_apnea else 0,
            "prediction_label": "Apnea" if is_apnea else "Normal",
            "confidence": confidence,
            "risk_level": risk,
            "threshold": 0.5
        }
        
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

# Load model on startup
@app.on_event("startup")
async def startup_event():
    """Load ML model when API starts"""
    print("🚀 Starting ApneaAlert API...")
    print(f"📊 Environment: {'Development' if DEBUG else 'Production'}")
    print(f"🌐 Host: {HOST}:{PORT}")
    
    success = load_model()
    if success:
        print("✅ Model loaded successfully!")
        print(f"   Model: {type(model).__name__}")
        if hasattr(model, 'n_estimators'):
            print(f"   Trees: {model.n_estimators}")
        print(f"   Features: {len(feature_names)}")
    else:
        print("⚠️ Model failed to load. Predictions will not work.")

# API Endpoints
@app.get("/")
def home():
    return {
        "message": "Welcome to ApneaAlert API",
        "version": "1.0.0",
        "service": "Sleep Apnea Detection",
        "model_status": "loaded" if model else "not loaded",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "POST /predict",
            "model_info": "GET /model/info"
        }
    }

@app.get("/health")
def health():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "service": "ApneaAlert API",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "timestamp": os.times().user  # Simple timestamp
    }

@app.post("/predict")
async def predict(features: ApneaFeatures):
    """Make an apnea prediction"""
    if model is None:
        raise HTTPException(status_code=503, detail="ML model not loaded")
    
    # Convert to dictionary
    features_dict = features.dict()
    
    # Make prediction
    result = predict_apnea(features_dict)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return {
        "success": True,
        **result
    }

@app.get("/model/info")
def model_info():
    """Get information about the loaded model"""
    if model is None:
        return {"status": "Model not loaded"}
    
    info = {
        "model_type": type(model).__name__,
        "features_count": len(feature_names),
        "features": feature_names,
        "model_loaded": True
    }
    
    # Add Random Forest specific info
    if hasattr(model, 'n_estimators'):
        info["n_estimators"] = model.n_estimators
    if hasattr(model, 'n_features_in_'):
        info["expected_features"] = model.n_features_in_
    
    return info

# Run the app
if __name__ == "__main__":
    print(f"📝 Documentation: http://{HOST}:{PORT}/docs")
    print(f"🔧 Health check: http://{HOST}:{PORT}/health")
    uvicorn.run(
        app, 
        host=HOST, 
        port=PORT,
        reload=DEBUG  # Auto-reload only in development
    )