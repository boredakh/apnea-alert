"""
Test loading YOUR actual apnea detection model
"""
import sys
import os
import joblib

print("🔍 Testing YOUR apnea detection model...")

# Get paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(project_root, "trained_models_retrained", "apnea_random_forest_20260121_140109.pkl")
scaler_path = os.path.join(project_root, "trained_models_retrained", "apnea_random_forest_20260121_140109_scaler.pkl")

print(f"Model path: {model_path}")
print(f"Scaler path: {scaler_path}")

# Check files exist
if not os.path.exists(model_path):
    print("❌ Model file not found!")
    sys.exit(1)

if not os.path.exists(scaler_path):
    print("⚠️ Scaler file not found, will continue without it")

try:
    # Load the model
    print("\n🔄 Loading your Random Forest model...")
    model = joblib.load(model_path)
    print(f"✅ Model loaded successfully!")
    print(f"   Type: {type(model).__name__}")
    
    # Check model properties
    if hasattr(model, 'n_estimators'):
        print(f"   Number of trees: {model.n_estimators}")
    if hasattr(model, 'n_features_in_'):
        print(f"   Expected features: {model.n_features_in_}")
    if hasattr(model, 'classes_'):
        print(f"   Classes: {model.classes_}")
    
    # Load the scaler
    print("\n🔄 Loading scaler...")
    scaler = joblib.load(scaler_path)
    print(f"✅ Scaler loaded successfully!")
    print(f"   Type: {type(scaler).__name__}")
    
    if hasattr(scaler, 'n_features_in_'):
        print(f"   Scaler expects: {scaler.n_features_in_} features")
    
    # Test a simple prediction
    print("\n🧪 Testing prediction...")
    
    # Create dummy features based on your model's expected input
    # These are just example values
    dummy_features = [
        0.001,  # mean
        0.35,   # std
        -1.2,   # min
        2.1,    # max
        3.3,    # range
        0.32,   # rms
        3.5,    # skewness
        18.5,   # kurtosis
        68.5,   # hr_mean
        8.2,    # hr_std
        0.88,   # rr_mean
        0.12,   # rr_std
        0.00015, # lf_power
        0.00008, # hf_power
        1.87    # lf_hf_ratio
    ]
    
    # Reshape for single prediction
    import numpy as np
    features_array = np.array([dummy_features])
    
    # Scale features
    scaled_features = scaler.transform(features_array)
    
    # Make prediction
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(scaled_features)
        print(f"✅ Prediction probabilities: {probabilities[0]}")
        
        apnea_prob = probabilities[0][1]  # Assuming class 1 is apnea
        prediction = 1 if apnea_prob >= 0.5 else 0
        confidence = max(probabilities[0])
        
        print(f"\n📊 Prediction results:")
        print(f"   Apnea probability: {apnea_prob:.3f}")
        print(f"   Prediction: {'Apnea' if prediction == 1 else 'Normal'}")
        print(f"   Confidence: {confidence:.3f}")
    else:
        prediction = model.predict(scaled_features)
        print(f"✅ Prediction: {prediction[0]}")
    
    print("\n🎉 SUCCESS! Your model is ready for the API!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()