"""
Test script to check if we can load the ML model
"""
import sys
import os

print("🔍 Testing ML model loading...")

# Add project root to path so we can find the model files
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"Project root: {project_root}")

# Try to import joblib
try:
    import joblib
    print("✅ joblib imported successfully")
except ImportError as e:
    print(f"❌ Failed to import joblib: {e}")
    sys.exit(1)

# Check if model files exist
model_path = os.path.join(project_root, "trained_models_retrained", "apnea_random_forest_20260121_140109.pkl")
scaler_path = os.path.join(project_root, "trained_models_retrained", "apnea_random_forest_20260121_140109_scaler.pkl")

print(f"\n📁 Looking for model files:")
print(f"   Model: {model_path}")
print(f"   Exists: {os.path.exists(model_path)}")
print(f"   Scaler: {scaler_path}")
print(f"   Exists: {os.path.exists(scaler_path)}")

if os.path.exists(model_path):
    try:
        print("\n🔄 Attempting to load model...")
        model = joblib.load(model_path)
        print(f"✅ Model loaded successfully!")
        print(f"   Model type: {type(model).__name__}")
        
        # Check if it's a Random Forest
        if hasattr(model, 'n_estimators'):
            print(f"   Number of trees: {model.n_estimators}")
        if hasattr(model, 'n_features_in_'):
            print(f"   Expected features: {model.n_features_in_}")
            
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
else:
    print("\n⚠️ Model file not found. Looking for alternatives...")
    
    # List available model files
    models_dir = os.path.join(project_root, "trained_models_retrained")
    if os.path.exists(models_dir):
        print(f"Files in {models_dir}:")
        for file in os.listdir(models_dir):
            if file.endswith('.pkl'):
                print(f"  - {file}")
    else:
        print("trained_models_retrained directory not found")
        
    # Check trained_models directory
    alt_dir = os.path.join(project_root, "trained_models")
    if os.path.exists(alt_dir):
        print(f"\nFiles in {alt_dir}:")
        for file in os.listdir(alt_dir):
            if file.endswith('.pkl'):
                print(f"  - {file}")