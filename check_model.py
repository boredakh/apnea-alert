# File: check_model.py
import joblib
import numpy as np

print("=" * 60)
print("CHECKING MODEL FILES")
print("=" * 60)

# Check the problematic model
model_path = 'trained_models/apnea_random_forest_20260121_134233.pkl'
print(f"Checking: {model_path}")

try:
    model = joblib.load(model_path)
    print(f"Model type: {type(model)}")
    
    # Check if model is fitted
    if hasattr(model, 'estimators_'):
        print(f"Model is fitted: {len(model.estimators_) > 0}")
        if len(model.estimators_) > 0:
            print(f"Number of trees: {len(model.estimators_)}")
            print(f"Number of features expected: {model.n_features_in_}")
        else:
            print("ERROR: Model has 0 estimators (not fitted)")
    else:
        print("ERROR: Model doesn't have 'estimators_' attribute")
        
except Exception as e:
    print(f"Error loading model: {e}")

# Check the earlier model
print("\n" + "=" * 60)
print("Checking earlier model...")
model_path2 = 'trained_models/apnea_random_forest_20260121_115025.pkl'
print(f"Checking: {model_path2}")

try:
    model2 = joblib.load(model_path2)
    print(f"Model type: {type(model2)}")
    
    if hasattr(model2, 'estimators_'):
        print(f"Model is fitted: {len(model2.estimators_) > 0}")
        if len(model2.estimators_) > 0:
            print(f"Number of trees: {len(model2.estimators_)}")
            print(f"Number of features expected: {model2.n_features_in_}")
            print("✅ This model IS fitted properly!")
        else:
            print("ERROR: Model has 0 estimators")
    else:
        print("ERROR: Model doesn't have 'estimators_' attribute")
        
except Exception as e:
    print(f"Error loading model: {e}")

print("\n" + "=" * 60)
print("Checking scaler...")
scaler_path = 'trained_models/apnea_random_forest_20260121_134233_scaler.pkl'
try:
    scaler = joblib.load(scaler_path)
    print(f"Scaler type: {type(scaler)}")
    if hasattr(scaler, 'scale_'):
        print(f"Scaler is fitted: {scaler.scale_ is not None}")
        print(f"Number of features in scaler: {len(scaler.scale_)}")
    else:
        print("Scaler not properly fitted")
except Exception as e:
    print(f"Error loading scaler: {e}")