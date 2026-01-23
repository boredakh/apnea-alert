# File: fix_feature_issue.py
import pandas as pd
import numpy as np
import joblib

print("=" * 60)
print("FIXING FEATURE NAMES ISSUE")
print("=" * 60)

# Load the latest model
model_path = 'trained_models/apnea_random_forest_20260121_134233.pkl'
scaler_path = 'trained_models/apnea_random_forest_20260121_134233_scaler.pkl'

print(f"Loading model: {model_path}")
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

print(f"Model type: {type(model)}")
print(f"Has feature_importances_: {hasattr(model, 'feature_importances_')}")

if hasattr(model, 'feature_importances_'):
    print(f"Number of features in model: {len(model.feature_importances_)}")
    print(f"Feature importances: {model.feature_importances_}")

# Create the features file with correct feature names
print("\nCreating features file...")
feature_names = [
    'mean', 'std', 'min', 'max', 'range', 'rms', 'skewness', 'kurtosis',
    'hr_mean', 'hr_std', 'rr_mean', 'rr_std', 'lf_power', 'hf_power', 'lf_hf_ratio'
]

features_path = 'trained_models/apnea_random_forest_20260121_134233_features.txt'
with open(features_path, 'w') as f:
    for feature in feature_names:
        f.write(f"{feature}\n")

print(f"Created features file: {features_path}")
print(f"Features: {feature_names}")

# Test loading
print("\nTesting feature loading...")
with open(features_path, 'r') as f:
    loaded_features = [line.strip() for line in f]
print(f"Loaded {len(loaded_features)} features: {loaded_features[:5]}...")

print("\n" + "=" * 60)
print("FIX COMPLETE!")
print("=" * 60)