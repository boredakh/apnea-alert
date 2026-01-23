# File: retrain_model.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from src.models.ml_pipeline_fixed import ApneaMLPipeline
import joblib
from sklearn.model_selection import train_test_split

print("=" * 60)
print("RETRAINING MODEL PROPERLY")
print("=" * 60)

# Load the dataset
df = pd.read_csv('data/processed/combined_features.csv')
print(f"Loaded dataset: {len(df)} segments")

# Prepare features
X = df.drop(['label', 'segment_id'], axis=1, errors='ignore')
y = df['label'].map({'A': 1, 'N': 0})

print(f"Features shape: {X.shape}")
print(f"Apnea samples: {sum(y == 1)} ({sum(y == 1)/len(y)*100:.1f}%)")
print(f"Normal samples: {sum(y == 0)} ({sum(y == 0)/len(y)*100:.1f}%)")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nData split:")
print(f"  Training: {len(X_train)} samples")
print(f"  Testing:  {len(X_test)} samples")

# Train model
print("\nTraining Random Forest model...")
pipeline = ApneaMLPipeline(model_type='random_forest')
pipeline.train(X_train, y_train)

# Evaluate
print("\nEvaluating model...")
results = pipeline.evaluate(X_test, y_test, "Retrained Model")

# Save model
print("\nSaving model...")
saved_paths = pipeline.save_model('trained_models_retrained')

# Test the saved model
print("\n" + "=" * 60)
print("TESTING SAVED MODEL")
print("=" * 60)

# Load the saved model
model = joblib.load(saved_paths['model_path'])
scaler = joblib.load(saved_paths['scaler_path'])

print(f"Model loaded: {type(model)}")
print(f"Scaler loaded: {type(scaler)}")

# Check if model is fitted
if hasattr(model, 'estimators_'):
    print(f"✅ Model is properly fitted!")
    print(f"   Number of trees: {len(model.estimators_)}")
    print(f"   Number of features: {model.n_features_in_}")
    
    # Test prediction
    print("\nTesting prediction...")
    test_sample = X_test.iloc[:1]
    test_scaled = scaler.transform(test_sample)
    prediction = model.predict_proba(test_scaled)
    
    print(f"   Test sample shape: {test_sample.shape}")
    print(f"   Prediction probabilities: {prediction[0]}")
    print(f"   Predicted class: {'Apnea' if prediction[0][1] > 0.5 else 'Normal'}")
    print(f"   True label: {'Apnea' if y_test.iloc[0] == 1 else 'Normal'}")
else:
    print("❌ Model is NOT fitted!")

print("\n" + "=" * 60)
print("RETRAINING COMPLETE!")
print("=" * 60)