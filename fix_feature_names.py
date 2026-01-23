# File: fix_feature_names.py
import pandas as pd
import numpy as np

print("Checking feature names in combined_features.csv...")
df = pd.read_csv('data/processed/combined_features.csv')

print(f"\nColumns in the dataset:")
for i, col in enumerate(df.columns):
    print(f"{i:2}. '{col}'")

print(f"\nDataset shape: {df.shape}")
print(f"Feature columns (excluding label and segment_id):")

feature_cols = [col for col in df.columns if col not in ['label', 'segment_id']]
print(feature_cols)
print(f"\nTotal features: {len(feature_cols)}")

# Check if there are any columns with name 'None' or empty
print("\nChecking for problematic column names...")
problematic = [col for col in df.columns if col is None or col == '' or col == 'None']
if problematic:
    print(f"Problematic columns: {problematic}")
else:
    print("No problematic column names found")