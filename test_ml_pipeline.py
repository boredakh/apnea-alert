# File: test_ml_pipeline.py (in project root)
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.ml_pipeline import test_pipeline

print("Testing ML Pipeline...")
print("=" * 60)

pipeline, results = test_pipeline()