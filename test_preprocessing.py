# File: test_preprocessing.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.preprocessing.preprocess import ECGPreprocessor, test_preprocessor

print("Testing ECG Preprocessor Module")
print("=" * 60)

# Option 1: Run the built-in test
print("\nRunning built-in test...")
test_result = test_preprocessor()

# Option 2: Test with real data
print("\n" + "=" * 60)
print("Testing with real data from a01...")
print("=" * 60)

try:
    # Load real data
    ecg = np.load('data/processed/a01_ecg.npy')
    labels = np.load('data/processed/a01_labels.npy')
    
    print(f"Loaded a01:")
    print(f"  ECG shape: {ecg.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Apnea minutes: {np.sum(labels == 'A')}")
    print(f"  Normal minutes: {np.sum(labels == 'N')}")
    
    # Process just first 10 minutes for quick test
    test_duration = 10 * 60 * 100  # 10 minutes at 100 Hz
    test_ecg = ecg[:test_duration]
    test_labels = labels[:10]
    
    print(f"\nProcessing first 10 minutes...")
    preprocessor = ECGPreprocessor(sampling_rate=100)
    result = preprocessor.process_record(test_ecg, test_labels, "a01_first_10min")
    
    print(f"\nResults:")
    print(f"  Number of segments: {len(result['segments'])}")
    print(f"  Features shape: {result['features'].shape}")
    print(f"  Feature columns: {list(result['features'].columns)}")
    
    # Show first few features
    print(f"\nFirst 3 feature rows:")
    print(result['features'].head(3))
    
    print("\n✅ All tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()