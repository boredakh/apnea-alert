"""
Simple model test - no paths, just imports
"""
print("Testing basic imports...")

try:
    import joblib
    print("✅ joblib works")
    
    import numpy as np
    print(f"✅ numpy works (version: {np.__version__})")
    
    import sklearn
    print(f"✅ scikit-learn works (version: {sklearn.__version__})")
    
    # Try to create a simple model and save/load it
    from sklearn.ensemble import RandomForestClassifier
    print("✅ RandomForestClassifier imported")
    
    # Create a tiny model to test
    X = [[0, 0], [1, 1]]
    y = [0, 1]
    model = RandomForestClassifier(n_estimators=2, random_state=42)
    model.fit(X, y)
    
    # Save and load it
    joblib.dump(model, 'test_model.pkl')
    loaded_model = joblib.load('test_model.pkl')
    print(f"✅ Can save/load models: {type(loaded_model).__name__}")
    
    # Clean up
    import os
    if os.path.exists('test_model.pkl'):
        os.remove('test_model.pkl')
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()