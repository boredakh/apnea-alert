# File: system_check.py
import os
import sys

print("=" * 70)
print("APNEA ALERT - FINAL SYSTEM CHECK")
print("=" * 70)

# Check directory structure
directories = [
    'data/raw',
    'data/processed', 
    'src',
    'src/preprocessing',
    'src/models',
    'src/utils',
    'notebooks',
    'trained_models_retrained',
    'results'
]

print("\n📁 Directory Structure:")
all_good = True
for directory in directories:
    if os.path.exists(directory):
        print(f"  ✅ {directory}/")
    else:
        print(f"  ❌ {directory}/ (missing)")
        all_good = False

# Check key files
files = [
    'data/processed/combined_features.csv',
    'src/preprocessing/preprocess.py',
    'src/models/ml_pipeline_fixed.py',
    'process_real_data.py',
    'demo_final.py',
    'requirements.txt'
]

print("\n📄 Essential Files:")
for file in files:
    if os.path.exists(file):
        size_kb = os.path.getsize(file) / 1024
        print(f"  ✅ {file} ({size_kb:.1f} KB)")
    else:
        print(f"  ❌ {file} (missing)")
        all_good = False

# Check trained models
print("\n🤖 Trained Models:")
models_dir = 'trained_models_retrained'
if os.path.exists(models_dir):
    model_files = os.listdir(models_dir)
    if model_files:
        for file in model_files:
            if file.endswith('.pkl'):
                size_mb = os.path.getsize(os.path.join(models_dir, file)) / (1024 * 1024)
                print(f"  📦 {file} ({size_mb:.2f} MB)")
    else:
        print("  ❌ No model files found")
        all_good = False
else:
    print(f"  ❌ {models_dir} not found")
    all_good = False

# Check dataset
print("\n📊 Dataset Status:")
if os.path.exists('data/processed/combined_features.csv'):
    import pandas as pd
    df = pd.read_csv('data/processed/combined_features.csv')
    print(f"  ✅ {len(df)} segments")
    print(f"     Apnea: {df['label'].value_counts().get('A', 0)}")
    print(f"     Normal: {df['label'].value_counts().get('N', 0)}")
    print(f"     Features: {df.shape[1] - 2}")
else:
    print("  ❌ No dataset found")
    all_good = False

print("\n" + "=" * 70)
if all_good:
    print("✅ SYSTEM READY FOR DEPLOYMENT!")
    print("\nNext Steps:")
    print("1. python demo_final.py - Run final demo")
    print("2. Expand to more records (a06-a20)")
    print("3. Integrate with wearable APIs")
    print("4. Build mobile app interface")
else:
    print("⚠️  Some issues found. Please check above.")
print("=" * 70)