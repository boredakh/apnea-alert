# File: process_real_data.py (in project root)
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from src.preprocessing.preprocess import ECGPreprocessor
from src.models.ml_pipeline_fixed import ApneaMLPipeline, load_and_prepare_data
import matplotlib.pyplot as plt
import seaborn as sns

def process_record_to_features(record_name, data_dir='data/processed', overwrite=False):
    """Process a single record and save features"""
    print(f"\nProcessing {record_name}...")
    print("-" * 40)
    
    # Check if features already exist
    features_path = os.path.join(data_dir, f"{record_name}_features.csv")
    
    if os.path.exists(features_path) and not overwrite:
        print(f"Features already exist for {record_name}")
        features_df = pd.read_csv(features_path)
        print(f"  Loaded {len(features_df)} segments")
        return features_df
    
    try:
        # Load ECG and labels
        ecg = np.load(os.path.join(data_dir, f"{record_name}_ecg.npy"))
        labels = np.load(os.path.join(data_dir, f"{record_name}_labels.npy"))
        
        print(f"  ECG shape: {ecg.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Apnea minutes: {np.sum(labels == 'A')}")
        print(f"  Normal minutes: {np.sum(labels == 'N')}")
        
        # Initialize preprocessor
        preprocessor = ECGPreprocessor(sampling_rate=100)
        
        # Process the record
        result = preprocessor.process_record(ecg, labels, record_name)
        
        # Save features
        features_df = result['features']
        features_df.to_csv(features_path, index=False)
        print(f"  Saved features to: {features_path}")
        
        return features_df
        
    except Exception as e:
        print(f"  Error processing {record_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_real_dataset(record_list, data_dir='data/processed'):
    """Create dataset from multiple records"""
    print("=" * 60)
    print("CREATING REAL DATASET FROM APNEA-ECG RECORDS")
    print("=" * 60)
    
    all_features = []
    successful_records = []
    failed_records = []
    
    for record in record_list:
        features_df = process_record_to_features(record, data_dir)
        
        if features_df is not None and len(features_df) > 0:
            all_features.append(features_df)
            successful_records.append(record)
            print(f"✓ {record}: {len(features_df)} segments")
        else:
            failed_records.append(record)
            print(f"✗ {record}: Failed to process")
    
    if not all_features:
        raise ValueError("No records were successfully processed!")
    
    # Combine all features
    combined_df = pd.concat(all_features, ignore_index=True)
    
    # Save combined dataset
    combined_path = os.path.join(data_dir, 'combined_features.csv')
    combined_df.to_csv(combined_path, index=False)
    
    print("\n" + "=" * 60)
    print("DATASET CREATION SUMMARY")
    print("=" * 60)
    print(f"Successful records: {len(successful_records)}")
    print(f"Failed records: {len(failed_records)}")
    print(f"\nCombined dataset:")
    print(f"  Total segments: {len(combined_df)}")
    
    if 'label' in combined_df.columns:
        apnea_count = combined_df['label'].value_counts().get('A', 0)
        normal_count = combined_df['label'].value_counts().get('N', 0)
        print(f"  Apnea segments: {apnea_count} ({apnea_count/len(combined_df)*100:.1f}%)")
        print(f"  Normal segments: {normal_count} ({normal_count/len(combined_df)*100:.1f}%)")
    
    print(f"  Features: {combined_df.shape[1] - 2}")  # Excluding label and segment_id
    print(f"\nDataset saved to: {combined_path}")
    
    # Plot class distribution
    if 'label' in combined_df.columns:
        plt.figure(figsize=(10, 6))
        combined_df['label'].value_counts().plot(kind='bar', color=['skyblue', 'lightcoral'])
        plt.title('Class Distribution in Dataset', fontsize=14)
        plt.xlabel('Label', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=0)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('data/processed/class_distribution.png', dpi=150)
        plt.show()
    
    return combined_df, successful_records

def analyze_features(features_df):
    """Analyze feature distributions"""
    print("\n" + "=" * 60)
    print("FEATURE ANALYSIS")
    print("=" * 60)
    
    # Separate features and labels
    if 'label' not in features_df.columns:
        print("No labels found in dataframe")
        return
    
    X = features_df.drop(['label', 'segment_id'], axis=1, errors='ignore')
    y = features_df['label'].map({'A': 1, 'N': 0})
    
    print(f"Feature shape: {X.shape}")
    
    # Basic statistics
    print("\nFeature statistics:")
    stats_df = X.describe().T
    print(stats_df[['mean', 'std', 'min', 'max']].head(10))
    
    # Correlation with labels
    correlation_df = pd.DataFrame({
        'feature': X.columns,
        'correlation_with_label': [X[col].corr(y) for col in X.columns]
    }).sort_values('correlation_with_label', key=abs, ascending=False)
    
    print("\nTop 10 features by correlation with apnea:")
    print(correlation_df.head(10).to_string(index=False))
    
    return X, y

def train_real_model(features_df, test_size=0.2):
    """Train model on real data"""
    print("\n" + "=" * 60)
    print("TRAINING MODEL ON REAL DATA")
    print("=" * 60)
    
    # Analyze features first
    X, y = analyze_features(features_df)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"\nData split:")
    print(f"  Training: {len(X_train)} samples ({X_train.shape[1]} features)")
    print(f"  Testing:  {len(X_test)} samples")
    print(f"  Class distribution (train):")
    print(f"    Apnea: {np.sum(y_train == 1)} ({np.sum(y_train == 1)/len(y_train)*100:.1f}%)")
    print(f"    Normal: {np.sum(y_train == 0)} ({np.sum(y_train == 0)/len(y_train)*100:.1f}%)")
    
    # Train Random Forest model
    print("\nTraining Random Forest model...")
    rf_pipeline = ApneaMLPipeline(model_type='random_forest')
    rf_pipeline.train(X_train, y_train)
    
    # Evaluate
    rf_results = rf_pipeline.evaluate(X_test, y_test, "Random Forest (Real Data)")
    
    # Cross-validation
    rf_pipeline.cross_validate(X, y, cv_folds=5)
    
    # Save model
    rf_paths = rf_pipeline.save_model('trained_models')
    
    # Train SVM model (optional)
    print("\n" + "=" * 60)
    print("TRAINING SVM MODEL")
    print("=" * 60)
    
    svm_pipeline = ApneaMLPipeline(model_type='svm')
    svm_pipeline.train(X_train, y_train)
    svm_results = svm_pipeline.evaluate(X_test, y_test, "SVM (Real Data)")
    
    # Compare models
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    
    comparison = pd.DataFrame({
        'Model': ['Random Forest', 'SVM'],
        'Accuracy': [rf_results['accuracy'], svm_results['accuracy']],
        'Precision': [rf_results['precision'], svm_results['precision']],
        'Recall': [rf_results['recall'], svm_results['recall']],
        'F1-Score': [rf_results['f1'], svm_results['f1']],
        'ROC-AUC': [rf_results['roc_auc'], svm_results['roc_auc']]
    })
    
    print(comparison.to_string(index=False))
    
    return {
        'rf_pipeline': rf_pipeline,
        'rf_results': rf_results,
        'svm_pipeline': svm_pipeline,
        'svm_results': svm_results,
        'comparison': comparison
    }

def main():
    """Main function to process data and train models"""
    print("APNEA DETECTION - REAL DATA PROCESSING")
    print("=" * 60)
    
    # Records to process (start with a few)
    records_to_process = ['a01', 'a02', 'a03', 'a04', 'a05']
    
    # Step 1: Process records and create dataset
    features_df, successful_records = create_real_dataset(records_to_process)
    
    # Step 2: Train models
    results = train_real_model(features_df)
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"Successfully processed: {len(successful_records)} records")
    print(f"Total segments: {len(features_df)}")
    print(f"Models saved to: trained_models/")
    
    return results

if __name__ == "__main__":
    main()

