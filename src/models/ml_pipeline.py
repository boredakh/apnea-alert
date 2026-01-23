# File: src/models/ml_pipeline.py
"""
Machine Learning Pipeline for Sleep Apnea Detection
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime

class ApneaMLPipeline:
    """Machine Learning pipeline for apnea detection"""
    
    def __init__(self, model_type='random_forest', random_state=42):
        self.model_type = model_type
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        
    def create_model(self):
        """Create ML model based on type"""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        print(f"Created {self.model_type} model")
        return self.model
    
    def prepare_features(self, features_df):
        """Prepare features for training"""
        # Separate features and labels
        X = features_df.drop(['label', 'segment_id'], axis=1, errors='ignore')
        y = features_df['label'].map({'A': 1, 'N': 0})  # Convert to binary: 1=Apnea, 0=Normal
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def train(self, X_train, y_train):
        """Train the model"""
        print(f"Training {self.model_type} model...")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Class distribution:")
        print(f"    Apnea (1): {np.sum(y_train == 1)} ({np.sum(y_train == 1)/len(y_train)*100:.1f}%)")
        print(f"    Normal (0): {np.sum(y_train == 0)} ({np.sum(y_train == 0)/len(y_train)*100:.1f}%)")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Create and train model
        if self.model is None:
            self.create_model()
        
        self.model.fit(X_train_scaled, y_train)
        
        print("Training complete!")
        return self
    
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_scaled)[:, 1]  # Probability of apnea
            predictions = (probabilities >= 0.5).astype(int)
        else:
            predictions = self.model.predict(X_scaled)
            probabilities = None
            
        return predictions, probabilities
    
    def evaluate(self, X_test, y_test, model_name="Model"):
        """Evaluate model performance"""
        print(f"\n{'='*60}")
        print(f"EVALUATING {model_name}")
        print(f"{'='*60}")
        
        predictions, probabilities = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        
        if probabilities is not None:
            roc_auc = roc_auc_score(y_test, probabilities)
        else:
            roc_auc = roc_auc_score(y_test, predictions)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        
        # Print results
        print(f"\nPerformance Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives:  {cm[0, 0]}")
        print(f"  False Positives: {cm[0, 1]}")
        print(f"  False Negatives: {cm[1, 0]}")
        print(f"  True Positives:  {cm[1, 1]}")
        
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, predictions, target_names=['Normal', 'Apnea']))
        
        # Feature importance (for Random Forest)
        if self.model_type == 'random_forest' and hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 10 Most Important Features:")
            print(feature_importance.head(10).to_string(index=False))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'predictions': predictions,
            'probabilities': probabilities
        }
    
    def cross_validate(self, X, y, cv_folds=5):
        """Perform cross-validation"""
        print(f"\n{'='*60}")
        print(f"CROSS-VALIDATION ({cv_folds}-fold)")
        print(f"{'='*60}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create model for CV
        cv_model = self.create_model()
        
        # Perform cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        cv_scores = cross_val_score(
            cv_model, X_scaled, y,
            cv=cv, scoring='roc_auc',
            n_jobs=-1
        )
        
        print(f"Cross-validation ROC-AUC scores:")
        for i, score in enumerate(cv_scores):
            print(f"  Fold {i+1}: {score:.4f}")
        
        print(f"\nMean ROC-AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        return cv_scores
    

    
    def load_model(self, model_path, scaler_path, features_path):
        """Load a trained model"""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        with open(features_path, 'r') as f:
            self.feature_names = [line.strip() for line in f]
        
        print(f"Model loaded from: {model_path}")
        return self

def save_model(self, model_dir='models'):
    """Save the trained model and scaler"""
    os.makedirs(model_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"apnea_{self.model_type}_{timestamp}"
    
    # Save model
    model_path = os.path.join(model_dir, f"{model_name}.pkl")
    joblib.dump(self.model, model_path)
    
    # Save scaler
    scaler_path = os.path.join(model_dir, f"{model_name}_scaler.pkl")
    joblib.dump(self.scaler, scaler_path)
    
    # Save feature names (check if they exist)
    if self.feature_names is not None:
        features_path = os.path.join(model_dir, f"{model_name}_features.txt")
        with open(features_path, 'w') as f:
            for feature in self.feature_names:
                f.write(f"{feature}\n")
        print(f"Features saved to: {features_path}")
    else:
        print("Warning: Feature names not available")
        features_path = None
    
    print(f"\nModel saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    
    return {
        'model_path': model_path,
        'scaler_path': scaler_path,
        'features_path': features_path
    }

def load_and_prepare_data(record_names, data_dir='data/processed'):
    """Load and prepare data from multiple records"""
    all_features = []
    
    for record in record_names:
        try:
            # Load features if they exist, otherwise process the record
            features_path = os.path.join(data_dir, f"{record}_features.csv")
            
            if os.path.exists(features_path):
                # Load preprocessed features
                features_df = pd.read_csv(features_path)
                print(f"Loaded features for {record}: {len(features_df)} segments")
            else:
                print(f"Features not found for {record}, need to preprocess first")
                continue
                
            all_features.append(features_df)
            
        except Exception as e:
            print(f"Error loading {record}: {e}")
    
    if not all_features:
        raise ValueError("No data loaded!")
    
    # Combine all features
    combined_df = pd.concat(all_features, ignore_index=True)
    
    print(f"\nCombined dataset:")
    print(f"  Total segments: {len(combined_df)}")
    print(f"  Apnea segments: {combined_df['label'].value_counts().get('A', 0)}")
    print(f"  Normal segments: {combined_df['label'].value_counts().get('N', 0)}")
    print(f"  Features: {combined_df.shape[1] - 2}")  # Excluding label and segment_id
    
    return combined_df


def create_sample_dataset():
    """Create a sample dataset for testing"""
    print("Creating sample dataset...")
    
    # Generate synthetic features
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate features similar to real ECG features
    features = {
        'mean': np.random.normal(0, 0.3, n_samples),
        'std': np.random.normal(0.25, 0.05, n_samples),
        'range': np.random.normal(2.5, 0.5, n_samples),
        'hr_mean': np.random.normal(65, 10, n_samples),
        'hr_std': np.random.normal(5, 2, n_samples),
        'lf_power': np.random.exponential(0.001, n_samples),
        'hf_power': np.random.exponential(0.001, n_samples),
        'lf_hf_ratio': np.random.exponential(1, n_samples)
    }
    
    # Create labels (60% apnea, 40% normal)
    labels = np.random.choice(['A', 'N'], n_samples, p=[0.6, 0.4])
    
    # Make apnea samples slightly different
    for i, label in enumerate(labels):
        if label == 'A':
            # Apnea features are slightly different
            features['hr_std'][i] += np.random.normal(2, 0.5)
            features['lf_hf_ratio'][i] += np.random.normal(0.5, 0.2)
    
    # Create DataFrame
    df = pd.DataFrame(features)
    df['label'] = labels
    df['segment_id'] = range(n_samples)
    
    print(f"Created sample dataset with {n_samples} samples")
    print(f"  Apnea: {np.sum(labels == 'A')}")
    print(f"  Normal: {np.sum(labels == 'N')}")
    
    return df


def test_pipeline():
    """Test the ML pipeline"""
    print("=" * 60)
    print("TESTING APNEA ML PIPELINE")
    print("=" * 60)
    
    # Create sample data
    data = create_sample_dataset()
    
    # Initialize pipeline
    pipeline = ApneaMLPipeline(model_type='random_forest')
    
    # Prepare features
    X, y = pipeline.prepare_features(data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nData split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Testing:  {len(X_test)} samples")
    
    # Train model
    pipeline.train(X_train, y_train)
    
    # Evaluate
    results = pipeline.evaluate(X_test, y_test, "Random Forest Model")
    
    # Cross-validation
    pipeline.cross_validate(X, y, cv_folds=5)
    
    # Save model
    saved_paths = pipeline.save_model('test_models')
    
    print("\n" + "=" * 60)
    print("PIPELINE TEST COMPLETE!")
    print("=" * 60)
    
    return pipeline, results


if __name__ == "__main__":
    test_pipeline()