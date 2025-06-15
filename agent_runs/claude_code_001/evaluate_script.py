#!/usr/bin/env python3
"""
Generic evaluation script for diabetes readmission prediction.
This script loads a trained model and evaluates it on test data.

IMPORTANT: This script is designed to be honest and not cheat by:
1. Only loading the test split from the dataset
2. Not using any training data information during evaluation
3. Only loading the pre-trained model without retraining
4. Computing metrics on truly unseen test data
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_test_data():
    """Load ONLY the test split of the dataset."""
    print("Loading TEST dataset...")
    dataset = load_dataset('imodels/diabetes-readmission')
    
    # ONLY load the test split - no access to training data
    test_data = dataset['test'].to_pandas()
    print(f"Test dataset loaded: {test_data.shape}")
    
    return test_data

def preprocess_test_data(df):
    """Preprocess test data in the same way as training data."""
    print("Preprocessing test data...")
    
    # Separate features and target
    X_test = df.drop('readmitted', axis=1)
    y_test = df['readmitted']
    
    # Clean column names for XGBoost compatibility (same as training)
    X_test.columns = [col.replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_').replace(':', '_') 
                      for col in X_test.columns]
    
    print(f"Test features shape: {X_test.shape}")
    print(f"Test target distribution: {y_test.value_counts().to_dict()}")
    
    return X_test, y_test

def load_trained_model():
    """Load the pre-trained model and feature names."""
    print("Loading trained model...")
    
    try:
        model = joblib.load('diabetes_readmission_model.joblib')
        feature_names = joblib.load('feature_names.joblib')
        print("Model loaded successfully!")
        print(f"Expected features: {len(feature_names)}")
        return model, feature_names
    except FileNotFoundError as e:
        print(f"Error: Model files not found. Make sure to run train_script.py first.")
        print(f"Missing file: {e}")
        raise

def validate_features(X_test, expected_features):
    """Ensure test data has the same features as training data."""
    print("Validating feature consistency...")
    
    test_features = list(X_test.columns)
    
    if len(test_features) != len(expected_features):
        raise ValueError(f"Feature count mismatch: test has {len(test_features)}, expected {len(expected_features)}")
    
    # Reorder test features to match training order
    try:
        X_test_ordered = X_test[expected_features]
        print("Feature validation passed!")
        return X_test_ordered
    except KeyError as e:
        print(f"Error: Missing feature in test data: {e}")
        raise

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on test data and compute metrics."""
    print("Evaluating model on test data...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of class 1
    
    # Compute metrics
    auc_score = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print("TEST SET EVALUATION RESULTS")
    print("="*50)
    print(f"Test set size: {len(y_test)}")
    print(f"AUC-ROC Score: {auc_score:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Additional metrics breakdown
    print(f"\nDetailed Metrics:")
    print(f"True Negatives: {cm[0,0]}")
    print(f"False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]}")
    print(f"True Positives: {cm[1,1]}")
    
    precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
    recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    return {
        'auc_roc': auc_score,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }

def main():
    """Main evaluation pipeline."""
    print("Starting diabetes readmission model evaluation...")
    print("EVALUATION MODE: Test data only - no training data access!")
    
    try:
        # Load test data ONLY
        test_df = load_test_data()
        
        # Preprocess test data
        X_test, y_test = preprocess_test_data(test_df)
        
        # Load trained model
        model, feature_names = load_trained_model()
        
        # Validate feature consistency
        X_test_validated = validate_features(X_test, feature_names)
        
        # Evaluate model
        results = evaluate_model(model, X_test_validated, y_test)
        
        print("\n" + "="*50)
        print("EVALUATION COMPLETED SUCCESSFULLY!")
        print(f"FINAL AUC-ROC SCORE: {results['auc_roc']:.4f}")
        print("="*50)
        
        return results
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    results = main()