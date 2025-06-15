#!/usr/bin/env python3
"""
Training script for diabetes readmission prediction.
This script loads data, trains a model, and exports it for evaluation.
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the diabetes readmission dataset."""
    print("Loading dataset...")
    dataset = load_dataset('imodels/diabetes-readmission')
    train_data = dataset['train'].to_pandas()
    print(f"Dataset loaded: {train_data.shape}")
    return train_data

def preprocess_data(df):
    """Preprocess the data for training."""
    print("Preprocessing data...")
    
    # Separate features and target
    X = df.drop('readmitted', axis=1)
    y = df['readmitted']
    
    # Clean column names for XGBoost compatibility
    X.columns = [col.replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_').replace(':', '_') 
                 for col in X.columns]
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    print(f"Class balance: {y.value_counts(normalize=True).to_dict()}")
    
    return X, y

def create_models():
    """Create different model pipelines to try."""
    models = {
        'logistic_regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(random_state=42, max_iter=1000))
        ]),
        
        'random_forest': Pipeline([
            ('clf', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ))
        ]),
        
        'xgboost': Pipeline([
            ('clf', xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            ))
        ])
    }
    
    return models

def evaluate_models(models, X, y):
    """Evaluate models using cross-validation."""
    print("Evaluating models...")
    results = {}
    
    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Validate
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        auc_score = roc_auc_score(y_val, y_pred_proba)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        
        results[name] = {
            'model': model,
            'val_auc': auc_score,
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std()
        }
        
        print(f"{name} - Val AUC: {auc_score:.4f}, CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return results

def train_final_model(X, y, best_model):
    """Train the final model on all training data."""
    print("Training final model on full dataset...")
    best_model.fit(X, y)
    
    # Get training predictions for metrics
    y_pred_proba = best_model.predict_proba(X)[:, 1]
    train_auc = roc_auc_score(y, y_pred_proba)
    print(f"Training AUC: {train_auc:.4f}")
    
    return best_model

def main():
    """Main training pipeline."""
    print("Starting diabetes readmission training pipeline...")
    
    # Load data
    df = load_data()
    
    # Preprocess
    X, y = preprocess_data(df)
    
    # Create models
    models = create_models()
    
    # Evaluate models
    results = evaluate_models(models, X, y)
    
    # Select best model based on validation AUC
    best_model_name = max(results.keys(), key=lambda k: results[k]['val_auc'])
    best_model = results[best_model_name]['model']
    best_auc = results[best_model_name]['val_auc']
    
    print(f"\nBest model: {best_model_name} with validation AUC: {best_auc:.4f}")
    
    # Train final model on full dataset
    final_model = train_final_model(X, y, best_model)
    
    # Export model
    model_path = 'diabetes_readmission_model.joblib'
    joblib.dump(final_model, model_path)
    print(f"Model exported to: {model_path}")
    
    # Export feature names for later use
    feature_names = list(X.columns)
    joblib.dump(feature_names, 'feature_names.joblib')
    print(f"Feature names exported to: feature_names.joblib")
    
    print("Training completed successfully!")
    
    return final_model, feature_names

if __name__ == "__main__":
    model, features = main()