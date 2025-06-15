#!/usr/bin/env python
"""Evaluate the trained model on test data and report metrics."""

import argparse
import pathlib
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from train_script import load_table


def evaluate_model(model_path: str, test_path: str) -> None:
    """Load trained model and evaluate on test data."""
    # Load the trained model
    clf = joblib.load(model_path)
    
    # Load test data
    test_df = load_table(test_path)
    
    if "readmitted" not in test_df.columns:
        raise KeyError(
            "Expected target column 'readmitted' in test data but was not found."
        )
    
    y_test = test_df["readmitted"]
    X_test = test_df.drop(columns=["readmitted"])
    
    # Make predictions
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    
    # Print results
    print("📊 Test Set Evaluation Results")
    print("=" * 40)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc_roc:.4f}")
    print()
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"True Negatives:  {cm[0,0]:,}")
    print(f"False Positives: {cm[0,1]:,}")
    print(f"False Negatives: {cm[1,0]:,}")
    print(f"True Positives:  {cm[1,1]:,}")
    print()
    
    # Detailed classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Readmitted', 'Readmitted']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained model on test data"
    )
    parser.add_argument(
        "--model_path", 
        default="readmission_model.joblib",
        help="Path to the trained model file"
    )
    parser.add_argument(
        "--test_path", 
        required=True,
        help="Path to the test dataset file"
    )
    args = parser.parse_args()
    
    try:
        evaluate_model(args.model_path, args.test_path)
    except Exception as err:
        print("❌ Evaluation failed:", err, file=sys.stderr)
        sys.exit(1)