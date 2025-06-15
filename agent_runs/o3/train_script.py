#!/usr/bin/env python
"""TRAIN_SCRIPT for the diabetes-readmission challenge.

Usage (example):
    python train_script.py \
        --train_path /mnt/data/diabetes-readmission/train/data-00000-of-00001.arrow \
        --model_output readmission_model.joblib

This script expects a training dataset that **includes** a binary target column
named **readmitted**. It automatically handles common dataset file formats
(CSV, Parquet, Feather, and Arrow IPC).  Categorical features are one-hot
encoded, numeric features are median-imputed & standard-scaled, and an
XGBoost classifier (with a HistGradientBoostingClassifier fallback) is trained.
The trained pipeline is exported via `joblib.dump` so it can be loaded and
used for inference on the hidden test set.

The script prints a validation AUC-ROC (20 % stratified split) for quick
feedback, then retrains on 100 % of the data before exporting the final model.
"""

import argparse
import pathlib
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Try to use XGBoost if it is available; otherwise fall back to
# HistGradientBoostingClassifier from scikit-learn (pure-python).
# ---------------------------------------------------------------------------

try:
    from xgboost import XGBClassifier  # type: ignore

    def _make_model():  # pragma: no cover – only compiled if XGBoost exists
        return XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="auc",
            n_jobs=-1,
            random_state=42,
        )

    _MODEL_NAME = "XGBoost"
except ModuleNotFoundError:  # noqa: D401 – graceful degradation
    from sklearn.ensemble import HistGradientBoostingClassifier

    def _make_model():  # type: ignore[override]
        return HistGradientBoostingClassifier(
            max_depth=6,
            max_iter=300,
            learning_rate=0.05,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=42,
        )

    _MODEL_NAME = "HistGradientBoosting"


# ---------------------------------------------------------------------------
# Utility: load many possible tabular formats seamlessly.
# ---------------------------------------------------------------------------

def load_table(path: str) -> pd.DataFrame:
    """Load Arrow IPC, Parquet, Feather, or CSV into a DataFrame."""
    ext = pathlib.Path(path).suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if ext == ".feather":
        return pd.read_feather(path)
    if ext == ".arrow":
        # Try HuggingFace datasets format first
        try:
            from datasets import Dataset
            dataset = Dataset.load_from_disk(pathlib.Path(path).parent)
            return dataset.to_pandas()
        except Exception:
            # Fallback to direct Arrow IPC reading
            with open(path, "rb") as f:
                reader = ipc.RecordBatchFileReader(f)
                table = reader.read_all()
            return table.to_pandas()
    raise ValueError(f"Unsupported file extension: {ext}")


# ---------------------------------------------------------------------------
# Main training routine.
# ---------------------------------------------------------------------------

def train(train_path: str, model_output: str) -> None:
    df = load_table(train_path)
    if "readmitted" not in df.columns:
        raise KeyError(
            "Expected target column 'readmitted' in training data but was not found."
        )

    y = df["readmitted"]
    X = df.drop(columns=["readmitted"])

    # Identify column types.
    cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    num_cols = (
        X.select_dtypes(include=["number", "bool", "datetime", "timedelta"])  # type: ignore[arg-type]
        .columns.tolist()
    )

    # Pre-processing pipelines.
    numeric_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ]
    )

    model = _make_model()
    clf = Pipeline(steps=[("prep", preprocessor), ("clf", model)])

    # Quick hold-out validation for feedback.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    with warnings.catch_warnings():  # silence Nan warnings for AUC if any
        warnings.simplefilter("ignore")
        clf.fit(X_train, y_train)
        val_scores = clf.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_scores)
    print(f"🔍 Validation AUC-ROC ({_MODEL_NAME}): {val_auc:.4f}")

    # Retrain on full data before export.
    clf.fit(X, y)
    joblib.dump(clf, model_output)
    print(f"✅ Trained {_MODEL_NAME} pipeline saved to: {model_output}")


# ---------------------------------------------------------------------------
# CLI entrypoint.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a diabetes readmission binary classifier and export the model."
    )
    parser.add_argument(
        "--train_path", required=True, help="Path to the training dataset file"
    )
    parser.add_argument(
        "--model_output",
        default="readmission_model.joblib",
        help="Output path for the serialized model",
    )
    args = parser.parse_args()

    try:
        train(args.train_path, args.model_output)
    except Exception as err:  # noqa: BLE001
        print("❌ Training failed:", err, file=sys.stderr)
        sys.exit(1)

