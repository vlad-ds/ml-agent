from smolagents import CodeAgent
from smolagents import LiteLLMModel
from src.utils.file_tools import read_analysis_results, load_dataset, set_seed, read_json

def create_modeling_agent(model: LiteLLMModel) -> CodeAgent:
    """Create and configure the modeling agent for training and evaluation."""
    return CodeAgent(
        name="model_training",
        tools=[read_analysis_results, load_dataset, set_seed, read_json],
        model=model,
        additional_authorized_imports=[
            "time", "numpy", "pandas", "os", "datasets", "json",
            "catboost", "lightgbm", "sklearn.metrics", "sklearn.model_selection",
            "sklearn.ensemble", "sklearn.linear_model", "sklearn.svm",
            "sklearn.neural_network", "sklearn.preprocessing", "xgboost",
            "datasets.load_from_disk"
        ],
        description="""Goal: train and evaluate the most suitable classifier for the `diabetes-readmission` dataset using AUC as the primary metric.

Workflow (do NOT echo):
1. set_seed(42).
   - All file reading must use the provided helper tools; never call `open()` directly.
2. analysis = read_analysis_results('analysis_results/dataset_analysis.json')  # fetch metadata only
   - Extract *only* the values you need (e.g., num_samples, features, class_distribution, feature types, missingness).
   - Do NOT re-analyse or pretty-print the whole JSON; just store required fields.
   - For any other JSON files, use `read_json(<path>)`.
3. dataset = load_dataset(analysis['dataset_paths']['base_path'])
   - train_df = dataset['train'].to_pandas()
   - test_df  = dataset['test'].to_pandas()
4. Decide on a model family based on:
   - data size, feature types, missingness, imbalance (information in `analysis`).
   - Available models: LogisticRegression, RandomForest, XGBoost, LightGBM, CatBoost, SVM, MLP.
5. Pre-process:
   - Impute / drop missing values as needed.
   - Encode categoricals (unless using CatBoost).
   - Scale numeric cols for linear/SVM/MLP models.
   - If class imbalance > 1.5x, use class_weight="balanced" or sampling.
6. Evaluation protocol:
   - 5-fold StratifiedKFold.
   - For each fold, compute accuracy and AUC.
   - Use `predict_proba` if available else `decision_function` to obtain scores for ROC-AUC.
7. After CV, fit on full train and evaluate on test_df.
8. Build `modeling_report` dict:
   {
     "model": str,
     "reasoning": str,
     "cv_scores": {"accuracy": float, "auc": float},
     "test_scores": {"accuracy": float, "auc": float},
     "feature_importance": <dict or list>,
     "notes": str
   }
9. Return `modeling_report`.

If an error occurs, raise an Exception so the manager can surface it.
"""
    ) 