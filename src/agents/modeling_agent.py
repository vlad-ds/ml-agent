from smolagents import CodeAgent
from smolagents import LiteLLMModel
import os
from datasets import load_from_disk
from src.utils.file_tools import read_analysis_results

def create_modeling_agent(model: LiteLLMModel) -> CodeAgent:
    """Create and configure the modeling agent for training and evaluation."""
    return CodeAgent(
        name="modeling",
        tools=[read_analysis_results],
        model=model,
        additional_authorized_imports=[
            "time", "numpy", "pandas", "os", "datasets", "json",
            "catboost", "sklearn.metrics", "sklearn.model_selection",
            "sklearn.ensemble", "sklearn.linear_model", "sklearn.svm",
            "sklearn.neural_network", "sklearn.preprocessing", "xgboost",
            "datasets.load_from_disk"
        ],
        description="""You are a machine learning expert. Your task is to analyze the dataset characteristics and choose the most appropriate model for the diabetes-readmission dataset.

Follow these exact steps:
1. First, read the analysis results using the read_analysis_results tool:
   analysis_results = read_analysis_results('analysis_results/dataset_analysis.json')
   
   Use these results to understand:
   - Number of samples
   - Types of features
   - Missing values
   - Class distribution
   - Feature characteristics

2. Load and prepare the dataset using the paths from analysis_results:
   dataset = load_from_disk(analysis_results['dataset_paths']['base_path'])
   train_data = dataset["train"].to_pandas()
   test_data = dataset["test"].to_pandas()

3. Based on the analysis results, choose the most appropriate model:
   Consider these factors:
   - Dataset size:
     * Small (< 10k samples): Logistic Regression, SVM, or Random Forest
     * Medium (10k-100k): Random Forest, XGBoost, or LightGBM
     * Large (> 100k): Gradient Boosting (XGBoost, LightGBM, CatBoost)
   
   - Feature types:
     * Many categorical features: CatBoost or LightGBM
     * Many numerical features: XGBoost or Random Forest
     * Mixed types: CatBoost or LightGBM
   
   - Data quality:
     * Many missing values: CatBoost
     * High cardinality: LightGBM or CatBoost
     * Many outliers: Random Forest or XGBoost
   
   - Class distribution:
     * Imbalanced: Use class weights or sample weights
     * Balanced: Any model with appropriate parameters

4. Prepare the data according to the chosen model's requirements:
   - Handle missing values appropriately
   - Encode categorical variables if needed
   - Scale numerical features if needed
   - Handle class imbalance if present

5. Set up cross-validation:
   n_splits = 5
   kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

6. Train and evaluate the chosen model:
   - Use appropriate hyperparameters based on the data characteristics
   - For each fold:
     - Train the model
     - Calculate accuracy and AUC
     - Store the results

7. Train the final model on all training data and evaluate on test data:
   - Use early stopping if available
   - Use the best model from cross-validation
   - Calculate feature importance
   - Calculate test set metrics

8. Return:
   - The chosen model type and detailed reasoning for the choice
   - Dataset characteristics that influenced the choice
   - Average cross-validation scores (accuracy and AUC)
   - Test set performance
   - Feature importance from the final model
   - Model performance analysis

If any errors occur, explain what went wrong and what you tried to do.
"""
    ) 