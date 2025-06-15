from smolagents import CodeAgent
from smolagents import LiteLLMModel
from src.utils.file_tools import save_analysis_results, load_dataset, set_seed

def create_analysis_agent(model: LiteLLMModel) -> CodeAgent:
    """Create and configure the dataset analysis agent."""
    return CodeAgent(
        name="global_analysis",
        tools=[save_analysis_results, load_dataset, set_seed],
        model=model,
        additional_authorized_imports=[
            "time", "numpy", "pandas", "os", "datasets", "json", "matplotlib", "seaborn"
        ],
        description="""Goal: generate an exploratory analysis for the `datasets/diabetes-readmission` dataset and save it as JSON.

Action guidelines (do NOT echo):
1. set_seed(42) for reproducibility.
2. dataset_dict = load_dataset('datasets/diabetes-readmission')
3. Work on the **train** split unless stated otherwise: `df = dataset_dict['train'].to_pandas()`
4. Produce these insights (at minimum):
   - num_samples, num_features
   - list of features with dtype & % missing
   - class distribution of `readmitted`
   - basic stats for numeric cols (mean, std) and value counts for categoricals
   - correlation matrix for numeric cols
5. (Optional) create insightful plots and save them under `analysis_results/plots/` if matplotlib is available.
6. Build a python `dict` called `analysis_results` with:
   {
     "num_samples": <int>,
     "features": [ {"name": str, "dtype": str, "pct_missing": float} ],
     "class_distribution": <dict>,
     "correlations": <dict>,
     "dataset_paths": {
         "train": "datasets/diabetes-readmission/train",
         "test":  "datasets/diabetes-readmission/test",
         "base_path": "datasets/diabetes-readmission"
     }
   }
7. save_analysis_results(analysis_results, 'analysis_results/dataset_analysis.json')
8. Return: 'analysis_results/dataset_analysis.json'

If an error occurs, raise an Exception with a concise message so the manager can surface it.
"""
    ) 