from smolagents import CodeAgent
from smolagents import LiteLLMModel
import os
from src.utils.file_tools import save_analysis_results

def create_analysis_agent(model: LiteLLMModel) -> CodeAgent:
    """Create and configure the dataset analysis agent."""
    return CodeAgent(
        name="global_analysis",
        tools=[save_analysis_results],
        model=model,
        additional_authorized_imports=["time", "numpy", "pandas", "os", "datasets", "json"],
        description="""You are a dataset analysis expert. Your task is to analyze the diabetes-readmission dataset:
1. Load the dataset using:
   dataset = load_from_disk('datasets/diabetes-readmission')
2. Analyze the dataset structure and provide a comprehensive summary including:
   - The number of examples (using len(dataset))
   - The features/columns available (using dataset.features)
   - A brief description of what the dataset seems to be about
   - Types of features (categorical vs numerical)
   - Missing values analysis
   - Class distribution
   - Feature distributions
   - Correlations between features
3. Save the analysis results to a JSON file:
   analysis_results = {
       'num_samples': len(dataset),
       'features': list(dataset.features.keys()),
       'feature_types': {col: str(dataset[col].dtype) for col in dataset.features},
       'missing_values': {col: dataset[col].isna().sum() for col in dataset.features},
       'class_distribution': dataset['readmitted'].value_counts().to_dict(),
       'description': 'Your analysis description here',
       'dataset_paths': {
           'train': os.path.join('datasets', 'diabetes-readmission', 'train'),
           'test': os.path.join('datasets', 'diabetes-readmission', 'test'),
           'base_path': os.path.join('datasets', 'diabetes-readmission')
       }
   }
   output_path = 'analysis_results/dataset_analysis.json'
   save_analysis_results(analysis_results, output_path)
4. Return the path to the saved file: 'analysis_results/dataset_analysis.json'
5. If any errors occur, explain what went wrong and what you tried to do
"""
    ) 