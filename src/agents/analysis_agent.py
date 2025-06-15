from smolagents import CodeAgent
from smolagents import LiteLLMModel

def create_analysis_agent(model: LiteLLMModel) -> CodeAgent:
    """Create and configure the dataset analysis agent."""
    return CodeAgent(
        name="global_analysis",
        tools=[],
        model=model,
        additional_authorized_imports=["time", "numpy", "pandas", "os", "datasets"],
        description="""You are a dataset analysis expert. Your task is to analyze the diabetes-readmission dataset:
1. Load the dataset using:
   dataset = load_from_disk('datasets/diabetes-readmission')
2. Analyze the dataset structure and provide a comprehensive summary including:
   - The number of examples (using len(dataset))
   - The features/columns available (using dataset.features)
   - A brief description of what the dataset seems to be about
3. If any errors occur, explain what went wrong and what you tried to do
"""
    ) 