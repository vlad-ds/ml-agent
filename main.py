from smolagents import CodeAgent, InferenceClientModel
from smolagents import LiteLLMModel
import os
from dotenv import load_dotenv
from datasets import load_from_disk

# Load environment variables
load_dotenv()

# Initialize the model with Claude
model = LiteLLMModel(model_id="claude-3-opus-latest")

# Create the global analysis agent with detailed instructions
global_analysis_agent = CodeAgent(
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

# Create the manager agent that will coordinate the analysis
manager_agent = CodeAgent(
    name="manager",
    tools=[],
    model=model,
    managed_agents=[global_analysis_agent],
    additional_authorized_imports=["time", "numpy", "pandas", "os", "datasets"],
)

# Let the manager ask the global analysis agent to analyze the datasets
result = manager_agent.run("Ask the global_analysis agent to analyze the diabetes-readmission dataset")
print(result)

