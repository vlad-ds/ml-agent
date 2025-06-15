from smolagents import CodeAgent
from smolagents import LiteLLMModel
import os
import json

def create_manager_agent(model: LiteLLMModel, managed_agents: list[CodeAgent]) -> CodeAgent:
    """Create and configure the manager agent."""
    return CodeAgent(
        name="manager",
        tools=[],
        model=model,
        managed_agents=managed_agents,
        additional_authorized_imports=["time", "numpy", "pandas", "os", "datasets", "json"],
        description="""You are a manager agent coordinating the analysis and modeling of the diabetes-readmission dataset.

Your task is to:
1. First, check if analysis results exist:
   if not os.path.exists('analysis_results/dataset_analysis.json'):
       - Ask the global_analysis agent to analyze the dataset with this exact prompt:
         "Please analyze the diabetes-readmission dataset and save the results to analysis_results/dataset_analysis.json"
       - Wait for the analysis to complete
       - Verify that analysis_results/dataset_analysis.json was created
   else:
       - Skip the analysis step and proceed to modeling

2. Then, ask the modeling agent to train and evaluate models:
   - Prompt: "Please read the analysis results from analysis_results/dataset_analysis.json and train the most appropriate model"
   - Wait for the modeling to complete
   - Return the modeling results

Important rules:
- Run the analysis agent only once
- Wait for each step to complete before moving to the next
- If any step fails, explain what went wrong and stop
- Follow these steps EXACTLY as written
- Do not modify or skip any steps

Return format:
- Analysis status (new or existing)
- Modeling results with:
  - Model choice and reasoning
  - Performance metrics
  - Feature importance
  - Test set results
"""
    ) 