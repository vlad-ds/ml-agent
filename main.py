from src.utils.model_setup import setup_model
from src.agents.analysis_agent import create_analysis_agent
from src.agents.modeling_agent import create_modeling_agent
from src.agents.manager_agent import create_manager_agent
import os

def main():
    # Initialize the model
    model = setup_model()

    # Create the agents
    analysis_agent = create_analysis_agent(model)
    modeling_agent = create_modeling_agent(model)

    # Create the manager agent with both agents
    manager_agent = create_manager_agent(model, [analysis_agent, modeling_agent])
    manager_agent.run("Analyze the datasets/diabetes-readmission dataset, then train and evaluate models on datasets/diabetes-readmission. Use AUC as the evaluation metric ")

if __name__ == "__main__":
    main()

