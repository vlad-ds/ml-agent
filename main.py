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

    # First, check if analysis results exist
    if not os.path.exists('analysis_results/dataset_analysis.json'):
        print("Running analysis agent...")
        analysis_result = manager_agent.run("Analyze the diabetes-readmission dataset")
        print("Dataset Analysis Results:")
        print(analysis_result)
        print("\n" + "="*50 + "\n")
    else:
        print("Analysis results already exist, skipping analysis step...")

    # Then, train and evaluate models
    print("Running modeling agent...")
    modeling_result = manager_agent.run("Train and evaluate models on the diabetes-readmission dataset using AUC as the evaluation metric")
    print("Modeling Results:")
    print(modeling_result)

if __name__ == "__main__":
    main()

