from src.utils.model_setup import setup_model
from src.agents.analysis_agent import create_analysis_agent
from src.agents.manager_agent import create_manager_agent

def main():
    # Initialize the model
    model = setup_model()

    # Create the analysis agent
    analysis_agent = create_analysis_agent(model)

    # Create the manager agent
    manager_agent = create_manager_agent(model, [analysis_agent])

    # Run the analysis
    result = manager_agent.run("Ask the global_analysis agent to analyze the diabetes-readmission dataset")
    print(result)

if __name__ == "__main__":
    main()

