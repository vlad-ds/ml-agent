from src.utils.model_setup import setup_model
from src.agents.manager_agent import create_manager_agent

def main():
    # Initialize the model
    model = setup_model()

    manager_agent = create_manager_agent(model)
    manager_agent.run(
        "Train and evaluate models on datasets/diabetes-readmission. Use AUC as the evaluation metric"
    )

if __name__ == "__main__":
    main()

