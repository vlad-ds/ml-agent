from smolagents import tool

@tool
def run_global_analysis(message: str) -> str:
    """Run the global analysis agent and return its output.

    Args:
        message: The textual instruction or query from the user that should
            be forwarded to the analysis agent.

    Returns:
        The raw string (usually JSON or a path) produced by the analysis agent.
    """
    # Local import to avoid circular dependencies at module import time.
    from src.utils.model_setup import setup_model
    from src.agents.analysis_agent import create_analysis_agent

    model = setup_model()
    analysis_agent = create_analysis_agent(model)
    return analysis_agent.run(message)


@tool
def run_modeling(message: str) -> str:
    """Run the modeling agent and return its output.

    Args:
        message: The textual instruction or query from the user that should
            be forwarded to the modeling agent.

    Returns:
        The modeling report (as a JSON-serialisable string) produced by the
        modeling agent.

    Note:
        The modeling agent assumes that a dataset analysis JSON already
        exists at ``analysis_results/dataset_analysis.json``. If it does not,
        callers should make sure to run ``run_global_analysis`` first.
    """
    from src.utils.model_setup import setup_model
    from src.agents.modeling_agent import create_modeling_agent

    model = setup_model()
    modeling_agent = create_modeling_agent(model)
    return modeling_agent.run(message) 