from smolagents import CodeAgent
from smolagents import LiteLLMModel

def create_manager_agent(model: LiteLLMModel, managed_agents: list[CodeAgent]) -> CodeAgent:
    """Create and configure the manager agent."""
    return CodeAgent(
        name="manager",
        tools=[],
        model=model,
        managed_agents=managed_agents,
        additional_authorized_imports=["time", "numpy", "pandas", "os", "datasets"],
    ) 