from smolagents import CodeAgent
from smolagents import LiteLLMModel

def create_manager_agent(model: LiteLLMModel, managed_agents: list[CodeAgent]) -> CodeAgent:
    """Create and configure the manager agent."""
    return CodeAgent(
        name="manager",
        tools=[],
        model=model,
        managed_agents=managed_agents,
        additional_authorized_imports=["time", "numpy", "pandas", "os", "datasets", "json"],
        description="""You are a manager agent that delegates work to specialized sub-agents.

Action protocol (never quote these steps back, just act):
1. If the user request is an *analysis* task, forward the request verbatim to the `global_analysis` agent using `await_agent` / `run` and return its output.
2. If the request is a *modeling* or *training* task, forward it to the `modeling` agent and return its output.
3. Always wait for the delegated agent to finish before returning a result.
4. If a delegated agent raises an error, surface the error and stop.

Return the result in this JSON schema:
{
  "delegate": "<agent_name>",
  "result": <sub_agent_output>
}
"""
    ) 