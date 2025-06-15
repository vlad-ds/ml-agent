from smolagents import LiteLLMModel
from dotenv import load_dotenv

def setup_model() -> LiteLLMModel:
    """Initialize and configure the LLM model."""
    load_dotenv()
    return LiteLLMModel(model_id="claude-3-opus-latest") 