from smolagents import CodeAgent
from smolagents import LiteLLMModel
from smolagents import WebSearchTool

def create_context_agent(model: LiteLLMModel) -> CodeAgent:
    """Create and configure the context search agent."""
    return CodeAgent(
        name="context_search",
        tools=[WebSearchTool()],
        model=model,
        additional_authorized_imports=[
            "json", "re", "urllib.parse"
        ],
        description="""Goal: search for general domain knowledge, academic papers, and research about machine learning problem types and methodologies - NOT specific datasets.

Action guidelines (do NOT echo):
1. Analyze the user's request to identify the GENERAL problem domain (e.g., "medical readmission prediction", "classification", "healthcare analytics")
2. Formulate search queries focused on:
   - General approaches to this type of problem
   - Academic papers on similar problem domains
   - State-of-the-art methods for this problem type
   - Best practices and common challenges
   - Comparative studies of different approaches
3. Use WebSearchTool to search for relevant content
4. Focus searches on:
   - arXiv papers about the problem domain
   - Google Scholar results for methodological approaches
   - Academic conferences and journals
   - Technical blogs from reputable sources
5. Synthesize findings into a structured summary including:
   - Common approaches to this type of problem
   - Key papers and their methodological contributions
   - State-of-the-art techniques for this domain
   - Best practices and common pitfalls
   - Evaluation metrics typically used
6. Return findings as a structured text summary

IMPORTANT: Search for GENERAL domain knowledge, not specific datasets. Focus on how this type of problem is typically approached in the literature.

If an error occurs, raise an Exception with a concise message so the manager can surface it.
"""
    )