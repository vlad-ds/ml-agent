# ML Agent

🚧 **Hackathon Experiment - Work in Progress** 🚧

An autonomous multi-agent system for end-to-end machine learning workflows powered by **Claude Sonnet** and **smolagents**. This system uses specialized AI agents to automatically analyze datasets, research best practices, train models, and evaluate performance.

## Overview

ML Agent is a proof-of-concept system that demonstrates how multiple AI agents can collaborate to handle complex ML tasks autonomously. The system consists of four specialized agents:

- **Manager Agent**: Routes tasks and orchestrates the workflow
- **Analysis Agent**: Performs exploratory data analysis and generates insights
- **Context Agent**: Researches domain knowledge and ML best practices
- **Modeling Agent**: Trains and evaluates machine learning models

## Architecture

The system uses a hierarchical agent structure where the Manager Agent coordinates the work of specialized sub-agents based on the task requirements and current state.

```
Manager Agent
├── Analysis Agent (EDA & data insights)
├── Context Agent (domain research)
└── Modeling Agent (training & evaluation)
```

## Project Structure

```
.
├── agent_runs/           # Results from agent executions
├── analysis_results/     # Dataset analysis outputs
├── datasets/            # Dataset storage
├── src/
│   ├── agents/         # Agent definitions
│   │   ├── manager_agent.py     # Orchestrates workflow
│   │   ├── analysis_agent.py    # Dataset analysis
│   │   ├── context_agent.py     # Domain research
│   │   └── modeling_agent.py    # Model training
│   ├── tools/          # Agent wrapper functions
│   └── utils/          # Utility functions
├── main.py             # Main entry point
└── download_dataset.py # Dataset download script
```

## Features

- **Automated EDA**: Generates comprehensive dataset analysis including statistics, correlations, and visualizations
- **Domain Research**: Automatically researches best practices and state-of-the-art methods for the problem domain
- **Model Selection**: Intelligently selects appropriate algorithms based on data characteristics
- **Evaluation**: Uses proper cross-validation and reports multiple metrics
- **Reproducibility**: Sets seeds and saves all results for reproducible experiments

## Quick Start

1. **Setup Environment** (using uv - recommended):
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

2. **Download Dataset**:
```bash
python download_dataset.py
```

3. **Run the Agent System**:
```bash
python main.py
```

The system will automatically:
- Analyze the diabetes readmission dataset
- Research relevant ML approaches
- Train and evaluate models using AUC as the primary metric
- Save results and trained models

## Current Status

This is a hackathon experiment demonstrating autonomous ML workflows. The system currently works with the diabetes readmission prediction task but is designed to be extensible to other ML problems.

**What Works:**
- Multi-agent coordination and task routing
- Automated dataset analysis and visualization
- Domain-specific research via web search
- Model training with multiple algorithms (CatBoost, XGBoost, LightGBM, etc.)
- Proper evaluation with cross-validation

**Limitations:**
- Partly hardcoded for diabetes readmission dataset
- Limited error handling and recovery

## Dependencies

- smolagents (multi-agent framework)
- pandas, numpy (data manipulation)
- scikit-learn (ML algorithms)
- catboost, lightgbm, xgboost (gradient boosting)
- datasets (HuggingFace datasets)
- matplotlib, seaborn (visualization)

## Example Output

The system generates:
- `analysis_results/dataset_analysis.json` - Comprehensive EDA results
- `analysis_results/context_research.json` - Domain research findings
- `agent_runs/*/` - Training scripts, models, and evaluation results
- Model files and feature importance rankings

See `logs_example.txt` for a complete example of an agent run showing the multi-agent coordination in action.

## Future Directions

- Support for custom datasets and problem types
- Automated feature engineering
- Hyperparameter optimization
- Model ensembling and stacking
- **HuggingFace MCP** to source papers and models. Note: Our models are already finding HuggingFace models!
- **MCP to upload model to HuggingFace**
- MLOps integration (experiment tracking, model deployment)
- Interactive web interface

---

*This is an experimental project exploring autonomous AI agents for machine learning workflows.*
