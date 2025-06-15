# ML Agent

A machine learning agent project for dataset analysis.

## Project Structure

```
.
├── datasets/              # Dataset storage
├── src/                   # Source code
│   ├── agents/           # Agent definitions
│   │   ├── analysis_agent.py
│   │   └── manager_agent.py
│   └── utils/            # Utility functions
│       └── model_setup.py
├── main.py               # Main entry point
├── download_dataset.py   # Dataset download script
└── requirements.txt      # Project dependencies
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download the dataset:
```bash
python download_dataset.py
```

3. Run the analysis:
```bash
python main.py
```
