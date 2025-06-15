import json
import os
from typing import Dict, Any, List
from smolagents import tool
from datasets import DatasetDict, load_from_disk

@tool
def save_analysis_results(results: Dict[str, Any], output_path: str) -> str:
    """Save dataset analysis to disk in *analysis_results* directory.

    Args:
        results: Dictionary holding the analysis output.
        output_path: Desired file path. If the path is not inside
            the ``analysis_results`` directory, it will be redirected
            there automatically, preserving the original file name.

    Returns:
        A message with the absolute path to the saved file.
    """
    # Redirect to analysis_results directory when needed
    abs_path = os.path.abspath(output_path)
    cwd = os.getcwd()

    # Convert absolute path to relative from cwd for inspection
    rel_parts = os.path.relpath(abs_path, start=cwd).split(os.sep)
    if "analysis_results" not in rel_parts:
        filename = os.path.basename(abs_path)
        abs_path = os.path.join(cwd, "analysis_results", filename)

    # Ensure standard filename
    dirname, fname = os.path.split(abs_path)
    if fname != "dataset_analysis.json":
        abs_path = os.path.join(dirname, "dataset_analysis.json")

    # Create directory tree
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)

    # Ensure all keys are strings and non-serialisable objects are stringified
    def _stringify(o):
        if isinstance(o, dict):
            return {str(k): _stringify(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_stringify(x) for x in o]
        return o

    serialisable = _stringify(results)

    with open(abs_path, "w") as f:
        json.dump(serialisable, f, indent=2, default=str)

    return f"Results saved to {abs_path}"

@tool
def read_analysis_results(input_path: str) -> Dict[str, Any]:
    """
    Reads analysis results from a JSON file.
    Args:
        input_path: The path to the JSON file
    Returns:
        The analysis results
    """
    with open(input_path, 'r') as f:
        return json.load(f) 

@tool
def analysis_present(path: str) -> bool:
    """Check if a given file path exists on disk.

    Args:
        path: Absolute or relative path to a file on the local filesystem.

    Returns:
        True if the file exists, otherwise False.
    """
    return os.path.exists(path)

@tool
def list_files(directory: str) -> List[str]:
    """List files in a directory.

    Args:
        directory: The directory path whose files you want to list.

    Returns:
        A list containing the file and sub-directory names found in `directory`.
    """
    return os.listdir(directory)

@tool
def load_dataset(path: str) -> DatasetDict:
    """Load a HuggingFace DatasetDict that was previously saved to disk.

    Args:
        path: Directory where the dataset was stored via `datasets.DatasetDict.save_to_disk()`.

    Returns:
        The loaded `DatasetDict` instance.
    """
    return load_from_disk(path)

@tool
def set_seed(seed: int = 42) -> str:
    """Seed common random number generators for reproducible experiments.

    Args:
        seed: Integer value to use as the seed. Defaults to 42.

    Returns:
        Confirmation message with the applied seed.
    """
    import random
    import numpy as np
    import pandas as pd
    try:
        from sklearn import utils as sk_utils
        sk_utils.check_random_state(seed)
    except ImportError:
        pass
    random.seed(seed)
    np.random.seed(seed)
    pd.options.mode.chained_assignment = None
    return f"Seed set to {seed}"

@tool
def read_json(path: str) -> Dict[str, Any]:
    """Read a JSON file and return its contents.

    Args:
        path: Path to the JSON file to read.

    Returns:
        Parsed JSON as a Python object (usually dict or list).
    """
    with open(path, "r") as fp:
        return json.load(fp)
    
@tool
def save_model(model: Any, path: str) -> str:
    """Save a model to disk.

    Args:
        model: The model to save.
        path: Path to the directory where the model will be saved.
    """
    model.save(path)
    return f"Model saved to {path}"