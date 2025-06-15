import json
import os
from typing import Dict, Any
from smolagents import tool
from datasets import load_from_disk

@tool
def save_analysis_results(results: Dict[str, Any], output_path: str) -> str:
    """
    Saves analysis results to a JSON file.
    Args:
        results: The analysis results to save
        output_path: The path where to save the file
    Returns:
        The path where the file was saved
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    return f"Results saved to {output_path}"

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
def read_dataset(input_path: str) -> Dict[str, Any]:
    """
    Reads the user provided dataset
    Args:
        input_path: The path to the dataset
    Returns:
        The dataset
    """
    return load_from_disk(input_path)