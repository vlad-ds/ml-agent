import os
from datasets import load_dataset

# Check if dataset already exists
dataset_path = "datasets/diabetes-readmission"

if os.path.exists(dataset_path):
    print(f"Dataset already exists at {dataset_path}")
else:
    print("Downloading diabetes readmission dataset...")
    # Download the diabetes readmission dataset
    dataset = load_dataset("imodels/diabetes-readmission")
    
    # Save to datasets folder
    dataset.save_to_disk(dataset_path)
    print(f"Dataset downloaded and saved to {dataset_path}")