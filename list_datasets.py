#!/usr/bin/env python3
"""
Utility script to list available datasets
"""

import json
import os
from pathlib import Path
from config import get_data_paths


def list_datasets():
    """List all available datasets"""
    datasets_dir = Path("datasets")
    if not datasets_dir.exists():
        print("No datasets directory found.")
        return []

    datasets = []
    for dataset_dir in datasets_dir.iterdir():
        if dataset_dir.is_dir():
            dataset_name = dataset_dir.name
            data_paths = get_data_paths(dataset_name)
            golden_labels_path = data_paths["golden_labels"]

            if os.path.exists(golden_labels_path):
                try:
                    with open(golden_labels_path, "r") as f:
                        data = json.load(f)
                        samples = data.get("samples", [])
                        metadata = data.get("metadata", {})

                    datasets.append(
                        {
                            "name": dataset_name,
                            "samples": len(samples),
                            "created": metadata.get("collection_date", "Unknown"),
                            "path": golden_labels_path,
                        }
                    )
                except Exception as e:
                    print(f"❌ Error reading dataset '{dataset_name}': {e}")

    return datasets


def main():
    print("📊 Available Datasets:")
    print("=" * 50)

    datasets = list_datasets()

    if not datasets:
        print("No datasets found.")
        print("\nTo create a new dataset, run:")
        print("python main.py --mode collect --dataset <name> --samples <count>")
        return

    for dataset in sorted(datasets, key=lambda x: x["name"]):
        print(f"Dataset: {dataset['name']}")
        print(f"  Samples: {dataset['samples']}")
        print(f"  Created: {dataset['created']}")
        print(f"  Path: {dataset['path']}")
        print()

    print("To use a dataset, run:")
    print("python main.py --mode benchmark --dataset <name>")
    print("python main.py --mode agent --dataset <name>")


if __name__ == "__main__":
    main()
