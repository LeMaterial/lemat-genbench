#!/usr/bin/env python3
"""
Download and save the above-hull datasets locally.

This script downloads the private datasets required for energy above hull calculations
and saves them as pickle files in the data/ folder for local access.

Usage:
    python scripts/download_above_hull_datasets.py
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from datasets import load_dataset

from lemat_genbench.utils.logging import logger


def download_and_save_datasets():
    """Download above-hull datasets and save them locally."""
    
    # Define paths
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    dataset_path = data_dir / "lematbulk_above_hull_dataset.pkl"
    matrix_path = data_dir / "lematbulk_above_hull_composition_matrix.pkl"
    
    try:
        # Download and save the main dataset
        logger.info("Downloading LeMaterial-Above-Hull-dataset...")
        dataset = load_dataset("Entalpic/LeMaterial-Above-Hull-dataset")
        dataset_df = pd.DataFrame(dataset["dataset"])
        dataset_df.to_pickle(dataset_path)
        logger.info(f"✅ Saved dataset to {dataset_path}")
        
        # Download and save the composition matrix
        logger.info("Downloading LeMaterial-Above-Hull-composition_matrix...")
        composition_matrix = load_dataset("Entalpic/LeMaterial-Above-Hull-composition_matrix")
        composition_matrix_data = composition_matrix["composition_matrix"]
        composition_df = composition_matrix_data.to_pandas()
        composition_df.drop("Unnamed: 0", axis=1, inplace=True)
        composition_array = composition_df.to_numpy()
        pd.DataFrame(composition_array).to_pickle(matrix_path)
        logger.info(f"✅ Saved composition matrix to {matrix_path}")
        
        logger.info("🎉 All datasets downloaded and saved successfully!")
        logger.info("Energy above hull calculations will now work without HuggingFace Hub access.")
        
    except Exception as e:
        logger.error(f"❌ Failed to download datasets: {e}")
        logger.error("You may need to request access to these private datasets from the dataset owners.")
        sys.exit(1)


if __name__ == "__main__":
    download_and_save_datasets() 