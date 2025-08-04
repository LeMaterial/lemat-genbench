#!/usr/bin/env python3
"""
Create a 15K sample from the full MMD values pickle file for efficient loading.

This script:
1. Loads the full lematbulk_mmd_values.pkl (5M+ rows)
2. Uses pre-computed sample indices from lematbulk_mmd_sample_indices_15k.npy
3. Creates a smaller lematbulk_mmd_values_15k.pkl with just 15K samples
4. Preserves reproducibility while dramatically reducing file size and load time

Usage:
    python scripts/create_mmd_15k_sample.py

Result:
    Creates data/lematbulk_mmd_values_15k.pkl (352KB vs 122MB original)
"""

import pickle
from pathlib import Path

import numpy as np


def create_15k_sample():
    """Create 15K sample pickle file from full MMD values."""
    
    # File paths
    full_pickle_path = "data/lematbulk_mmd_values.pkl"
    sample_indices_path = "data/lematbulk_mmd_sample_indices_15k.npy"
    output_pickle_path = "data/lematbulk_mmd_values_15k.pkl"
    
    print("🔄 Creating 15K MMD sample pickle file...")
    
    # Load sample indices
    print(f"📂 Loading sample indices from {sample_indices_path}")
    try:
        sample_indices = np.load(sample_indices_path)
        print(f"✅ Loaded {len(sample_indices)} sample indices")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Sample indices file not found: {sample_indices_path}. "
            f"This file should be included in the data directory."
        )
    
    # Load full MMD values
    print(f"📂 Loading full MMD values from {full_pickle_path}")
    try:
        with open(full_pickle_path, "rb") as f:
            full_mmd_data = pickle.load(f)
        print(f"✅ Loaded full MMD data with properties: {list(full_mmd_data.keys())}")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Full MMD values file not found: {full_pickle_path}. "
            f"Please ensure this file exists in the data directory."
        )
    
    # Create sampled data dictionary
    sampled_mmd_data = {}
    
    for property_name, property_values in full_mmd_data.items():
        print(f"🔢 Sampling property: {property_name}")
        print(f"   Original size: {len(property_values):,}")
        
        # Convert to numpy array if not already
        if not isinstance(property_values, np.ndarray):
            property_values = np.array(property_values)
        
        # Sample using the pre-computed indices
        sampled_values = property_values[sample_indices]
        sampled_mmd_data[property_name] = sampled_values
        
        print(f"   Sampled size: {len(sampled_values):,}")
        print(f"   Sample shape: {sampled_values.shape}")
        print(f"   Data type: {sampled_values.dtype}")
        
    # Save the sampled data
    print(f"💾 Saving sampled data to {output_pickle_path}")
    with open(output_pickle_path, "wb") as f:
        pickle.dump(sampled_mmd_data, f)
    
    # Report file sizes
    original_size = Path(full_pickle_path).stat().st_size / (1024 * 1024)  # MB
    new_size = Path(output_pickle_path).stat().st_size / (1024 * 1024)  # MB
    
    print("📊 Summary:")
    print(f"   Original file: {original_size:.1f} MB")
    print(f"   Sampled file: {new_size:.1f} MB")
    print(f"   Size reduction: {100 * (1 - new_size/original_size):.1f}%")
    print(f"   Sample ratio: {len(sample_indices):,} / {len(next(iter(full_mmd_data.values()))):,}")
    
    print("✅ Successfully created 15K MMD sample pickle file!")
    print(f"📁 Output: {output_pickle_path}")


if __name__ == "__main__":
    create_15k_sample()