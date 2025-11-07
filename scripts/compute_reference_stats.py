#!/usr/bin/env python3
"""
Compute reference statistics (mu, sigma) for Fréchet distance from
LeMat-GenBench-embeddings dataset.

This script loads the full dataset and computes mean and covariance matrices
for each model's embeddings using direct computation with memory-efficient
processing for large models.

Usage:
    uv run scripts/compute_reference_stats.py --cache-dir ./data --models mace orb uma

Requirements:
    uv add datasets psutil
"""

import argparse
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lemat_genbench.utils.distribution_utils import compute_reference_stats_direct


def main():
    parser = argparse.ArgumentParser(
        description="Compute reference statistics for Fréchet distance computation"
    )
    parser.add_argument(
        "--dataset-name",
        default="LeMaterial/LeMat-GenBench-embeddings-mp20",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mace", "orb", "uma"],
        choices=["mace", "orb", "uma"],
        help="Models to compute statistics for"
    )
    parser.add_argument(
        "--cache-dir",
        default="./data",
        help="Directory to save computed statistics (default: ./data)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Computing Reference Statistics for Fréchet Distance")
    print("=" * 60)
    print(f"Dataset: {args.dataset_name}")
    print(f"Models: {args.models}")
    print(f"Cache directory: {args.cache_dir}")
    print("Method: Direct computation with memory-efficient processing")
    print()

    # Ensure cache directory exists
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    # Start computation
    start_time = time.time()

    try:
        stats = compute_reference_stats_direct(
            dataset_name=args.dataset_name,
            model_names=args.models,
            cache_dir=args.cache_dir
        )

        end_time = time.time()
        duration = end_time - start_time

        print("\n" + "=" * 60)
        print("COMPUTATION COMPLETED!")
        print("=" * 60)
        print(f"Total time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print()

        # Print summary
        for model_name, model_stats in stats.items():
            mu_shape = model_stats["mu"].shape
            sigma_shape = model_stats["sigma"].shape
            print(f"{model_name.upper()}:")
            print(f"  Mean shape: {mu_shape}")
            print(f"  Covariance shape: {sigma_shape}")
            print(f"  Memory usage: ~{(mu_shape[0] * 4 + sigma_shape[0] * sigma_shape[1] * 4) / 1024:.1f} KB")
            print()

        print(f"Statistics saved to: {args.cache_dir}")
        print("\nFiles created:")
        cache_path = Path(args.cache_dir)
        for model_name in args.models:
            mu_file = cache_path / f"{model_name}_mu.npy"
            sigma_file = cache_path / f"{model_name}_sigma.npy"
            if mu_file.exists() and sigma_file.exists():
                mu_size = mu_file.stat().st_size / 1024
                sigma_size = sigma_file.stat().st_size / 1024
                print(f"  {model_name}_mu.npy ({mu_size:.1f} KB)")
                print(f"  {model_name}_sigma.npy ({sigma_size:.1f} KB)")

        metadata_file = cache_path / "metadata.json"
        if metadata_file.exists():
            print(f"  metadata.json ({metadata_file.stat().st_size} bytes)")

    except Exception as e:
        print(f"\nERROR: {e}")
        print("Make sure you have the required libraries installed:")
        print("  uv add datasets psutil")
        sys.exit(1)


if __name__ == "__main__":
    main()