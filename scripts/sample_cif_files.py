#!/usr/bin/env python
"""Script to sample a specified number of CIF files from a source directory.

This script copies a random sample of CIF files from a source directory to a new
destination directory. Useful for creating smaller test sets or subsamples of
large CIF file collections.

Usage:
    python sample_cif_files.py <source_dir> <dest_dir> --n_samples 2500
    python sample_cif_files.py /path/to/cifs /path/to/output --n_samples 1000 --random_seed 42
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import List, Optional


def get_cif_files(directory: Path) -> List[Path]:
    """Get all CIF files from a directory.
    
    Parameters
    ----------
    directory : Path
        Directory to search for CIF files.
        
    Returns
    -------
    List[Path]
        List of paths to CIF files.
    """
    # Support both .cif and .CIF extensions
    cif_files = list(directory.glob("*.cif")) + list(directory.glob("*.CIF"))
    return sorted(cif_files)  # Sort for reproducibility


def sample_and_copy_cifs(
    source_dir: Path,
    dest_dir: Path,
    n_samples: int,
    random_seed: Optional[int] = None,
    overwrite: bool = False,
) -> None:
    """Sample and copy CIF files from source to destination directory.
    
    Parameters
    ----------
    source_dir : Path
        Source directory containing CIF files.
    dest_dir : Path
        Destination directory where sampled CIFs will be copied.
    n_samples : int
        Number of CIF files to sample.
    random_seed : int, optional
        Random seed for reproducibility. If None, sampling is not reproducible.
    overwrite : bool, default=False
        Whether to overwrite the destination directory if it exists.
        
    Raises
    ------
    ValueError
        If source directory doesn't exist, contains no CIF files, or n_samples
        is greater than available CIF files.
    FileExistsError
        If destination directory exists and overwrite is False.
    """
    # Validate source directory
    if not source_dir.exists():
        raise ValueError(f"Source directory does not exist: {source_dir}")
    
    if not source_dir.is_dir():
        raise ValueError(f"Source path is not a directory: {source_dir}")
    
    # Get all CIF files
    print(f"Scanning for CIF files in: {source_dir}")
    cif_files = get_cif_files(source_dir)
    
    if not cif_files:
        raise ValueError(f"No CIF files found in source directory: {source_dir}")
    
    print(f"Found {len(cif_files)} CIF files in source directory")
    
    # Validate n_samples
    if n_samples > len(cif_files):
        raise ValueError(
            f"Requested {n_samples} samples but only {len(cif_files)} CIF files available. "
            f"Cannot sample more files than available."
        )
    
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got: {n_samples}")
    
    # Handle destination directory
    if dest_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Destination directory already exists: {dest_dir}. "
                f"Use --overwrite to replace it."
            )
        print(f"Removing existing destination directory: {dest_dir}")
        shutil.rmtree(dest_dir)
    
    # Create destination directory
    print(f"Creating destination directory: {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample CIF files
    if random_seed is not None:
        random.seed(random_seed)
        print(f"Using random seed: {random_seed}")
    
    print(f"Sampling {n_samples} CIF files...")
    sampled_files = random.sample(cif_files, n_samples)
    
    # Copy sampled files
    print(f"Copying files to destination...")
    for i, source_file in enumerate(sampled_files, 1):
        dest_file = dest_dir / source_file.name
        shutil.copy2(source_file, dest_file)
        
        # Progress update every 100 files
        if i % 100 == 0:
            print(f"  Copied {i}/{n_samples} files...")
    
    print(f"\n✓ Successfully copied {n_samples} CIF files to: {dest_dir}")
    print(f"  Source: {source_dir}")
    print(f"  Destination: {dest_dir}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Sample a specified number of CIF files from a source directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sample 2500 CIF files
  python sample_cif_files.py /path/to/cifs /path/to/output --n_samples 2500
  
  # Sample with a random seed for reproducibility
  python sample_cif_files.py ./cifs ./cifs_sample --n_samples 1000 --random_seed 42
  
  # Overwrite existing destination directory
  python sample_cif_files.py ./cifs ./cifs_sample --n_samples 500 --overwrite
        """,
    )
    
    parser.add_argument(
        "source_dir",
        type=Path,
        help="Source directory containing CIF files",
    )
    
    parser.add_argument(
        "dest_dir",
        type=Path,
        help="Destination directory where sampled CIF files will be copied",
    )
    
    parser.add_argument(
        "--n_samples",
        type=int,
        default=2500,
        help="Number of CIF files to sample (default: 2500)",
    )
    
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling (default: None)",
    )
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination directory if it exists",
    )
    
    args = parser.parse_args()
    
    try:
        sample_and_copy_cifs(
            source_dir=args.source_dir,
            dest_dir=args.dest_dir,
            n_samples=args.n_samples,
            random_seed=args.random_seed,
            overwrite=args.overwrite,
        )
    except (ValueError, FileExistsError) as e:
        print(f"\n❌ Error: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

