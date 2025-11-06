"""Pre-compute reference data for MP-20 dataset.

This script generates all necessary reference data files for fast novelty checking:
1. Composition matrix (mp20_compositions.npz) - for fast element-based filtering
2. BAWL fingerprints (mp20_bawl_fingerprints.pkl) - for fast fingerprint-based novelty
3. Dataset statistics (mp20_stats.json) - for analysis and reporting

Usage:
    # Compute all reference data
    uv run scripts/precompute_mp20_reference.py
    
    # Specify custom paths
    uv run scripts/precompute_mp20_reference.py --csv-path mp-20-data/train.csv --output-dir data
    
    # Skip fingerprints (if only need compositions)
    uv run scripts/precompute_mp20_reference.py --skip-fingerprints
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tqdm import tqdm

from lemat_genbench.data.mp20_loader import (
    compute_mp20_statistics,
    get_mp20_compositions,
    load_mp20_dataset,
    mp20_item_to_structure,
)
from lemat_genbench.fingerprinting.utils import get_fingerprint, get_fingerprinter
from lemat_genbench.utils.logging import logger


def compute_fingerprints(csv_path: str, output_path: Path) -> int:
    """Compute and cache BAWL fingerprints for MP-20 dataset.
    
    Parameters
    ----------
    csv_path : str
        Path to MP-20 CSV file
    output_path : Path
        Path to save fingerprints
        
    Returns
    -------
    int
        Number of unique fingerprints computed
    """
    logger.info("Computing BAWL fingerprints for MP-20 dataset...")
    logger.info("This may take 10-30 minutes depending on dataset size...")
    
    # Load dataset
    df = load_mp20_dataset(csv_path)
    
    # Initialize fingerprinter
    fingerprinter = get_fingerprinter("bawl")
    
    # Compute fingerprints
    fingerprints = set()
    failed_count = 0
    
    start_time = time.time()
    
    for idx, row in tqdm(
        df.iterrows(), 
        total=len(df), 
        desc="Computing fingerprints",
        unit="structure"
    ):
        try:
            structure = mp20_item_to_structure(row)
            fp = get_fingerprint(structure, fingerprinter)
            
            if fp:
                fingerprints.add(fp)
            else:
                failed_count += 1
                
        except Exception as e:
            failed_count += 1
            if failed_count <= 10:  # Only show first 10 errors
                logger.warning(
                    f"Failed to process {row.get('material_id', idx)}: {e}"
                )
    
    elapsed = time.time() - start_time
    
    # Save fingerprints
    with open(output_path, 'wb') as f:
        pickle.dump(fingerprints, f)
    
    logger.info(f"✅ Computed {len(fingerprints):,} unique fingerprints")
    logger.info(f"   Failed: {failed_count} structures")
    logger.info(f"   Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    logger.info(f"   Saved to: {output_path}")
    
    return len(fingerprints)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute reference data for MP-20 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compute all reference data using default paths
  uv run scripts/precompute_mp20_reference.py
  
  # Use custom CSV path
  uv run scripts/precompute_mp20_reference.py --csv-path mp-20-data/train.csv
  
  # Skip fingerprint computation (faster, but structure-matcher won't benefit)
  uv run scripts/precompute_mp20_reference.py --skip-fingerprints
  
  # Custom output directory
  uv run scripts/precompute_mp20_reference.py --output-dir custom_data/
        """
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="mp-20-data/mp_20.csv",
        help="Path to MP-20 CSV file (default: mp-20-data/mp_20.csv)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for reference files (default: data)"
    )
    parser.add_argument(
        "--skip-fingerprints",
        action="store_true",
        help="Skip fingerprint computation (faster but less complete)"
    )
    args = parser.parse_args()
    
    # Validate input
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        logger.error("Please provide a valid path to MP-20 CSV file")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("🚀 Pre-computing MP-20 Reference Data")
    print("=" * 70)
    print(f"Input CSV:    {csv_path}")
    print(f"Output dir:   {output_dir}")
    print(f"Skip fingerprints: {args.skip_fingerprints}")
    print("=" * 70)
    print()
    
    # Track what was computed
    computed_files = []
    
    # 1. Compute composition matrix
    print("1️⃣  Computing composition matrix...")
    print("-" * 70)
    try:
        compositions = get_mp20_compositions(str(csv_path))
        comp_file = output_dir / "mp20_compositions.npz"
        computed_files.append(("Composition matrix", comp_file, compositions.shape[0]))
        print(f"✅ Composition matrix: {compositions.shape}")
        print()
    except Exception as e:
        logger.error(f"Failed to compute compositions: {e}")
        sys.exit(1)
    
    # 2. Compute fingerprints (optional)
    n_fingerprints = 0
    if not args.skip_fingerprints:
        print("2️⃣  Computing BAWL fingerprints...")
        print("-" * 70)
        try:
            fingerprint_file = output_dir / "mp20_bawl_fingerprints.pkl"
            n_fingerprints = compute_fingerprints(str(csv_path), fingerprint_file)
            computed_files.append(("BAWL fingerprints", fingerprint_file, n_fingerprints))
            print()
        except Exception as e:
            logger.error(f"Failed to compute fingerprints: {e}")
            logger.warning("Continuing without fingerprints...")
    else:
        print("2️⃣  Skipping fingerprint computation (--skip-fingerprints)")
        print()
    
    # 3. Compute dataset statistics
    print("3️⃣  Computing dataset statistics...")
    print("-" * 70)
    try:
        stats = compute_mp20_statistics(str(csv_path))
        
        # Add fingerprint info if computed
        if n_fingerprints > 0:
            stats['n_unique_fingerprints'] = n_fingerprints
            stats['fingerprint_method'] = 'bawl'
        
        # Save statistics
        stats_file = output_dir / "mp20_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        computed_files.append(("Dataset statistics", stats_file, stats['n_structures']))
        
        logger.info(f"✅ Dataset statistics:")
        logger.info(f"   Total structures: {stats['n_structures']:,}")
        logger.info(f"   Unique elements: {stats['n_unique_elements']}")
        logger.info(f"   Splits: {stats.get('splits', 'N/A')}")
        if 'e_above_hull_stats' in stats and stats['e_above_hull_stats']:
            logger.info(f"   E_above_hull mean: {stats['e_above_hull_stats']['mean']:.4f} eV")
        logger.info(f"   Saved to: {stats_file}")
        print()
    except Exception as e:
        logger.error(f"Failed to compute statistics: {e}")
        logger.warning("Continuing without statistics...")
    
    # Summary
    print("=" * 70)
    print("🎉 MP-20 Reference Data Computation Complete!")
    print("=" * 70)
    print()
    print("Generated files:")
    print("-" * 70)
    for name, filepath, count in computed_files:
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"  {name:25} {str(filepath):40}")
        print(f"  {'':25} Size: {size_mb:.2f} MB, Entries: {count:,}")
        print()
    
    print("=" * 70)
    print()
    print("✅ Reference data is ready!")
    print()
    print("You can now run novelty/SUN benchmarks with MP-20 as reference:")
    print("  uv run scripts/run_benchmarks_ssh.py \\")
    print("    --config comprehensive_multi_mlip_hull \\")
    print("    --cifs your_structures/ \\")
    print("    --name your_experiment")
    print()


if __name__ == "__main__":
    main()

