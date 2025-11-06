"""Test novelty metric with MP-20 structures.

This script tests the novelty metric by:
1. Loading structures from MP-20 train and test splits
2. Running novelty metric with MP-20 as reference
3. Verifying expected behavior:
   - Train structures should be NOT novel (in reference)
   - Test structures should be NOT novel if using full mp_20.csv
   - Test structures should be NOVEL if using only train.csv as reference

Usage:
    # Test with full MP-20 as reference (train structures = not novel)
    uv run scripts/test_mp20_novelty.py
    
    # Test with only train as reference (test structures = novel)
    uv run scripts/test_mp20_novelty.py --reference-split train
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lemat_genbench.data.mp20_loader import load_mp20_dataset, mp20_item_to_structure
from lemat_genbench.metrics.novelty_metric import NoveltyMetric
from lemat_genbench.utils.logging import logger


def test_novelty_with_mp20(
    reference_csv: str = "mp-20-data/mp_20.csv",
    n_train_samples: int = 10,
    n_test_samples: int = 10,
    fingerprint_method: str = "bawl"
):
    """Test novelty metric with MP-20 structures.
    
    Parameters
    ----------
    reference_csv : str
        Path to CSV file to use as reference
    n_train_samples : int
        Number of train structures to test
    n_test_samples : int
        Number of test structures to test
    fingerprint_method : str
        Fingerprinting method to use
    """
    
    print("=" * 80)
    print("🧪 Testing MP-20 Novelty Integration")
    print("=" * 80)
    print()
    
    # Load train and test splits
    logger.info("Loading MP-20 train split...")
    train_df = load_mp20_dataset("mp-20-data/train.csv")
    logger.info(f"✅ Loaded {len(train_df)} train structures")
    
    logger.info("Loading MP-20 test split...")
    test_df = load_mp20_dataset("mp-20-data/test.csv")
    logger.info(f"✅ Loaded {len(test_df)} test structures")
    print()
    
    # Sample structures
    train_samples = train_df.head(n_train_samples)
    test_samples = test_df.head(n_test_samples)
    
    # Convert to pymatgen structures
    logger.info(f"Converting {n_train_samples} train structures...")
    train_structures = []
    train_ids = []
    for idx, row in train_samples.iterrows():
        try:
            structure = mp20_item_to_structure(row)
            train_structures.append(structure)
            train_ids.append(row['material_id'])
        except Exception as e:
            logger.warning(f"Failed to convert train structure {idx}: {e}")
    
    logger.info(f"Converting {n_test_samples} test structures...")
    test_structures = []
    test_ids = []
    for idx, row in test_samples.iterrows():
        try:
            structure = mp20_item_to_structure(row)
            test_structures.append(structure)
            test_ids.append(row['material_id'])
        except Exception as e:
            logger.warning(f"Failed to convert test structure {idx}: {e}")
    
    logger.info(f"✅ Converted {len(train_structures)} train + {len(test_structures)} test structures")
    print()
    
    # Initialize novelty metric
    logger.info(f"Initializing NoveltyMetric with reference: {reference_csv}")
    logger.info(f"Fingerprint method: {fingerprint_method}")
    metric = NoveltyMetric(
        reference_dataset_path=reference_csv,
        fingerprint_method=fingerprint_method,
        cache_reference=True,
    )
    print()
    
    # Test train structures
    print("=" * 80)
    print("📊 Testing TRAIN Structures (should be NOT NOVEL)")
    print("=" * 80)
    print()
    
    logger.info("Computing novelty for train structures...")
    train_result = metric.compute(train_structures)
    
    print("\n📈 Train Results:")
    print(f"  Novelty Rate: {train_result.value:.2%}")
    print(f"  Novel Count: {train_result.metrics.get('novel_structures_count', 0)}")
    print(f"  Total: {train_result.metrics.get('total_structures_evaluated', 0)}")
    print()
    
    # Show individual results
    print("Individual Train Structures:")
    for i, (material_id, structure) in enumerate(zip(train_ids, train_structures)):
        is_novel = train_result.metrics.get('individual_results', [None] * len(train_structures))[i]
        status = "✅ Novel" if is_novel == 1.0 else "❌ Not Novel" if is_novel == 0.0 else "⚠️  Error"
        formula = structure.composition.reduced_formula
        print(f"  {material_id:15} {formula:15} {status}")
    print()
    
    # Expected: All train structures should be NOT novel (0%)
    if reference_csv == "mp-20-data/mp_20.csv" or "train" in reference_csv:
        if train_result.value == 0.0:
            print("✅ PASS: All train structures correctly identified as NOT NOVEL")
        else:
            print(f"⚠️  WARNING: Expected 0% novelty for train structures, got {train_result.value:.2%}")
            print("   This might indicate an issue with fingerprinting or reference loading.")
    
    print()
    
    # Test test structures
    print("=" * 80)
    print("📊 Testing TEST Structures")
    print("=" * 80)
    print()
    
    logger.info("Computing novelty for test structures...")
    test_result = metric.compute(test_structures)
    
    print("\n📈 Test Results:")
    print(f"  Novelty Rate: {test_result.value:.2%}")
    print(f"  Novel Count: {test_result.metrics.get('novel_structures_count', 0)}")
    print(f"  Total: {test_result.metrics.get('total_structures_evaluated', 0)}")
    print()
    
    # Show individual results
    print("Individual Test Structures:")
    for i, (material_id, structure) in enumerate(zip(test_ids, test_structures)):
        is_novel = test_result.metrics.get('individual_results', [None] * len(test_structures))[i]
        status = "✅ Novel" if is_novel == 1.0 else "❌ Not Novel" if is_novel == 0.0 else "⚠️  Error"
        formula = structure.composition.reduced_formula
        print(f"  {material_id:15} {formula:15} {status}")
    print()
    
    # Expected behavior depends on reference
    if reference_csv == "mp-20-data/mp_20.csv":
        # Using full dataset as reference - test should also be not novel
        if test_result.value == 0.0:
            print("✅ PASS: All test structures correctly identified as NOT NOVEL")
            print("   (Test structures are in the full MP-20 reference dataset)")
        else:
            print(f"⚠️  WARNING: Expected 0% novelty for test structures, got {test_result.value:.2%}")
            print("   This might indicate an issue with reference loading.")
    elif "train" in reference_csv:
        # Using only train as reference - test should be novel
        if test_result.value > 0.0:
            print(f"✅ PASS: Test structures are NOVEL ({test_result.value:.2%})")
            print("   (Test structures are not in the train-only reference dataset)")
        else:
            print(f"⚠️  WARNING: Expected >0% novelty for test structures, got {test_result.value:.2%}")
            print("   Test structures should be novel when using train-only reference.")
    
    print()
    
    # Summary
    print("=" * 80)
    print("📋 Summary")
    print("=" * 80)
    print()
    print(f"Reference Dataset: {reference_csv}")
    print(f"Fingerprint Method: {fingerprint_method}")
    print()
    print(f"Train Structures Tested: {len(train_structures)}")
    print(f"Train Novelty Rate: {train_result.value:.2%}")
    print()
    print(f"Test Structures Tested: {len(test_structures)}")
    print(f"Test Novelty Rate: {test_result.value:.2%}")
    print()
    
    # Final verdict
    if reference_csv == "mp-20-data/mp_20.csv":
        if train_result.value == 0.0 and test_result.value == 0.0:
            print("🎉 SUCCESS: Novelty metric is working correctly!")
            print("   ✅ Train structures: Not novel (as expected)")
            print("   ✅ Test structures: Not novel (as expected, in full reference)")
            return True
        else:
            print("❌ FAILURE: Unexpected novelty results!")
            print(f"   Train novelty: {train_result.value:.2%} (expected 0%)")
            print(f"   Test novelty: {test_result.value:.2%} (expected 0%)")
            return False
    elif "train" in reference_csv:
        if train_result.value == 0.0 and test_result.value > 0.0:
            print("🎉 SUCCESS: Novelty metric is working correctly!")
            print("   ✅ Train structures: Not novel (as expected)")
            print("   ✅ Test structures: Novel (as expected, not in train reference)")
            return True
        else:
            print("❌ FAILURE: Unexpected novelty results!")
            print(f"   Train novelty: {train_result.value:.2%} (expected 0%)")
            print(f"   Test novelty: {test_result.value:.2%} (expected >0%)")
            return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test novelty metric with MP-20 structures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with full MP-20 (train + val + test) as reference
  uv run scripts/test_mp20_novelty.py
  
  # Test with only train as reference (test should be novel)
  uv run scripts/test_mp20_novelty.py --reference-split train
  
  # Test with more samples
  uv run scripts/test_mp20_novelty.py --n-train 20 --n-test 20
  
  # Test with structure-matcher (slower but more accurate)
  uv run scripts/test_mp20_novelty.py --fingerprint structure-matcher --n-train 5 --n-test 5
        """
    )
    parser.add_argument(
        "--reference-split",
        type=str,
        default="full",
        choices=["full", "train"],
        help="Which split to use as reference (default: full = mp_20.csv)"
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=10,
        help="Number of train structures to test (default: 10)"
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=10,
        help="Number of test structures to test (default: 10)"
    )
    parser.add_argument(
        "--fingerprint",
        type=str,
        default="bawl",
        choices=["bawl", "short-bawl", "structure-matcher"],
        help="Fingerprinting method (default: bawl)"
    )
    args = parser.parse_args()
    
    # Determine reference CSV
    if args.reference_split == "train":
        reference_csv = "mp-20-data/train.csv"
    else:
        reference_csv = "mp-20-data/mp_20.csv"
    
    # Check if files exist
    for csv_file in ["mp-20-data/train.csv", "mp-20-data/test.csv", reference_csv]:
        if not Path(csv_file).exists():
            print(f"❌ ERROR: Required file not found: {csv_file}")
            print(f"   Please ensure MP-20 dataset is available in mp-20-data/")
            sys.exit(1)
    
    # Run test
    try:
        success = test_novelty_with_mp20(
            reference_csv=reference_csv,
            n_train_samples=args.n_train,
            n_test_samples=args.n_test,
            fingerprint_method=args.fingerprint
        )
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

