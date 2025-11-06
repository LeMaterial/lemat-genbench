#!/usr/bin/env python3
"""Quick script to check validity of AFLOW structures."""

import logging
from pathlib import Path
from typing import Dict, List

from pymatgen.core import Structure

from lemat_genbench.benchmarks.validity_benchmark import ValidityBenchmark

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_structures_from_cifs(cif_dir: Path) -> List[Structure]:
    """Load all structures from CIF files in a directory."""
    structures = []
    cif_files = sorted(list(cif_dir.glob("*.cif")))
    
    logger.info(f"Found {len(cif_files)} CIF files in {cif_dir}")
    
    for cif_file in cif_files:
        try:
            structure = Structure.from_file(str(cif_file))
            structures.append(structure)
        except Exception as e:
            logger.warning(f"Failed to load {cif_file.name}: {e}")
    
    logger.info(f"Successfully loaded {len(structures)} structures")
    return structures


def main():
    # Setup paths
    aflow_dir = Path("/Users/siddharthbetala/Desktop/lemat-genbench/baseline_data/aflow")
    
    if not aflow_dir.exists():
        logger.error(f"AFLOW directory not found: {aflow_dir}")
        return
    
    logger.info("="*80)
    logger.info("AFLOW Structures - Validity Check")
    logger.info("="*80)
    
    # Load structures
    structures = load_structures_from_cifs(aflow_dir)
    
    if not structures:
        logger.error("No structures loaded!")
        return
    
    # Create validity benchmark
    logger.info("\nInitializing ValidityBenchmark...")
    benchmark = ValidityBenchmark(
        charge_tolerance=0.1,
        distance_scaling=0.5,
        min_atomic_density=0.00001,
        max_atomic_density=0.5,
        min_mass_density=0.01,
        max_mass_density=25.0,
        check_format=True,
        check_symmetry=True,
    )
    
    # Run benchmark
    logger.info(f"\nRunning validity checks on {len(structures)} AFLOW structures...")
    logger.info("This may take a few minutes...\n")
    
    benchmark_result = benchmark.evaluate(structures)
    
    # Display results
    logger.info("\n" + "="*80)
    logger.info("VALIDITY RESULTS - AFLOW Structures")
    logger.info("="*80)
    
    # Access the final scores
    final_scores = benchmark_result.final_scores
    
    logger.info(f"\nTotal Structures: {final_scores.get('total_structures', 0)}")
    logger.info("")
    logger.info("Individual Validity Checks:")
    logger.info("-" * 80)
    
    # Charge Neutrality
    charge_ratio = final_scores.get('charge_neutrality_ratio', 0)
    charge_count = final_scores.get('charge_neutrality_count', 0)
    total = final_scores.get('total_structures', 0)
    logger.info(f"  Charge Neutrality:      {charge_ratio*100:5.2f}% ({charge_count}/{total})")
    
    # Interatomic Distance
    dist_ratio = final_scores.get('interatomic_distance_ratio', 0)
    dist_count = final_scores.get('interatomic_distance_count', 0)
    logger.info(f"  Interatomic Distance:   {dist_ratio*100:5.2f}% ({dist_count}/{total})")
    
    # Physical Plausibility
    phys_ratio = final_scores.get('physical_plausibility_ratio', 0)
    phys_count = final_scores.get('physical_plausibility_count', 0)
    logger.info(f"  Physical Plausibility:  {phys_ratio*100:5.2f}% ({phys_count}/{total})")
    
    logger.info("")
    logger.info("="*80)
    
    # Overall Validity
    overall_ratio = final_scores.get('overall_validity_ratio', 0)
    overall_count = final_scores.get('overall_validity_count', 0)
    invalid_count = final_scores.get('any_invalid_count', 0)
    
    logger.info(f"OVERALL VALIDITY: {overall_ratio*100:.2f}%")
    logger.info(f"  Valid:   {overall_count}/{total}")
    logger.info(f"  Invalid: {invalid_count}/{total}")
    logger.info("="*80)
    
    # Also print to stdout for visibility
    print("\n" + "="*80)
    print("AFLOW VALIDITY CHECK RESULTS")
    print("="*80)
    print(f"Total Structures: {total}")
    print(f"Charge Neutrality:     {charge_ratio*100:5.2f}% ({charge_count}/{total})")
    print(f"Interatomic Distance:  {dist_ratio*100:5.2f}% ({dist_count}/{total})")
    print(f"Physical Plausibility: {phys_ratio*100:5.2f}% ({phys_count}/{total})")
    print(f"\nOVERALL VALIDITY: {overall_ratio*100:.2f}% ({overall_count}/{total})")
    print("="*80)
    
    return benchmark_result


if __name__ == "__main__":
    main()

