"""Compare SMACT validity check with our custom validity checks.

This script samples structures from LeMat-Bulk, evaluates them using both
SMACT validity and our custom validity metrics, and identifies differences
between the two approaches.

COMPARISON:
- SMACT: Charge neutrality based on oxidation states only
- Our checks: OverallValidityMetric = ALL of the following must pass:
  1. ChargeNeutralityMetric (more comprehensive than SMACT)
  2. MinimumInteratomicDistanceMetric (atomic overlaps)
  3. PhysicalPlausibilityMetric (density, lattice, format, symmetry)

The script runs multiple seeds for reliable statistics and saves:
1. CIF files where checks disagree
2. Detailed statistics for each seed
3. Aggregate statistics across all seeds
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
from datasets import load_dataset
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from smact.screening import smact_validity
from tqdm import tqdm

from lemat_genbench.metrics.validity_metrics import OverallValidityMetric
from lemat_genbench.utils.logging import logger

# Suppress warnings
warnings.filterwarnings("ignore")


def lematbulk_item_to_structure(item: dict) -> Structure:
    """Convert a LeMat-Bulk dataset item to a pymatgen Structure.
    
    Parameters
    ----------
    item : dict
        Dictionary containing structure information from LeMat-Bulk dataset.
        
    Returns
    -------
    Structure
        Pymatgen Structure object.
    """
    sites = item["species_at_sites"]
    coords = item["cartesian_site_positions"]
    cell = item["lattice_vectors"]

    structure = Structure(
        species=sites, coords=coords, lattice=cell, coords_are_cartesian=True
    )

    return structure


def check_smact_validity(structure: Structure) -> bool:
    """Check SMACT validity for a structure.
    
    Parameters
    ----------
    structure : Structure
        Pymatgen Structure object.
        
    Returns
    -------
    bool
        True if structure passes SMACT validity, False otherwise.
    """
    try:
        # Get composition formula without spaces
        formula = structure.composition.formula.replace(" ", "")
        return smact_validity(formula, oxidation_states_set="smact14")
    except Exception as e:
        logger.debug(f"SMACT validity check failed with error: {str(e)}")
        return False


def check_our_validity(
    structure: Structure,
    metric: OverallValidityMetric,
    compute_args: dict
) -> bool:
    """Check validity using our custom metrics.
    
    Parameters
    ----------
    structure : Structure
        Pymatgen Structure object.
    metric : OverallValidityMetric
        Our custom validity metric.
    compute_args : dict
        Arguments for the compute function.
        
    Returns
    -------
    bool
        True if structure passes our validity checks, False otherwise.
    """
    try:
        result = metric.compute_structure(structure, **compute_args)
        return result >= 0.999  # Valid if essentially 1.0
    except Exception as e:
        logger.debug(f"Our validity check failed with error: {str(e)}")
        return False


def save_structure_as_cif(structure: Structure, filepath: Path, structure_id: str):
    """Save a structure as a CIF file.
    
    Parameters
    ----------
    structure : Structure
        Pymatgen Structure object.
    filepath : Path
        Directory path to save the CIF file.
    structure_id : str
        Identifier for the structure (used in filename).
    """
    filepath.mkdir(parents=True, exist_ok=True)
    cif_path = filepath / f"structure_{structure_id}.cif"
    
    try:
        cif_writer = CifWriter(structure)
        cif_writer.write_file(str(cif_path))
    except Exception as e:
        logger.warning(f"Failed to write CIF for {structure_id}: {str(e)}")


def process_seed(
    seed: int,
    n_samples: int,
    dataset,
    output_dir: Path
) -> Dict:
    """Process a single seed: sample structures and compare validity checks.
    
    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    n_samples : int
        Number of structures to sample.
    dataset
        LeMat-Bulk dataset.
    output_dir : Path
        Base output directory.
        
    Returns
    -------
    dict
        Statistics for this seed.
    """
    # Set random seed
    np.random.seed(seed)
    
    # Create seed-specific directory
    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for different cases
    ours_valid_smact_invalid_dir = seed_dir / "ours_valid_smact_invalid"
    smact_valid_ours_invalid_dir = seed_dir / "smact_valid_ours_invalid"
    both_valid_dir = seed_dir / "both_valid"
    both_invalid_dir = seed_dir / "both_invalid"
    
    # Initialize our validity metric
    validity_metric = OverallValidityMetric(
        charge_tolerance=0.1,
        distance_scaling=0.5,
        min_atomic_density=0.00001,
        max_atomic_density=0.5,
        min_mass_density=0.01,
        max_mass_density=25.0,
        check_format=True,
        check_symmetry=True,
        n_jobs=1,
    )
    compute_args = validity_metric._get_compute_attributes()
    
    # Sample random indices
    dataset_size = len(dataset)
    indices = np.random.choice(dataset_size, size=n_samples, replace=False)
    
    # Statistics
    stats = {
        "seed": seed,
        "n_samples": n_samples,
        "ours_valid_smact_invalid": [],
        "smact_valid_ours_invalid": [],
        "both_valid": [],
        "both_invalid": [],
        "smact_error_count": 0,
        "ours_error_count": 0,
    }
    
    # Process each structure
    logger.info(f"Processing seed {seed}...")
    for idx in tqdm(indices, desc=f"Seed {seed}"):
        structure_id = f"{seed}_{idx}"
        
        try:
            # Load structure
            item = dataset[int(idx)]
            structure = lematbulk_item_to_structure(item)
            
            # Check SMACT validity
            try:
                smact_valid = check_smact_validity(structure)
            except Exception as e:
                logger.debug(f"SMACT error for {structure_id}: {str(e)}")
                stats["smact_error_count"] += 1
                smact_valid = False
            
            # Check our validity
            try:
                ours_valid = check_our_validity(structure, validity_metric, compute_args)
            except Exception as e:
                logger.debug(f"Our validity error for {structure_id}: {str(e)}")
                stats["ours_error_count"] += 1
                ours_valid = False
            
            # Get composition for statistics
            composition = structure.composition.formula.replace(" ", "")
            structure_info = {
                "structure_id": structure_id,
                "dataset_index": int(idx),
                "composition": composition,
                "smact_valid": smact_valid,
                "ours_valid": ours_valid,
            }
            
            # Categorize and save CIF
            if ours_valid and not smact_valid:
                stats["ours_valid_smact_invalid"].append(structure_info)
                save_structure_as_cif(structure, ours_valid_smact_invalid_dir, structure_id)
            elif smact_valid and not ours_valid:
                stats["smact_valid_ours_invalid"].append(structure_info)
                save_structure_as_cif(structure, smact_valid_ours_invalid_dir, structure_id)
            elif smact_valid and ours_valid:
                stats["both_valid"].append(structure_info)
                save_structure_as_cif(structure, both_valid_dir, structure_id)
            else:
                stats["both_invalid"].append(structure_info)
                save_structure_as_cif(structure, both_invalid_dir, structure_id)
                
        except Exception as e:
            logger.warning(f"Failed to process structure {structure_id}: {str(e)}")
            continue
    
    # Calculate summary statistics
    stats["summary"] = {
        "ours_valid_smact_invalid_count": len(stats["ours_valid_smact_invalid"]),
        "smact_valid_ours_invalid_count": len(stats["smact_valid_ours_invalid"]),
        "both_valid_count": len(stats["both_valid"]),
        "both_invalid_count": len(stats["both_invalid"]),
        "total_processed": sum([
            len(stats["ours_valid_smact_invalid"]),
            len(stats["smact_valid_ours_invalid"]),
            len(stats["both_valid"]),
            len(stats["both_invalid"]),
        ]),
        "smact_valid_rate": (
            len(stats["smact_valid_ours_invalid"]) + len(stats["both_valid"])
        ) / n_samples,
        "ours_valid_rate": (
            len(stats["ours_valid_smact_invalid"]) + len(stats["both_valid"])
        ) / n_samples,
        "agreement_rate": (
            len(stats["both_valid"]) + len(stats["both_invalid"])
        ) / n_samples,
        "disagreement_rate": (
            len(stats["ours_valid_smact_invalid"]) + len(stats["smact_valid_ours_invalid"])
        ) / n_samples,
    }
    
    # Save seed statistics
    stats_file = seed_dir / "statistics.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Seed {seed} complete. Summary:")
    logger.info(f"  Ours valid, SMACT invalid: {stats['summary']['ours_valid_smact_invalid_count']}")
    logger.info(f"  SMACT valid, ours invalid: {stats['summary']['smact_valid_ours_invalid_count']}")
    logger.info(f"  Both valid: {stats['summary']['both_valid_count']}")
    logger.info(f"  Both invalid: {stats['summary']['both_invalid_count']}")
    logger.info(f"  Agreement rate: {stats['summary']['agreement_rate']:.2%}")
    
    return stats


def aggregate_statistics(all_stats: List[Dict], output_dir: Path):
    """Aggregate statistics across all seeds.
    
    Parameters
    ----------
    all_stats : list of dict
        Statistics from each seed.
    output_dir : Path
        Base output directory.
    """
    aggregate = {
        "n_seeds": len(all_stats),
        "n_samples_per_seed": all_stats[0]["n_samples"],
        "total_samples": sum(s["n_samples"] for s in all_stats),
        "seeds": [s["seed"] for s in all_stats],
    }
    
    # Aggregate counts
    for key in ["ours_valid_smact_invalid", "smact_valid_ours_invalid", "both_valid", "both_invalid"]:
        counts = [len(s[key]) for s in all_stats]
        aggregate[f"{key}_counts"] = counts
        aggregate[f"{key}_mean"] = np.mean(counts)
        aggregate[f"{key}_std"] = np.std(counts)
        aggregate[f"{key}_total"] = sum(counts)
    
    # Aggregate rates
    for key in ["smact_valid_rate", "ours_valid_rate", "agreement_rate", "disagreement_rate"]:
        rates = [s["summary"][key] for s in all_stats]
        aggregate[f"{key}_mean"] = np.mean(rates)
        aggregate[f"{key}_std"] = np.std(rates)
        aggregate[f"{key}_values"] = rates
    
    # Error counts
    aggregate["smact_error_counts"] = [s["smact_error_count"] for s in all_stats]
    aggregate["ours_error_counts"] = [s["ours_error_count"] for s in all_stats]
    
    # Save aggregate statistics
    aggregate_file = output_dir / "aggregate_statistics.json"
    with open(aggregate_file, "w") as f:
        json.dump(aggregate, f, indent=2)
    
    logger.info("\n" + "="*80)
    logger.info("AGGREGATE STATISTICS ACROSS ALL SEEDS")
    logger.info("="*80)
    logger.info(f"Total samples processed: {aggregate['total_samples']}")
    logger.info("\nOurs valid, SMACT invalid:")
    logger.info(f"  Mean: {aggregate['ours_valid_smact_invalid_mean']:.1f} ± {aggregate['ours_valid_smact_invalid_std']:.1f}")
    logger.info(f"  Total: {aggregate['ours_valid_smact_invalid_total']}")
    logger.info("\nSMACT valid, ours invalid:")
    logger.info(f"  Mean: {aggregate['smact_valid_ours_invalid_mean']:.1f} ± {aggregate['smact_valid_ours_invalid_std']:.1f}")
    logger.info(f"  Total: {aggregate['smact_valid_ours_invalid_total']}")
    logger.info("\nBoth valid:")
    logger.info(f"  Mean: {aggregate['both_valid_mean']:.1f} ± {aggregate['both_valid_std']:.1f}")
    logger.info(f"  Total: {aggregate['both_valid_total']}")
    logger.info("\nBoth invalid:")
    logger.info(f"  Mean: {aggregate['both_invalid_mean']:.1f} ± {aggregate['both_invalid_std']:.1f}")
    logger.info(f"  Total: {aggregate['both_invalid_total']}")
    logger.info("\nValidity rates:")
    logger.info(f"  SMACT: {aggregate['smact_valid_rate_mean']:.2%} ± {aggregate['smact_valid_rate_std']:.2%}")
    logger.info(f"  Ours:  {aggregate['ours_valid_rate_mean']:.2%} ± {aggregate['ours_valid_rate_std']:.2%}")
    logger.info(f"\nAgreement rate: {aggregate['agreement_rate_mean']:.2%} ± {aggregate['agreement_rate_std']:.2%}")
    logger.info(f"Disagreement rate: {aggregate['disagreement_rate_mean']:.2%} ± {aggregate['disagreement_rate_std']:.2%}")
    logger.info("="*80)


def main():
    """Main function to run the validity comparison.
    
    This compares:
    - SMACT validity (charge neutrality only)
    - OverallValidityMetric (charge + distance + physical plausibility)
    """
    # Configuration
    dataset_name = "Lematerial/LeMat-Bulk"
    dataset_config = "compatible_pbe"
    split = "train"
    n_samples = 1000
    seeds = [42, 123, 456, 789, 2024]  # 5 different seeds
    output_dir = Path(__file__).parent.parent / "validity_comparison_results"
    
    logger.info("="*80)
    logger.info("VALIDITY COMPARISON: SMACT vs OverallValidityMetric")
    logger.info("="*80)
    logger.info("SMACT: Charge neutrality based on oxidation states")
    logger.info("Ours:  OverallValidityMetric (Charge + Distance + Physical)")
    logger.info("="*80)
    
    logger.info("Loading LeMat-Bulk dataset...")
    dataset = load_dataset(
        dataset_name,
        name=dataset_config,
        split=split,
        streaming=False,
        trust_remote_code=True
    )
    logger.info(f"Dataset loaded. Total structures: {len(dataset)}")
    
    # Process each seed
    all_stats = []
    for seed in seeds:
        stats = process_seed(seed, n_samples, dataset, output_dir)
        all_stats.append(stats)
    
    # Aggregate results
    aggregate_statistics(all_stats, output_dir)
    
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("Done!")


if __name__ == "__main__":
    main()

