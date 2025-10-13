#!/usr/bin/env python3
"""
Single-MLIP benchmark runner for before/after relaxation comparison.

This script is specifically designed for evaluating structures before and after relaxation
with a single MLIP. Unlike the comprehensive benchmark, this:
1. Uses only ONE MLIP (no ensembling)
2. Evaluates against that MLIP's specific hull
3. Is optimized for before/after relaxation comparisons
4. ALWAYS runs validity benchmark first (mandatory)

Typical workflow:
1. Run this script on original structures (before relaxation)
2. Run relax_structures.py to relax with chosen MLIP
3. Run this script again on relaxed structures (after relaxation)
4. Compare metrics to see relaxation effects

Usage:
    uv run scripts/run_benchmarks_single_mlip.py --cifs path/to/cifs.txt --mlip orb --config single_mlip --name before_relax
    uv run scripts/run_benchmarks_single_mlip.py --csv path/to/structures.csv --mlip mace --config single_mlip --name after_relax
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import embedding utilities
from embedding_utils import save_embeddings_from_structures

# Import structure loading functions from run_benchmarks
from run_benchmarks import (
    cleanup_after_benchmark,
    cleanup_after_preprocessor,
    clear_memory,
    load_cif_files,
    load_structures_from_wycoff_csv,
    log_memory_usage,
)

from lemat_genbench.benchmarks.distribution_benchmark import DistributionBenchmark
from lemat_genbench.benchmarks.diversity_benchmark import DiversityBenchmark
from lemat_genbench.benchmarks.hhi_benchmark import HHIBenchmark
from lemat_genbench.benchmarks.multi_mlip_stability_benchmark import (
    StabilityBenchmark as MultiMLIPStabilityBenchmark,
)
from lemat_genbench.benchmarks.novelty_benchmark import NoveltyBenchmark
from lemat_genbench.benchmarks.sun_benchmark import SUNBenchmark
from lemat_genbench.benchmarks.uniqueness_benchmark import UniquenessBenchmark
from lemat_genbench.preprocess.distribution_preprocess import DistributionPreprocessor
from lemat_genbench.preprocess.fingerprint_preprocess import FingerprintPreprocessor
from lemat_genbench.preprocess.multi_mlip_preprocess import (
    MultiMLIPStabilityPreprocessor,
)
from lemat_genbench.preprocess.validity_preprocess import ValidityPreprocessor
from lemat_genbench.utils.logging import logger


def get_mlip_hull_type(mlip_name: str) -> str:
    """Get the hull type for a specific MLIP.
    
    Parameters
    ----------
    mlip_name : str
        Name of the MLIP ("orb", "mace", "uma")
        
    Returns
    -------
    str
        Hull type identifier for the MLIP
    """
    hull_types = {
        "orb": "orb_conserv_inf",
        "mace": "mace_mp",
        "uma": "uma",
    }
    
    if mlip_name not in hull_types:
        raise ValueError(f"Unknown MLIP: {mlip_name}. Available: {list(hull_types.keys())}")
    
    return hull_types[mlip_name]


def get_mlip_config(mlip_name: str, device: torch.device) -> Dict[str, Any]:
    """Get configuration for a specific MLIP.
    
    Parameters
    ----------
    mlip_name : str
        Name of the MLIP ("orb", "mace", "uma")
    device : torch.device
        Device to run MLIP on
        
    Returns
    -------
    Dict[str, Any]
        MLIP configuration dictionary
    """
    configs = {
        "orb": {
            "model_type": "orb_v3_conservative_inf_omat",
            "device": device,
            "hull_type": "orb_conserv_inf",
        },
        "mace": {
            "model_type": "mp",
            "device": device,
            "hull_type": "mace_mp",
        },
        "uma": {
            "task": "omat",
            "device": device,
            "hull_type": "uma",
        },
    }
    
    if mlip_name not in configs:
        raise ValueError(f"Unknown MLIP: {mlip_name}. Available: {list(configs.keys())}")
    
    return configs[mlip_name]


def load_benchmark_config(config_name: str) -> Dict[str, Any]:
    """Load benchmark configuration from YAML file."""
    config_dir = Path(__file__).parent.parent / "src" / "config"
    config_path = config_dir / f"{config_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def create_preprocessor_config(
    benchmark_families: List[str],
    fingerprint_method: str = "structure-matcher",
    generate_embedding_plots: bool = False,
) -> Dict[str, Any]:
    """Create preprocessor configuration based on required benchmark families.

    Note: validity preprocessing is ALWAYS included regardless of families.

    Parameters
    ----------
    benchmark_families : List[str]
        List of benchmark families to run
    fingerprint_method : str, default="structure-matcher"
        Fingerprinting method to use. Determines if fingerprint preprocessing is needed.
    generate_embedding_plots : bool, default=False
        Whether to generate embedding plots. If True, enables embeddings preprocessing.
    """
    config = {
        "validity": True,  # ALWAYS run validity preprocessing
        "distribution": False,
        "stability": False,
        "embeddings": False,
        "fingerprint": False,
    }

    # Determine which preprocessors are needed
    for family in benchmark_families:
        if family in ["distribution", "jsdistance", "mmd", "frechet"]:
            config["distribution"] = True
        if family in ["stability", "sun"]:
            config["stability"] = True
        if family in ["frechet", "distribution"]:
            config["embeddings"] = True
        # Original benchmarks need fingerprint preprocessing (unless using structure matcher)
        if family in ["novelty", "uniqueness", "sun"]:
            # Only run fingerprint preprocessor for BAWL/short-BAWL methods
            if fingerprint_method.lower() not in ["structure-matcher"]:
                config["fingerprint"] = True

    # Enable embeddings preprocessing if embedding plots are requested
    if generate_embedding_plots:
        config["embeddings"] = True

    return config


def run_validity_preprocessing_and_filtering(
    structures, config: Dict[str, Any], monitor_memory: bool = False
):
    """Run validity preprocessing and generate benchmark result, then filter to valid structures only.

    Returns
    -------
    tuple
        (validity_benchmark_result, valid_structures, validity_filtering_metadata)
    """
    # Log initial memory usage
    log_memory_usage("before validity processing", force_log=monitor_memory)

    n_total_structures = len(structures)
    logger.info(
        f"🔍 Starting MANDATORY validity processing for {n_total_structures} structures..."
    )

    # Run validity preprocessor on ALL structures
    logger.info("🔍 Running MANDATORY validity preprocessor on ALL structures...")
    start_time = time.time()

    validity_settings = config.get("validity_settings", {})
    charge_tolerance = validity_settings.get("charge_tolerance", 0.1)
    distance_scaling = validity_settings.get("distance_scaling", 0.5)
    min_atomic_density = validity_settings.get("min_atomic_density", 0.00001)
    max_atomic_density = validity_settings.get("max_atomic_density", 0.5)
    min_mass_density = validity_settings.get("min_mass_density", 0.01)
    max_mass_density = validity_settings.get("max_mass_density", 25.0)
    check_format = validity_settings.get("check_format", True)
    check_symmetry = validity_settings.get("check_symmetry", True)

    validity_preprocessor = ValidityPreprocessor(
        charge_tolerance=charge_tolerance,
        distance_scaling_factor=distance_scaling,
        plausibility_min_atomic_density=min_atomic_density,
        plausibility_max_atomic_density=max_atomic_density,
        plausibility_min_mass_density=min_mass_density,
        plausibility_max_mass_density=max_mass_density,
        plausibility_check_format=check_format,
        plausibility_check_symmetry=check_symmetry,
    )

    # Create source IDs for tracking
    structure_sources = [f"structure_{i}" for i in range(len(structures))]
    validity_preprocessor_result = validity_preprocessor.run(
        structures, structure_sources=structure_sources
    )
    processed_structures = validity_preprocessor_result.processed_structures

    # Generate benchmark result from preprocessor data
    validity_benchmark_result = validity_preprocessor.generate_benchmark_result(
        validity_preprocessor_result
    )

    elapsed_time = time.time() - start_time
    logger.info(
        f"✅ MANDATORY validity processing complete for {n_total_structures} structures in {elapsed_time:.1f}s"
    )

    # Clean up after validity processing
    cleanup_after_preprocessor("validity", monitor_memory)

    # Filter to only valid structures
    logger.info("🔍 Filtering to valid structures only...")

    valid_structures = []
    valid_structure_ids = []
    valid_structure_sources = []

    for structure in processed_structures:
        is_valid = structure.properties.get("overall_valid", False)
        if is_valid:
            valid_structures.append(structure)
            valid_structure_ids.append(
                structure.properties.get("structure_id", "unknown")
            )
            valid_structure_sources.append(
                structure.properties.get("original_source", "unknown")
            )

    n_valid_structures = len(valid_structures)
    n_invalid_structures = n_total_structures - n_valid_structures

    # Log filtering results
    logger.info(
        f"✅ Filtering complete: {n_valid_structures} valid structures out of {n_total_structures} total"
    )
    logger.info(f"📊 Valid: {n_valid_structures}, Invalid: {n_invalid_structures}")

    if n_valid_structures == 0:
        logger.warning(
            "⚠️  No valid structures found! All subsequent benchmarks will be skipped."
        )

    # Create filtering metadata
    validity_filtering_metadata = {
        "total_input_structures": n_total_structures,
        "valid_structures": n_valid_structures,
        "invalid_structures": n_invalid_structures,
        "validity_rate": n_valid_structures / n_total_structures
        if n_total_structures > 0
        else 0.0,
        "valid_structure_ids": valid_structure_ids,
        "valid_structure_sources": valid_structure_sources,
    }

    # Log final memory usage
    log_memory_usage("after validity filtering", force_log=monitor_memory)

    return validity_benchmark_result, valid_structures, validity_filtering_metadata


def run_remaining_preprocessors(
    valid_structures,
    preprocessor_config: Dict[str, Any],
    config: Dict[str, Any],
    mlip_name: str,
    run_name: str,
    monitor_memory: bool = False,
    generate_embedding_plots: bool = False,
):
    """Run remaining preprocessors on valid structures using single MLIP only.

    Note: validity preprocessing is already complete.
    """
    processed_structures = valid_structures
    preprocessor_results = {}

    if len(valid_structures) == 0:
        logger.warning(
            "⚠️  No valid structures to preprocess. Skipping remaining preprocessors."
        )
        return processed_structures, preprocessor_results

    # Log initial memory usage
    log_memory_usage("before remaining preprocessing", force_log=monitor_memory)

    # Fingerprint preprocessor (for BAWL/short-BAWL methods only)
    if preprocessor_config["fingerprint"]:
        logger.info(
            f"Running fingerprint preprocessor on {len(processed_structures)} valid structures..."
        )
        start_time = time.time()

        fingerprint_method = config.get("fingerprint_method", "structure-matcher")
        fingerprint_preprocessor = FingerprintPreprocessor(
            fingerprint_method=fingerprint_method
        )
        fingerprint_result = fingerprint_preprocessor(processed_structures)
        processed_structures = fingerprint_result.processed_structures
        preprocessor_results["fingerprint"] = fingerprint_result
        elapsed_time = time.time() - start_time
        logger.info(
            f"✅ Fingerprint preprocessing complete for {len(processed_structures)} valid structures in {elapsed_time:.1f}s"
        )

        # Clean up after fingerprint preprocessor
        cleanup_after_preprocessor("fingerprint", monitor_memory)

    # Distribution preprocessor (for MMD, JSDistance)
    if preprocessor_config["distribution"]:
        logger.info(
            f"Running distribution preprocessor on {len(processed_structures)} valid structures..."
        )
        start_time = time.time()
        dist_preprocessor = DistributionPreprocessor()
        dist_result = dist_preprocessor(processed_structures)
        processed_structures = dist_result.processed_structures
        preprocessor_results["distribution"] = dist_result
        elapsed_time = time.time() - start_time
        logger.info(
            f"✅ Distribution preprocessing complete for {len(processed_structures)} valid structures in {elapsed_time:.1f}s"
        )

        # Clean up after distribution preprocessor
        cleanup_after_preprocessor("distribution", monitor_memory)

    # Single-MLIP preprocessor (for stability, embeddings)
    if preprocessor_config["stability"] or preprocessor_config["embeddings"]:
        logger.info(
            f"Running Single-MLIP preprocessor ({mlip_name.upper()}) on {len(processed_structures)} valid structures..."
        )
        start_time = time.time()

        # Configure single MLIP with hull-specific settings
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # Get MLIP configuration
        mlip_config = get_mlip_config(mlip_name, device)

        # Determine what to extract based on requirements
        extract_embeddings = preprocessor_config["embeddings"]
        relax_structures = preprocessor_config["stability"]

        # Show progress for MLIP model loading
        logger.info(f"🔥 Initializing {mlip_name.upper()} model (this may take 1-2 minutes)...")

        mlip_preprocessor = MultiMLIPStabilityPreprocessor(
            mlip_names=[mlip_name],  # Single MLIP only
            mlip_configs={mlip_name: mlip_config},
            relax_structures=relax_structures,
            relaxation_config={"fmax": 0.02, "steps": 50},
            calculate_formation_energy=relax_structures,
            calculate_energy_above_hull=relax_structures,
            extract_embeddings=extract_embeddings,
            timeout=300,
        )

        # Add progress bar for structure processing
        logger.info(
            f"🔥 Processing {len(processed_structures)} valid structures with {mlip_name.upper()} model..."
        )
        mlip_result = mlip_preprocessor(processed_structures)
        processed_structures = mlip_result.processed_structures
        preprocessor_results["single_mlip"] = mlip_result
        elapsed_time = time.time() - start_time
        logger.info(
            f"✅ Single-MLIP preprocessing complete for {len(processed_structures)} valid structures in {elapsed_time:.1f}s"
        )

        # Save embeddings if they were extracted
        if extract_embeddings and processed_structures:
            save_embeddings_from_structures(
                processed_structures,
                config,
                run_name,
                generate_embedding_plots,
                logger=logger,
            )

        # Clean up after MLIP preprocessor
        cleanup_after_preprocessor(f"single_mlip_{mlip_name}", monitor_memory)

    # Log final memory usage
    log_memory_usage("after remaining preprocessing")

    return processed_structures, preprocessor_results


def run_remaining_benchmarks(
    valid_structures,
    benchmark_families: List[str],
    config: Dict[str, Any],
    mlip_name: str,
    monitor_memory: bool = False,
):
    """Run remaining benchmark families on valid structures using single MLIP only.

    Note: validity benchmark is already complete.
    """
    results = {}

    if len(valid_structures) == 0:
        logger.warning(
            "⚠️  No valid structures to benchmark. Skipping remaining benchmarks."
        )
        return results

    # Log initial memory usage
    log_memory_usage("before remaining benchmarks", force_log=monitor_memory)

    # Filter out validity from families since it's already done
    remaining_families = [f for f in benchmark_families if f != "validity"]

    if not remaining_families:
        logger.info("No remaining benchmarks to run.")
        return results

    # Add progress bar for benchmarks
    with tqdm(
        remaining_families, desc="Running remaining benchmarks", unit="benchmark"
    ) as pbar:
        for family in pbar:
            pbar.set_description(f"Running {family} benchmark")

            logger.info(
                f"Running {family} benchmark on {len(valid_structures)} valid structures..."
            )
            start_time = time.time()

            try:
                if family == "distribution":
                    benchmark = DistributionBenchmark(
                        mlips=[mlip_name],  # Single MLIP only
                        cache_dir=config.get("cache_dir", "./data"),
                        js_distributions_file=config.get(
                            "js_distributions_file",
                            "data/lematbulk_jsdistance_distributions.json",
                        ),
                        mmd_values_file=config.get(
                            "mmd_values_file", "data/lematbulk_mmd_values_15k.pkl"
                        ),
                    )

                elif family == "diversity":
                    benchmark = DiversityBenchmark()

                elif family == "novelty":
                    novelty_settings = config.get("novelty_settings", {})
                    benchmark = NoveltyBenchmark(
                        fingerprint_method=config.get(
                            "fingerprint_method", "structure-matcher"
                        ),
                        n_jobs=novelty_settings.get("n_jobs", 1),
                    )

                elif family == "uniqueness":
                    _ = config.get("uniqueness_settings", {})
                    benchmark = UniquenessBenchmark(
                        fingerprint_method=config.get(
                            "fingerprint_method", "structure-matcher"
                        ),
                        n_jobs=1,
                    )

                elif family == "hhi":
                    hhi_settings = config.get("hhi_settings", {})
                    benchmark = HHIBenchmark(
                        production_weight=hhi_settings.get("production_weight", 0.25),
                        reserve_weight=hhi_settings.get("reserve_weight", 0.75),
                        scale_to_0_10=hhi_settings.get("scale_to_0_10", True),
                    )

                elif family == "sun":
                    sun_settings = config.get("sun_settings", {})
                    benchmark = SUNBenchmark(
                        stability_threshold=sun_settings.get(
                            "stability_threshold", 0.0
                        ),
                        metastability_threshold=sun_settings.get(
                            "metastability_threshold", 0.1
                        ),
                        fingerprint_method=config.get(
                            "fingerprint_method", "structure-matcher"
                        ),
                        include_metasun=sun_settings.get("include_metasun", True),
                    )

                elif family == "stability":
                    stability_settings = config.get("stability_settings", {})
                    # Override to use single MLIP only (no ensemble)
                    stability_settings["use_ensemble"] = False
                    stability_settings["mlips"] = [mlip_name]
                    benchmark = MultiMLIPStabilityBenchmark(config=stability_settings)
                else:
                    logger.warning(f"Unknown benchmark family: {family}")
                    pbar.set_postfix({"status": "skipped"})
                    raise ValueError(f"Unknown benchmark family: {family}")

                # Run the benchmark
                benchmark_result = benchmark.evaluate(valid_structures)
                results[family] = benchmark_result

                elapsed_time = time.time() - start_time
                logger.info(
                    f"✅ {family} benchmark complete for {len(valid_structures)} valid structures in {elapsed_time:.1f}s"
                )

                # Clean up after each benchmark
                cleanup_after_benchmark(family, monitor_memory)

            except Exception as e:
                elapsed_time = time.time() - start_time
                logger.error(
                    f"❌ Failed to run {family} benchmark after {elapsed_time:.1f}s: {str(e)}"
                )
                results[family] = {"error": str(e)}

                # Clean up even if benchmark failed
                cleanup_after_benchmark(family, monitor_memory)

    # Log final memory usage
    log_memory_usage("after remaining benchmarks")

    return results


def save_results(
    validity_result: Dict[str, Any],
    remaining_results: Dict[str, Any],
    validity_filtering_metadata: Dict[str, Any],
    run_name: str,
    config_name: str,
    mlip_name: str,
    n_total_structures: int,
):
    """Save benchmark results to JSON file."""
    # Create results directory
    results_dir = Path(__file__).parent.parent / "results_final"
    results_dir.mkdir(exist_ok=True)

    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create filename
    filename = f"{run_name}_{mlip_name}_{config_name}_{timestamp}.json"
    filepath = results_dir / filename

    # Combine all results
    all_results = {"validity": validity_result}
    all_results.update(remaining_results)

    # Prepare results data
    output_data = {
        "run_info": {
            "run_name": run_name,
            "config_name": config_name,
            "mlip_name": mlip_name,
            "mlip_hull_type": get_mlip_hull_type(mlip_name),
            "timestamp": timestamp,
            "n_structures": n_total_structures,
            "benchmark_families": list(all_results.keys()),
            "validity_mandatory": True,
            "single_mlip_mode": True,
        },
        "validity_filtering": validity_filtering_metadata,
        "results": all_results,
    }

    # Save to JSON
    with open(filepath, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    logger.info(f"💾 Results saved to: {filepath}")
    return filepath


def main():
    """Main function to run single-MLIP benchmarks."""
    parser = argparse.ArgumentParser(
        description="Run material generation benchmarks with SINGLE MLIP (for before/after relaxation comparison)"
    )
    parser.add_argument(
        "--cifs",
        help="Path to text file containing CIF file paths OR directory containing CIF files",
    )
    parser.add_argument(
        "--csv", help="Path to CSV file containing structures in LeMatStructs column"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Benchmark configuration name (e.g., single_mlip)",
    )
    parser.add_argument("--name", required=True, help="Name for this benchmark run")
    parser.add_argument(
        "--mlip",
        required=True,
        choices=["orb", "mace", "uma"],
        help="MLIP to use for evaluation (orb, mace, or uma)",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        help="Specific benchmark families to run (validity is ALWAYS run regardless)",
    )
    parser.add_argument(
        "--fingerprint-method",
        default="structure-matcher",
        choices=["bawl", "short-bawl", "structure-matcher", "pdd"],
        help="Fingerprinting method to use (default: structure-matcher)",
    )
    parser.add_argument(
        "--monitor-memory",
        action="store_true",
        help="Enable detailed memory monitoring throughout the process",
    )
    parser.add_argument(
        "--generate-embedding-plots",
        action="store_true",
        help="Automatically generate embedding analysis plots after MLIP preprocessing",
    )
    parser.add_argument(
        "--respect-validity-flags",
        action="store_true",
        help="Respect validity flags in CSV (skip structures marked as invalid)",
    )

    args = parser.parse_args()

    # Validate input arguments
    if not args.cifs and not args.csv:
        parser.error("Either --cifs or --csv must be provided")
    if args.cifs and args.csv:
        parser.error("Only one of --cifs or --csv can be provided")

    try:
        # Log initial memory usage
        log_memory_usage("start of benchmark run", force_log=args.monitor_memory)

        # Load structures based on input type
        if args.csv:
            # Load structures from CSV
            structures = load_structures_from_wycoff_csv(
                args.csv, args.respect_validity_flags
            )
        else:
            # Load CIF files
            logger.info(f"Loading CIF files from: {args.cifs}")
            cif_paths = load_cif_files(args.cifs)
            logger.info(f"✅ Loaded {len(cif_paths)} CIF files")

            # Load structures from CIF files
            logger.info("Converting CIF files to structures...")
            structures = []

            # Add progress bar for structure loading
            with tqdm(cif_paths, desc="Loading CIF structures", unit="file") as pbar:
                for cif_path in pbar:
                    try:
                        # Load CIF file using pymatgen
                        from pymatgen.core import Structure

                        structure = Structure.from_file(cif_path)
                        structures.append(structure)
                        pbar.set_postfix(
                            {
                                "loaded": len(structures),
                                "failed": len(cif_paths) - len(structures),
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Failed to load {cif_path}: {str(e)}")
                        pbar.set_postfix(
                            {
                                "loaded": len(structures),
                                "failed": len(cif_paths) - len(structures),
                            }
                        )

            if not structures:
                raise ValueError("No valid structures loaded from CIF files")

        n_total_structures = len(structures)
        logger.info(f"✅ Loaded {n_total_structures} structures")

        # Load benchmark configuration
        logger.info(f"Loading benchmark configuration: {args.config}")
        config = load_benchmark_config(args.config)

        # Add fingerprint method to config
        if args.fingerprint_method != "structure-matcher":
            config["fingerprint_method"] = args.fingerprint_method
        logger.info(f"✅ Loaded configuration: {config.get('type', 'unknown')}")
        logger.info(
            f"🔍 Using fingerprint method: {config.get('fingerprint_method', args.fingerprint_method)}"
        )

        # Important note about single MLIP mode
        logger.info(f"🔥 SINGLE MLIP MODE: Using {args.mlip.upper()} for all evaluations")
        logger.info(f"🔥 Hull type: {get_mlip_hull_type(args.mlip)}")
        logger.info(
            "🔍 NOTE: Validity benchmark and preprocessor are MANDATORY and will ALWAYS run first"
        )

        # Determine benchmark families to run
        if args.families:
            benchmark_families = args.families
            logger.info(f"Using specified families: {benchmark_families}")
        else:
            # Default benchmark families
            benchmark_families = [
                "distribution",
                "diversity",
                "novelty",
                "uniqueness",
                "hhi",
                "sun",
                "stability",
            ]
            logger.info(f"Using benchmark families: {benchmark_families}")

        # Clear memory after loading structures
        clear_memory()
        log_memory_usage("after loading structures", force_log=args.monitor_memory)

        # Step 1: Run validity processing and filtering
        validity_result, valid_structures, validity_filtering_metadata = (
            run_validity_preprocessing_and_filtering(
                structures, config, args.monitor_memory
            )
        )

        # Check if we have valid structures to continue
        if len(valid_structures) == 0:
            logger.error(
                "❌ No valid structures found. Cannot continue with remaining benchmarks."
            )

            # Save results with empty remaining benchmarks
            results_file = save_results(
                validity_result,
                {},
                validity_filtering_metadata,
                args.name,
                args.config,
                args.mlip,
                n_total_structures,
            )

            # Print summary
            print("\n" + "=" * 60)
            print("⚠️  BENCHMARK RUN COMPLETE (NO VALID STRUCTURES)")
            print("=" * 60)
            print(f"📁 Results saved to: {results_file}")
            print(f"📊 Total structures: {n_total_structures}")
            print("📊 Valid structures: 0")
            print(f"🔥 MLIP: {args.mlip.upper()}")
            print("🔧 Only validity benchmark completed")
            print("=" * 60)
            return

        # Step 2: Determine preprocessor requirements
        preprocessor_config = create_preprocessor_config(
            benchmark_families, args.fingerprint_method, args.generate_embedding_plots
        )
        # Remove validity since it's already done
        preprocessor_config["validity"] = False
        logger.info(f"Remaining preprocessor config: {preprocessor_config}")

        # Step 3: Run remaining preprocessors with single MLIP
        processed_valid_structures, preprocessor_results = run_remaining_preprocessors(
            valid_structures,
            preprocessor_config,
            config,
            args.mlip,
            args.name,
            args.monitor_memory,
            args.generate_embedding_plots,
        )

        # Step 4: Run remaining benchmarks with single MLIP
        remaining_benchmark_results = run_remaining_benchmarks(
            processed_valid_structures,
            benchmark_families,
            config,
            args.mlip,
            args.monitor_memory,
        )

        # Save results
        results_file = save_results(
            validity_result,
            remaining_benchmark_results,
            validity_filtering_metadata,
            args.name,
            args.config,
            args.mlip,
            n_total_structures,
        )

        # Final cleanup
        logger.info("🧹 Performing final cleanup...")
        clear_memory()
        log_memory_usage("final cleanup", force_log=args.monitor_memory)

        # Print summary
        print("\n" + "=" * 60)
        print("🎉 SINGLE-MLIP BENCHMARK RUN COMPLETE")
        print("=" * 60)
        print(f"📁 Results saved to: {results_file}")
        print(f"📊 Total structures processed: {n_total_structures}")
        print(f"📊 Valid structures: {validity_filtering_metadata['valid_structures']}")
        print(
            f"📊 Invalid structures: {validity_filtering_metadata['invalid_structures']}"
        )
        print(f"📊 Validity rate: {validity_filtering_metadata['validity_rate']:.1%}")
        print(f"🔥 MLIP: {args.mlip.upper()} (hull: {get_mlip_hull_type(args.mlip)})")
        print(
            f"🔍 Fingerprint method: {config.get('fingerprint_method', args.fingerprint_method)}"
        )
        print(
            f"🔧 Benchmark families: {['validity (ALL structures)'] + [f'{family} (valid structures only)' for family in benchmark_families if family != 'validity']}"
        )
        print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        # Print key results
        all_results = {"validity": validity_result}
        all_results.update(remaining_benchmark_results)

        for family, result in all_results.items():
            if isinstance(result, dict) and "error" in result:
                scope = (
                    "ALL structures"
                    if family == "validity"
                    else "valid structures only"
                )
                print(f"❌ {family} ({scope}): {result['error']}")
            else:
                scope = (
                    "ALL structures (MANDATORY)"
                    if family == "validity"
                    else "valid structures only"
                )
                print(f"✅ {family} ({scope}): Completed successfully")

    except Exception as e:
        logger.error(f"Benchmark run failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

