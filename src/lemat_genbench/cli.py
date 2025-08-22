"""Command line interface for running benchmarks.

This module provides a CLI for running material generation benchmarks
using configuration files. This version uses the original benchmarks
that are currently available and working.
"""

import os
from pathlib import Path

import click
import yaml

from lemat_genbench.benchmarks.distribution_benchmark import (
    DistributionBenchmark,
)
from lemat_genbench.benchmarks.diversity_benchmark import (
    DiversityBenchmark,
)
from lemat_genbench.benchmarks.hhi_benchmark import HHIBenchmark
from lemat_genbench.benchmarks.multi_mlip_stability_benchmark import (
    StabilityBenchmark as MultiMLIPStabilityBenchmark,
)

# Use original benchmarks that are currently working
from lemat_genbench.benchmarks.novelty_benchmark import NoveltyBenchmark
from lemat_genbench.benchmarks.sun_benchmark import SUNBenchmark
from lemat_genbench.benchmarks.uniqueness_benchmark import UniquenessBenchmark
from lemat_genbench.benchmarks.validity_benchmark import (
    ValidityBenchmark,
)
from lemat_genbench.data.structure import format_structures
from lemat_genbench.utils.logging import logger

CONFIGS_DIR = Path(__file__).parent.parent / "config"


def load_benchmark_config(config_name: str) -> dict:
    """Load benchmark configuration from YAML file.

    Parameters
    ----------
    config_name : str
        Name of the config file (with or without .yaml extension)
        Will look for the config in the standard configs directory

    Returns
    -------
    dict
        Benchmark configuration
    """
    # Ensure configs directory exists
    if not CONFIGS_DIR.exists():
        CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    # If config_name is a full path, use it directly
    config_path = Path(config_name)
    if not config_path.is_absolute():
        # Add .yaml extension if not present
        if not config_name.endswith(".yaml"):
            config_name = f"{config_name}.yaml"
        config_path = CONFIGS_DIR / config_name

    # If config doesn't exist but it's the example config, create it
    if not config_path.exists() and config_path.name == "example.yaml":
        example_config = {
            "type": "example",
            "quality_weight": 0.4,
            "diversity_weight": 0.4,
            "novelty_weight": 0.2,
        }
        with open(config_path, "w") as f:
            yaml.dump(example_config, f, default_flow_style=False)

    # If config doesn't exist but it's the validity config, create it
    if not config_path.exists() and config_path.name == "validity.yaml":
        validity_config = {
            "type": "validity",
            "description": "Validity Benchmark for Materials Generation",
            "version": "0.2.0",
            # Individual metric configurations
            "charge_tolerance": 0.1,
            "distance_scaling": 0.5,
            "min_density": 1.0,
            "max_density": 25.0,
            "check_format": True,
            "check_symmetry": True,
            # Note: No weights needed - overall validity is intersection of all checks
        }
        with open(config_path, "w") as f:
            yaml.dump(validity_config, f, default_flow_style=False)

    # Create uniqueness config if needed (using original benchmark)
    if not config_path.exists() and config_path.name == "uniqueness.yaml":
        uniqueness_config = {
            "type": "uniqueness",
            "description": "Uniqueness Benchmark using BAWL/short-BAWL fingerprints",
            "version": "0.1.0",
            "fingerprint_method": "short-bawl",
            "n_jobs": 1,
        }
        with open(config_path, "w") as f:
            yaml.dump(uniqueness_config, f, default_flow_style=False)

    # Create novelty config if needed (using original benchmark)
    if not config_path.exists() and config_path.name == "novelty.yaml":
        novelty_config = {
            "type": "novelty",
            "description": "Novelty Benchmark using BAWL/short-BAWL fingerprints",
            "version": "0.1.0",
            "fingerprint_method": "short-bawl",
            "reference_dataset": "LeMaterial/LeMat-Bulk",
            "reference_config": "compatible_pbe",
            "cache_reference": True,
            "max_reference_size": None,
            "n_jobs": 1,
        }
        with open(config_path, "w") as f:
            yaml.dump(novelty_config, f, default_flow_style=False)

    # Create sun config if needed (using original benchmark)
    if not config_path.exists() and config_path.name == "sun.yaml":
        sun_config = {
            "type": "sun",
            "description": "SUN Benchmark using BAWL/short-BAWL fingerprints",
            "version": "0.1.0",
            "stability_threshold": 0.0,
            "metastability_threshold": 0.1,
            "fingerprint_method": "short-bawl",
            "include_metasun": True,
        }
        with open(config_path, "w") as f:
            yaml.dump(sun_config, f, default_flow_style=False)

    # Add HHI config creation
    if not config_path.exists() and config_path.name == "hhi.yaml":
        hhi_config = {
            "type": "hhi",
            "description": (
                "HHI (Herfindahl-Hirschman Index) Benchmark for Supply Risk Assessment"
            ),
            "version": "0.1.0",
            "production_weight": 0.25,
            "reserve_weight": 0.75,
            "scale_to_0_10": True,
            "metadata": {
                "reference": ("Herfindahl-Hirschman Index for supply risk assessment"),
                "use_case": (
                    "Evaluating element supply concentration risk in materials"
                ),
            },
        }
        with open(config_path, "w") as f:
            yaml.dump(hhi_config, f, default_flow_style=False)

    # Add distribution config creation
    if not config_path.exists() and config_path.name == "distribution.yaml":
        distribution_config = {
            "type": "distribution",
            "description": "Distribution Benchmark for Materials Generation",
            "version": "0.1.0",
            "mlips": ["orb", "mace", "uma"],
            "cache_dir": "./data",
            "js_distributions_file": "data/lematbulk_jsdistance_distributions.json",
            "mmd_values_file": "data/lematbulk_mmd_values_15k.pkl",
        }
        with open(config_path, "w") as f:
            yaml.dump(distribution_config, f, default_flow_style=False)

    # Add diversity config creation
    if not config_path.exists() and config_path.name == "diversity.yaml":
        diversity_config = {
            "type": "diversity",
            "description": "Diversity Benchmark for Materials Generation",
            "version": "0.1.0",
            "element_weight": 0.25,
            "space_group_weight": 0.25,
            "site_number_weight": 0.25,
            "physical_size_weight": 0.25,
        }
        with open(config_path, "w") as f:
            yaml.dump(diversity_config, f, default_flow_style=False)

    # Add multi_mlip_stability config creation
    if not config_path.exists() and config_path.name == "multi_mlip_stability.yaml":
        stability_config = {
            "type": "multi_mlip_stability",
            "description": "Multi-MLIP Stability Benchmark for Materials Generation",
            "version": "0.1.0",
            "models": ["mace", "orb"],
            "formation_energy_weight": 0.5,
            "e_above_hull_weight": 0.5,
        }
        with open(config_path, "w") as f:
            yaml.dump(stability_config, f, default_flow_style=False)

    # Create comprehensive config if needed (using original benchmarks)
    if not config_path.exists() and config_path.name == "comprehensive.yaml":
        comprehensive_config = {
            "type": "comprehensive",
            "description": "Comprehensive Benchmark Suite using Original Benchmarks",
            "version": "0.2.0",
            "benchmarks": {
                "validity": {
                    "weight": 0.2,
                    "config": {
                        "charge_tolerance": 0.1,
                        "distance_scaling": 0.5,
                        "min_density": 1.0,
                        "max_density": 25.0,
                        "check_format": True,
                        "check_symmetry": True,
                    },
                },
                "distribution": {
                    "weight": 0.15,
                    "config": {
                        "mlips": ["orb", "mace", "uma"],
                        "cache_dir": "./data",
                        "js_distributions_file": "data/lematbulk_jsdistance_distributions.json",
                        "mmd_values_file": "data/lematbulk_mmd_values_15k.pkl",
                    },
                },
                "diversity": {
                    "weight": 0.15,
                    "config": {
                        "element_weight": 0.25,
                        "space_group_weight": 0.25,
                        "site_number_weight": 0.25,
                        "physical_size_weight": 0.25,
                    },
                },
                "uniqueness": {
                    "weight": 0.15,
                    "config": {
                        "fingerprint_method": "short-bawl",
                        "n_jobs": 1,
                    },
                },
                "novelty": {
                    "weight": 0.15,
                    "config": {
                        "fingerprint_method": "short-bawl",
                        "reference_dataset": "LeMaterial/LeMat-Bulk",
                        "reference_config": "compatible_pbe",
                        "cache_reference": True,
                        "n_jobs": 1,
                    },
                },
                "sun": {
                    "weight": 0.2,
                    "config": {
                        "stability_threshold": 0.0,
                        "metastability_threshold": 0.1,
                        "fingerprint_method": "short-bawl",
                        "include_metasun": True,
                    },
                },
            },
        }
        with open(config_path, "w") as f:
            yaml.dump(comprehensive_config, f, default_flow_style=False)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            "Available configs in standard directory: "
            + ", ".join(f.stem for f in CONFIGS_DIR.glob("*.yaml"))
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_results(results: dict, output_path: str):
    """Save benchmark results to file.

    Parameters
    ----------
    results : dict
        Benchmark results
    output_path : str
        Path to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save results in YAML format
    with open(output_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False)


@click.command()
@click.argument("input", type=click.Path(exists=True))
@click.argument("config_name", type=str)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Path to save results",
    default="results/benchmark_results.yaml",
)
def main(input: str, config_name: str, output: str):
    """Run a benchmark on structures using the specified configuration.

    INPUT: Path to CSV file containing structures to evaluate or directory with CIF files
    CONFIG_NAME: Name of the benchmark configuration (e.g. 'novelty' for
    novelty.yaml) or path to a config file

    This CLI uses the original benchmarks that are currently working:
    - novelty: Original novelty evaluation using BAWL/short-BAWL fingerprints
    - uniqueness: Original uniqueness evaluation using BAWL/short-BAWL fingerprints  
    - sun: Original SUN benchmark using BAWL/short-BAWL fingerprints
    - validity: Validity benchmark for structure validation
    - distribution: Distribution similarity evaluation
    - diversity: Diversity metrics evaluation
    - hhi: Supply risk assessment
    - multi_mlip_stability: Multi-MLIP stability evaluation
    """
    try:
        # Load structures
        logger.info(f"Loading structures from {input}")
        structures = format_structures(input)
        if not structures:
            logger.error("No valid structures loaded")
            return

        # Benchmark configuration
        logger.info(f"Loading benchmark configuration '{config_name}'")
        config = load_benchmark_config(config_name)

        # Initialization
        benchmark_type = config.get("type", "example")

        if benchmark_type == "validity":
            # Extract validity parameters (no weights needed)
            charge_tolerance = config.get("charge_tolerance", 0.1)
            distance_scaling = config.get("distance_scaling", 0.5)
            min_density = config.get("min_density", 1.0)
            max_density = config.get("max_density", 25.0)
            check_format = config.get("check_format", True)
            check_symmetry = config.get("check_symmetry", True)

            # Create benchmark with validity logic
            benchmark = ValidityBenchmark(
                charge_tolerance=charge_tolerance,
                distance_scaling=distance_scaling,
                min_density=min_density,
                max_density=max_density,
                check_format=check_format,
                check_symmetry=check_symmetry,
                name=config.get("name", "ValidityBenchmark"),
                description=config.get("description"),
                metadata={
                    "version": config.get("version", "0.2.0"),
                    "config": config,
                },
            )

        elif benchmark_type == "distribution":
            # Extract distribution parameters
            mlips = config.get("mlips", ["orb", "mace", "uma"])
            cache_dir = config.get("cache_dir", "./data")
            js_distributions_file = config.get(
                "js_distributions_file",
                "data/lematbulk_jsdistance_distributions.json",
            )
            mmd_values_file = config.get(
                "mmd_values_file", "data/lematbulk_mmd_values_15k.pkl"
            )

            benchmark = DistributionBenchmark(
                mlips=mlips,
                cache_dir=cache_dir,
                js_distributions_file=js_distributions_file,
                mmd_values_file=mmd_values_file,
            )

        elif benchmark_type == "diversity":
            benchmark = DiversityBenchmark()

        elif benchmark_type == "hhi":
            # Extract HHI parameters
            production_weight = config.get("production_weight", 0.25)
            reserve_weight = config.get("reserve_weight", 0.75)
            scale_to_0_10 = config.get("scale_to_0_10", True)

            benchmark = HHIBenchmark(
                production_weight=production_weight,
                reserve_weight=reserve_weight,
                scale_to_0_10=scale_to_0_10,
            )

        elif benchmark_type == "multi_mlip_stability":
            benchmark = MultiMLIPStabilityBenchmark()

        # Original benchmarks
        elif benchmark_type == "uniqueness":
            # Extract configuration parameters
            fingerprint_method = config.get("fingerprint_method", "short-bawl")
            n_jobs = config.get("n_jobs", 1)

            benchmark = UniquenessBenchmark(
                fingerprint_method=fingerprint_method,
                n_jobs=n_jobs,
            )

        elif benchmark_type == "novelty":
            # Extract configuration parameters
            fingerprint_method = config.get("fingerprint_method", "short-bawl")
            reference_dataset = config.get("reference_dataset", "LeMaterial/LeMat-Bulk")
            reference_config = config.get("reference_config", "compatible_pbe")
            cache_reference = config.get("cache_reference", True)
            max_reference_size = config.get("max_reference_size", None)
            n_jobs = config.get("n_jobs", 1)

            benchmark = NoveltyBenchmark(
                fingerprint_method=fingerprint_method,
                reference_dataset=reference_dataset,
                reference_config=reference_config,
                cache_reference=cache_reference,
                max_reference_size=max_reference_size,
                n_jobs=n_jobs,
            )

        elif benchmark_type == "sun":
            # Extract configuration parameters
            stability_threshold = config.get("stability_threshold", 0.0)
            metastability_threshold = config.get("metastability_threshold", 0.1)
            fingerprint_method = config.get("fingerprint_method", "short-bawl")
            include_metasun = config.get("include_metasun", True)

            benchmark = SUNBenchmark(
                stability_threshold=stability_threshold,
                metastability_threshold=metastability_threshold,
                fingerprint_method=fingerprint_method,
                include_metasun=include_metasun,
            )

        else:
            logger.error(f"Unknown benchmark type: {benchmark_type}")
            return

        # Run benchmark
        logger.info(f"Running {benchmark_type} benchmark on {len(structures)} structures")
        results = benchmark.evaluate(structures)

        # Save results
        logger.info(f"Saving results to {output}")
        
        # Convert results to dictionary format for saving
        results_dict = {
            "benchmark_type": benchmark_type,
            "config": config,
            "final_scores": results.final_scores,
            "evaluator_results": results.evaluator_results,
            "metadata": results.metadata,
        }
        
        save_results(results_dict, output)
        
        logger.info("Benchmark completed successfully")
        
        # Print summary
        print(f"\n{'='*50}")
        print("Benchmark Results Summary")
        print(f"{'='*50}")
        print(f"Benchmark Type: {benchmark_type}")
        print(f"Structures Evaluated: {len(structures)}")
        print("Final Scores:")
        for metric, score in results.final_scores.items():
            print(f"  {metric}: {score:.4f}")
        print(f"Results saved to: {output}")

    except Exception as e:
        logger.error(f"Error running benchmark: {e}")
        raise


if __name__ == "__main__":
    main()