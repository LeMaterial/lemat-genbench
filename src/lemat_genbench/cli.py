"""Command line interface for running benchmarks with enhanced metrics.

This module provides a CLI for running material generation benchmarks
using configuration files. This version uses the new enhanced benchmarks
(novelty_new, uniqueness_new, sun_new) while retaining other existing benchmarks.
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
from lemat_genbench.benchmarks.novelty_new_benchmark import (
    AugmentedNoveltyBenchmark,
)
from lemat_genbench.benchmarks.sun_new_benchmark import SUNNewBenchmark
from lemat_genbench.benchmarks.uniqueness_new_benchmark import (
    UniquenessNewBenchmark,
)
from lemat_genbench.benchmarks.validity_benchmark import (
    ValidityBenchmark,
)
from lemat_genbench.data.structure import format_structures
from lemat_genbench.metrics.validity_metrics import (
    ChargeNeutralityMetric,
    CoordinationEnvironmentMetric,
    MinimumInteratomicDistanceMetric,
    PhysicalPlausibilityMetric,
)
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
            "charge_weight": 0.25,
            "distance_weight": 0.25,
            "plausibility_weight": 0.25,
            "description": "Validity Benchmark for Materials Generation",
            "version": "0.1.0",
            "metric_configs": {
                "charge_neutrality": {"tolerance": 0.1, "strict": False},
                "interatomic_distance": {"scaling_factor": 0.5},
                "coordination_environment": {
                    "nn_method": "crystalnn",
                    "tolerance": 0.2,
                },
            },
        }
        with open(config_path, "w") as f:
            yaml.dump(validity_config, f, default_flow_style=False)

    # Create uniqueness_new config if needed
    if not config_path.exists() and config_path.name == "uniqueness_new.yaml":
        uniqueness_new_config = {
            "type": "uniqueness_new",
            "description": "Enhanced Uniqueness Benchmark using Augmented Fingerprints",
            "version": "0.1.0",
            "fingerprint_source": "auto",
            "symprec": 0.01,
            "angle_tolerance": 5.0,
        }
        with open(config_path, "w") as f:
            yaml.dump(uniqueness_new_config, f, default_flow_style=False)

    # Create novelty_new config if needed
    if not config_path.exists() and config_path.name == "novelty_new.yaml":
        novelty_new_config = {
            "type": "novelty_new",
            "description": "Enhanced Novelty Benchmark using Augmented Fingerprints",
            "version": "0.2.0",
            "fingerprinting_method": "augmented",
            "fingerprint_source": "auto",
            "reference_fingerprints_path": None,
            "reference_dataset_name": "LeMat-Bulk",
            "symprec": 0.01,
            "angle_tolerance": 5.0,
            "fallback_to_computation": True,
            "variants": {
                "default": {
                    "description": "Standard augmented novelty benchmark",
                    "fingerprint_source": "auto",
                    "symprec": 0.01,
                    "angle_tolerance": 5.0,
                    "fallback_to_computation": True,
                },
                "property_only": {
                    "description": "Uses only preprocessed fingerprints",
                    "fingerprint_source": "property",
                    "fallback_to_computation": False,
                },
                "computation_only": {
                    "description": "Computes fingerprints on-demand",
                    "fingerprint_source": "compute",
                    "fallback_to_computation": True,
                },
            },
        }
        with open(config_path, "w") as f:
            yaml.dump(novelty_new_config, f, default_flow_style=False)

    # Create sun_new config if needed
    if not config_path.exists() and config_path.name == "sun_new.yaml":
        sun_new_config = {
            "type": "sun_new",
            "description": "Enhanced SUN Benchmark using Augmented Fingerprinting",
            "version": "0.2.0",
            "include_metasun": True,
            "stability_threshold": 0.0,
            "metastability_threshold": 0.1,
            "fingerprinting_method": "augmented",
            "fingerprint_source": "auto",
            "reference_fingerprints_path": None,
            "reference_dataset_name": "LeMat-Bulk",
            "symprec": 0.01,
            "angle_tolerance": 5.0,
            "fallback_to_computation": True,
            "variants": {
                "default": {
                    "description": "Standard augmented SUN benchmark",
                    "fingerprint_source": "auto",
                    "symprec": 0.01,
                    "angle_tolerance": 5.0,
                    "fallback_to_computation": True,
                    "include_metasun": True,
                },
                "sun_only": {
                    "description": "SUN evaluation only (no MetaSUN)",
                    "fingerprint_source": "auto",
                    "include_metasun": False,
                },
            },
        }
        with open(config_path, "w") as f:
            yaml.dump(sun_new_config, f, default_flow_style=False)

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
            "embeddings": {
                "models": ["mace"],
                "num_samples": 1000,
                "normalize": True,
            },
            "metrics": {
                "frechet_distance": {"weight": 0.4},
                "mmd": {"weight": 0.4},
                "jensen_shannon": {"weight": 0.2},
            },
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

    # Create comprehensive config if needed
    if not config_path.exists() and config_path.name == "comprehensive.yaml":
        comprehensive_config = {
            "type": "comprehensive",
            "description": "Comprehensive Benchmark Suite with Enhanced Metrics",
            "version": "0.2.0",
            "benchmarks": {
                "validity": {
                    "weight": 0.2,
                    "config": {
                        "charge_weight": 0.25,
                        "distance_weight": 0.25,
                        "plausibility_weight": 0.25,
                    },
                },
                "distribution": {
                    "weight": 0.15,
                    "config": {
                        "embeddings": {"models": ["mace"], "num_samples": 1000},
                        "metrics": {
                            "frechet_distance": {"weight": 0.4},
                            "mmd": {"weight": 0.4},
                            "jensen_shannon": {"weight": 0.2},
                        },
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
                "uniqueness_new": {
                    "weight": 0.15,
                    "config": {
                        "fingerprint_source": "auto",
                        "symprec": 0.01,
                        "angle_tolerance": 5.0,
                    },
                },
                "novelty_new": {
                    "weight": 0.15,
                    "config": {
                        "fingerprint_source": "auto",
                        "symprec": 0.01,
                        "angle_tolerance": 5.0,
                    },
                },
                "sun_new": {
                    "weight": 0.2,
                    "config": {
                        "include_metasun": True,
                        "fingerprint_source": "auto",
                        "symprec": 0.01,
                        "angle_tolerance": 5.0,
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
    default="results/benchmark_results_new.yaml",
)
def main(input: str, config_name: str, output: str):
    """Run a benchmark on structures using the specified configuration with enhanced metrics.

    INPUT: Path to CSV file containing structures to evaluate or directory with CIF files
    CONFIG_NAME: Name of the benchmark configuration (e.g. 'novelty_new' for
    novelty_new.yaml) or path to a config file

    This CLI uses enhanced benchmarks:
    - novelty_new: Enhanced novelty evaluation using augmented fingerprints
    - uniqueness_new: Enhanced uniqueness evaluation using augmented fingerprints  
    - sun_new: Enhanced SUN benchmark using augmented fingerprinting
    
    Other benchmarks (validity, distribution, diversity, hhi, multi_mlip_stability) 
    remain unchanged.
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
            # Get metric-specific configs if available
            metric_configs = config.get("metric_configs", {})

            # Extract charge neutrality config
            charge_config = metric_configs.get("charge_neutrality", {})
            charge_tolerance = charge_config.get("tolerance", 0.1)
            charge_strict = charge_config.get("strict", False)

            # Extract interatomic distance config
            distance_config = metric_configs.get("interatomic_distance", {})
            distance_scaling = distance_config.get("scaling_factor", 0.5)

            # Extract coordination environment config
            coord_config = metric_configs.get("coordination_environment", {})
            coord_nn_method = coord_config.get("nn_method", "crystalnn")
            coord_tolerance = coord_config.get("tolerance", 0.2)

            # Create custom metrics with configuration
            ChargeNeutralityMetric(tolerance=charge_tolerance, strict=charge_strict)

            MinimumInteratomicDistanceMetric(scaling_factor=distance_scaling)

            CoordinationEnvironmentMetric(
                nn_method=coord_nn_method, tolerance=coord_tolerance
            )

            PhysicalPlausibilityMetric()

            # Create benchmark with custom metrics
            benchmark = ValidityBenchmark()

        elif benchmark_type == "distribution":
            benchmark = DistributionBenchmark()

        elif benchmark_type == "diversity":
            benchmark = DiversityBenchmark()

        elif benchmark_type == "hhi":
            benchmark = HHIBenchmark()

        elif benchmark_type == "multi_mlip_stability":
            benchmark = MultiMLIPStabilityBenchmark()

        # Enhanced benchmarks using new implementations
        elif benchmark_type == "uniqueness_new":
            # Extract configuration parameters
            fingerprint_source = config.get("fingerprint_source", "auto")
            symprec = config.get("symprec", 0.01)
            angle_tolerance = config.get("angle_tolerance", 5.0)

            benchmark = UniquenessNewBenchmark(
                fingerprint_source=fingerprint_source,
                symprec=symprec,
                angle_tolerance=angle_tolerance,
            )

        elif benchmark_type == "novelty_new":
            # Extract configuration parameters
            fingerprint_source = config.get("fingerprint_source", "auto")
            reference_fingerprints_path = config.get("reference_fingerprints_path")
            reference_dataset_name = config.get("reference_dataset_name", "LeMat-Bulk")
            symprec = config.get("symprec", 0.01)
            angle_tolerance = config.get("angle_tolerance", 5.0)
            fallback_to_computation = config.get("fallback_to_computation", True)

            benchmark = AugmentedNoveltyBenchmark(
                fingerprint_source=fingerprint_source,
                reference_fingerprints_path=reference_fingerprints_path,
                reference_dataset_name=reference_dataset_name,
                symprec=symprec,
                angle_tolerance=angle_tolerance,
                fallback_to_computation=fallback_to_computation,
            )

        elif benchmark_type == "sun_new":
            # Extract configuration parameters
            include_metasun = config.get("include_metasun", True)
            stability_threshold = config.get("stability_threshold", 0.0)
            metastability_threshold = config.get("metastability_threshold", 0.1)
            fingerprint_source = config.get("fingerprint_source", "auto")
            reference_fingerprints_path = config.get("reference_fingerprints_path")
            reference_dataset_name = config.get("reference_dataset_name", "LeMat-Bulk")
            symprec = config.get("symprec", 0.01)
            angle_tolerance = config.get("angle_tolerance", 5.0)
            fallback_to_computation = config.get("fallback_to_computation", True)

            benchmark = SUNNewBenchmark(
                stability_threshold=stability_threshold,
                metastability_threshold=metastability_threshold,
                reference_fingerprints_path=reference_fingerprints_path,
                reference_dataset_name=reference_dataset_name,
                fingerprint_source=fingerprint_source,
                symprec=symprec,
                angle_tolerance=angle_tolerance,
                fallback_to_computation=fallback_to_computation,
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