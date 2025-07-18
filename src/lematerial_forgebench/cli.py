"""Command line interface for running benchmarks.

This module provides a CLI for running material generation benchmarks
using configuration files.
"""

import os
from pathlib import Path

import click
import yaml

# from lematerial_forgebench.benchmarks.example import ExampleBenchmark
from lematerial_forgebench.benchmarks.stability_benchmark import (
    StabilityBenchmark,
)
from lematerial_forgebench.benchmarks.uniqueness_benchmark import (
    UniquenessBenchmark,
)
from lematerial_forgebench.benchmarks.validity_benchmark import (
    ValidityBenchmark,
)
from lematerial_forgebench.data.structure import format_structures
from lematerial_forgebench.metrics.validity_metrics import (
    ChargeNeutralityMetric,
    CoordinationEnvironmentMetric,
    MinimumInteratomicDistanceMetric,
    PhysicalPlausibilityMetric,
)
from lematerial_forgebench.preprocess.universal_stability_preprocess import (
    UniversalStabilityPreprocessor,
)
from lematerial_forgebench.utils.logging import logger

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
            "coordination_weight": 0.25,
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
            
    if not config_path.exists() and config_path.name == "uniqueness.yaml":
        uniqueness_config = {
            "type": "uniqueness",
            "description": "Uniqueness Benchmark for Materials Generation",
            "version": "0.1.0",
            "fingerprint_method": "bawl",
        }
        with open(config_path, "w") as f:
            yaml.dump(uniqueness_config, f, default_flow_style=False)

    if not config_path.exists():
        raise click.ClickException(
            f"Config '{config_path}' not found. "
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

    STRUCTURES_CSV: Path to CSV file containing structures to evaluate
    CONFIG_NAME: Name of the benchmark configuration (e.g. 'example' for 
    example.yaml) or path to a config file
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

        # if benchmark_type == "example":
        #     benchmark = ExampleBenchmark(
        #         quality_weight=config.get("quality_weight", 0.4),
        #         diversity_weight=config.get("diversity_weight", 0.4),
        #         novelty_weight=config.get("novelty_weight", 0.2),
        #     )
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
            ChargeNeutralityMetric(
                tolerance=charge_tolerance, strict=charge_strict
            )

            MinimumInteratomicDistanceMetric(scaling_factor=distance_scaling)

            CoordinationEnvironmentMetric(
                nn_method=coord_nn_method, tolerance=coord_tolerance
            )

            PhysicalPlausibilityMetric()

            # Create benchmark with custom metrics
            benchmark = ValidityBenchmark(
                charge_weight=config.get("charge_weight", 0.25),
                distance_weight=config.get("distance_weight", 0.25),
                coordination_weight=config.get("coordination_weight", 0.25),
                plausibility_weight=config.get("plausibility_weight", 0.25),
                name=config.get("name", "ValidityBenchmark"),
                description=config.get("description"),
                metadata={
                    "version": config.get("version", "0.1.0"),
                    "metric_configs": metric_configs,
                },
            )

        elif benchmark_type == "stability":
            # before running the benchmark, we need to preprocess the structures
            ppc = config.get("preprocessor_config", {})
            stability_preprocessor = UniversalStabilityPreprocessor(
                model_type=ppc.get("model_type", "orb"),
                model_config=ppc.get("model_config", {}),
                relax_structures=ppc.get("relax_structures", True),
                relaxation_config=ppc.get("relaxation_config", {}),
                calculate_formation_energy=ppc.get(
                    "calculate_formation_energy", True
                ),
                calculate_energy_above_hull=ppc.get(
                    "calculate_energy_above_hull", True
                ),
                extract_embeddings=ppc.get("extract_embeddings", True),
            )
            # Use the preprocessor to process structures
            preprocessor_result = stability_preprocessor(structures)
            structures = preprocessor_result.processed_structures

            benchmark = StabilityBenchmark()
            
        elif benchmark_type == "uniqueness":
            # Create uniqueness benchmark from config
            benchmark = UniquenessBenchmark(
                fingerprint_method=config.get("fingerprint_method", "bawl"),
                name=config.get("name", "UniquenessBenchmark"),
                description=config.get("description"),
                metadata={
                    "version": config.get("version", "0.1.0"),
                    **(config.get("metadata", {})),
                },
            )
        else:
            raise ValueError(f"Unknown benchmark type: {benchmark_type}")

        # Run benchmark
        logger.info("Running benchmark evaluation")
        results = benchmark.evaluate(structures=structures)

        # Save results
        logger.info(f"Saving results to {output}")
        save_results(results.__dict__, output)

        click.echo("\nBenchmark Results Summary:")
        for score_name, score in results.final_scores.items():
            click.echo(f"{score_name}: {score:.3f}")

    except Exception as e:
        logger.error("Benchmark execution failed", exc_info=True)
        raise click.ClickException(str(e))


if __name__ == "__main__":
    main()
