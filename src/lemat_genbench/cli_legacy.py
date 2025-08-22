"""Command line interface for running benchmarks.

This module provides a CLI for running material generation benchmarks
using configuration files.
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
from lemat_genbench.benchmarks.novelty_benchmark import (
    NoveltyBenchmark,
)
from lemat_genbench.benchmarks.sun_benchmark import SUNBenchmark
from lemat_genbench.benchmarks.uniqueness_benchmark import (
    UniquenessBenchmark,
)
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
            # Individual metric configurations - no weights needed
            "charge_tolerance": 0.1,
            "distance_scaling": 0.5,
            "min_density": 0.01,
            "max_density": 25.0,
            "check_format": True,
            "check_symmetry": True,
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

    if not config_path.exists() and config_path.name == "novelty.yaml":
        novelty_config = {
            "type": "novelty",
            "description": "Novelty Benchmark for Materials Generation",
            "version": "0.1.0",
            "reference_dataset": "LeMaterial/LeMat-Bulk",
            "reference_config": "compatible_pbe",
            "fingerprint_method": "bawl",
            "cache_reference": True,
            "max_reference_size": None,
        }
        with open(config_path, "w") as f:
            yaml.dump(novelty_config, f, default_flow_style=False)

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

    # Add SUN config creation
    if not config_path.exists() and config_path.name == "sun.yaml":
        sun_config = {
            "type": "sun",
            "description": "SUN Benchmark for evaluating structures that are Stable, Unique, and Novel",
            "version": "0.1.0",
            "include_metasun": True,
            "stability_threshold": 0.0,
            "metastability_threshold": 0.1,
            "reference_dataset": "LeMaterial/LeMat-Bulk",
            "reference_config": "compatible_pbe",
            "fingerprint_method": "bawl",
            "cache_reference": True,
            "max_reference_size": None,
        }
        with open(config_path, "w") as f:
            yaml.dump(sun_config, f, default_flow_style=False)

    # Add Multi-MLIP Stability config creation
    if not config_path.exists() and config_path.name == "stability.yaml":
        stability_config = {
            "type": "stability",
            "description": "Multi-MLIP Stability Benchmark with Ensemble Predictions",
            "version": "0.1.0",
            "use_ensemble": True,
            "mlip_names": ["orb", "mace", "uma"],
            "metastable_threshold": 0.1,
            "ensemble_config": {
                "min_mlips_required": 2,
            },
            "individual_mlip_config": {
                "use_all_available": True,
                "require_all_mlips": False,
                "fallback_to_single": True,
            },
        }
        with open(config_path, "w") as f:
            yaml.dump(stability_config, f, default_flow_style=False)

    # Add Distribution config creation
    if not config_path.exists() and config_path.name == "distribution.yaml":
        distribution_config = {
            "type": "distribution",
            "mlips": ["orb", "mace", "uma"],
            "cache_dir": "./data",
            "js_distributions_file": "data/lematbulk_jsdistance_distributions.json",
            "mmd_values_file": "data/lematbulk_mmd_values_15k.pkl",
            "description": "Distribution Benchmark for Materials Generation - evaluates similarity to reference distributions",
            "version": "0.1.0",
            "metadata": {
                "reference": "Distribution similarity metrics for evaluating generated materials against reference datasets",
                "use_case": "Assessing whether generated structures follow realistic distributions of structural properties",
            },
        }
        with open(config_path, "w") as f:
            yaml.dump(distribution_config, f, default_flow_style=False)

    # Add Diversity config creation
    if not config_path.exists() and config_path.name == "diversity.yaml":
        diversity_config = {
            "type": "diversity",
            "description": "Diversity Benchmark for Materials Generation - evaluates structural diversity across multiple dimensions",
            "version": "0.1.0",
            "metric_configs": {
                "element_diversity": {
                    "reference_element_space": 118,
                },
                "space_group_diversity": {
                    "reference_space_group_space": 230,
                },
                "physical_size_diversity": {
                    "density_bin_size": 0.5,
                    "lattice_bin_size": 0.5,
                    "packing_factor_bin_size": 0.05,
                },
                "site_number_diversity": {
                    # No additional parameters needed
                },
            },
            "metadata": {
                "reference": "Diversity metrics for evaluating structural variety in generated materials",
                "use_case": "Assessing whether generated structures explore diverse chemical and structural space",
            },
        }
        with open(config_path, "w") as f:
            yaml.dump(diversity_config, f, default_flow_style=False)

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

        if benchmark_type == "validity":
            # Extract validity parameters (no weights needed in new implementation)
            charge_tolerance = config.get("charge_tolerance", 0.1)
            distance_scaling = config.get("distance_scaling", 0.5)
            min_density = config.get("min_density", 0.01)
            max_density = config.get("max_density", 25.0)
            check_format = config.get("check_format", True)
            check_symmetry = config.get("check_symmetry", True)

            # Create benchmark with new validity logic
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

        elif benchmark_type == "stability":
            # Create multi-MLIP stability benchmark from config
            benchmark = MultiMLIPStabilityBenchmark(
                config=config,
                name=config.get("name", "MultiMLIPStabilityBenchmark"),
                description=config.get("description"),
                metadata={
                    "version": config.get("version", "0.1.0"),
                    **(config.get("metadata", {})),
                },
            )

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

        elif benchmark_type == "novelty":
            # Create novelty benchmark from config
            benchmark = NoveltyBenchmark(
                reference_dataset=config.get(
                    "reference_dataset", "LeMaterial/LeMat-Bulk"
                ),
                reference_config=config.get("reference_config", "compatible_pbe"),
                fingerprint_method=config.get("fingerprint_method", "bawl"),
                cache_reference=config.get("cache_reference", True),
                max_reference_size=config.get("max_reference_size", None),
                name=config.get("name", "NoveltyBenchmark"),
                description=config.get("description"),
                metadata={
                    "version": config.get("version", "0.1.0"),
                    **(config.get("metadata", {})),
                },
            )

        elif benchmark_type == "hhi":
            # Create HHI benchmark from config
            benchmark = HHIBenchmark(
                production_weight=config.get("production_weight", 0.25),
                reserve_weight=config.get("reserve_weight", 0.75),
                scale_to_0_10=config.get("scale_to_0_10", True),
                name=config.get("name", "HHIBenchmark"),
                description=config.get("description"),
                metadata={
                    "version": config.get("version", "0.1.0"),
                    **(config.get("metadata", {})),
                },
            )

        elif benchmark_type == "sun":
            # Create SUN benchmark from config
            benchmark = SUNBenchmark(
                stability_threshold=config.get("stability_threshold", 0.0),
                metastability_threshold=config.get("metastability_threshold", 0.1),
                reference_dataset=config.get(
                    "reference_dataset", "LeMaterial/LeMat-Bulk"
                ),
                reference_config=config.get("reference_config", "compatible_pbe"),
                fingerprint_method=config.get("fingerprint_method", "bawl"),
                cache_reference=config.get("cache_reference", True),
                max_reference_size=config.get("max_reference_size", None),
                include_metasun=config.get("include_metasun", True),
                name=config.get("name", "SUNBenchmark"),
                description=config.get("description"),
                metadata={
                    "version": config.get("version", "0.1.0"),
                    **(config.get("metadata", {})),
                },
            )

        elif benchmark_type == "distribution":
            # Create distribution benchmark from config
            benchmark = DistributionBenchmark(
                mlips=config.get("mlips", ["orb", "mace", "uma"]),
                cache_dir=config.get("cache_dir", "./data"),
                js_distributions_file=config.get(
                    "js_distributions_file", "data/lematbulk_jsdistance_distributions.json"
                ),
                mmd_values_file=config.get(
                    "mmd_values_file", "data/lematbulk_mmd_values_15k.pkl"
                ),
                name=config.get("name", "DistributionBenchmark"),
                description=config.get("description"),
                metadata={
                    "version": config.get("version", "0.1.0"),
                    **(config.get("metadata", {})),
                },
            )

        elif benchmark_type == "diversity":
            # Create diversity benchmark from config
            benchmark = DiversityBenchmark(
                name=config.get("name", "DiversityBenchmark"),
                description=config.get("description"),
                metadata={
                    "version": config.get("version", "0.1.0"),
                    **(config.get("metadata", {})),
                },
            )

        else:
            raise ValueError(
                f"Unknown benchmark type: {benchmark_type}. "
                "Available types: validity, stability, uniqueness, novelty, hhi, sun, diversity, distribution"
            )

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