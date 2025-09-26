"""Multi-MLIP stability benchmark for ensemble-based evaluation.

This implementation provides objective statistical measures from multiple MLIP
ensemble predictions, supporting both individual and ensemble modes with
configurable reporting options.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import yaml

from lemat_genbench.benchmarks.base import BaseBenchmark
from lemat_genbench.evaluator import EvaluatorConfig
from lemat_genbench.metrics.multi_mlip_stability_metrics import (
    E_HullMetric,
    FormationEnergyMetric,
    MetastabilityMetric,
    RelaxationStabilityMetric,
    RelaxedE_HullMetric,
    RelaxedFormationEnergyMetric,
    RelaxedMetastabilityMetric,
    RelaxedStabilityMetric,
    StabilityMetric,
)


def safe_float(value: Any) -> float:
    """Safely convert value to float, returning NaN for None."""
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def safe_int(value: Any) -> int:
    """Safely convert value to int, returning 0 for None."""
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def load_config(config_path: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    """Load configuration from file or dict.

    Parameters
    ----------
    config_path : Union[str, Path, Dict[str, Any]]
        Path to YAML config file or config dictionary

    Returns
    -------
    Dict[str, Any]
        Configuration dictionary
    """
    if isinstance(config_path, dict):
        return config_path

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


class StabilityBenchmark(BaseBenchmark):
    """Multi-MLIP stability benchmark supporting both individual and ensemble modes.

    This benchmark evaluates structure stability using predictions from multiple MLIPs,
    supporting both ensemble statistics and individual model results with configurable
    reporting options.

    Parameters
    ----------
    config : Union[str, Path, Dict[str, Any]], optional
        Path to YAML configuration file or configuration dictionary.
        If None, uses default configuration.
    use_ensemble : bool, default=True
        Whether to use ensemble statistics or individual MLIP results
    mlip_names : List[str], optional
        Specific MLIPs to use
    metastable_threshold : float, default=0.1
        Energy above hull threshold for metastability classification (eV/atom)
    min_mlips_required : int, default=2
        Minimum MLIPs required for ensemble statistics
    include_individual_results : bool, default=True
        Whether to include individual MLIP results alongside ensemble results
    name : str, default="StabilityBenchmark"
        Name of the benchmark
    description : str, optional
        Description of the benchmark
    metadata : Dict[str, Any], optional
        Additional metadata for the benchmark
    """

    def __init__(
        self,
        config: Union[str, Path, Dict[str, Any]] = None,
        use_ensemble: bool = None,
        mlip_names: Optional[List[str]] = None,
        metastable_threshold: float = None,
        min_mlips_required: int = None,
        include_individual_results: bool = True,
        name: str = "StabilityBenchmark",
        description: str = None,
        metadata: Dict[str, Any] = None,
    ):
        # Load configuration
        if config is not None:
            config_dict = load_config(config)

            # Override with config values if not explicitly provided
            use_ensemble = (
                use_ensemble
                if use_ensemble is not None
                else config_dict.get("use_ensemble", True)
            )
            mlip_names = mlip_names or config_dict.get(
                "mlip_names", ["orb", "mace", "uma"]
            )
            metastable_threshold = (
                metastable_threshold
                if metastable_threshold is not None
                else config_dict.get("metastable_threshold", 0.1)
            )

            # Extract ensemble config settings
            ensemble_config = config_dict.get("ensemble_config", {})
            min_mlips_required = (
                min_mlips_required
                if min_mlips_required is not None
                else ensemble_config.get("min_mlips_required", 2)
            )

            # Extract reporting config settings
            reporting_config = config_dict.get("reporting", {})
            include_individual_results = (
                include_individual_results
                if include_individual_results is not None
                else reporting_config.get("include_individual_mlip_results", True)
            )

        # Set defaults if still None
        use_ensemble = use_ensemble if use_ensemble is not None else True
        mlip_names = mlip_names or ["orb", "mace", "uma"]
        metastable_threshold = (
            metastable_threshold if metastable_threshold is not None else 0.1
        )
        min_mlips_required = min_mlips_required if min_mlips_required is not None else 2
        include_individual_results = (
            include_individual_results
            if include_individual_results is not None
            else True
        )

        # Store configuration
        self.use_ensemble = use_ensemble
        self.mlip_names = mlip_names
        self.metastable_threshold = metastable_threshold
        self.min_mlips_required = min_mlips_required
        self.include_individual_results = include_individual_results

        # Set description
        if description is None:
            mode = "ensemble" if use_ensemble else "individual"
            description = (
                f"Evaluates the thermodynamic and structural stability of crystal "
                f"structures using {mode} predictions from multiple MLIPs "
                f"({', '.join(mlip_names)})."
            )

        # Create evaluator configurations
        evaluator_configs = self._create_evaluator_configs()

        # Create benchmark metadata
        benchmark_metadata = {
            "version": "0.1.0",
            "category": "stability",
            "use_ensemble": use_ensemble,
            "mlip_names": self.mlip_names,
            "metastable_threshold": metastable_threshold,
            "min_mlips_required": min_mlips_required,
            "include_individual_results": include_individual_results,
            **(metadata or {}),
        }

        super().__init__(
            name=name,
            description=description,
            evaluator_configs=evaluator_configs,
            metadata=benchmark_metadata,
        )

    def _create_evaluator_configs(self) -> Dict[str, EvaluatorConfig]:
        """Create evaluator configurations for all stability metrics."""
        evaluator_configs = {}

        # Common parameters for all metrics
        metric_params = {
            "use_ensemble": self.use_ensemble,
            "mlip_names": self.mlip_names,
            "min_mlips_required": self.min_mlips_required,
            "include_individual_results": self.include_individual_results,
        }

        # Stability evaluator
        stability_metric = StabilityMetric(**metric_params)
        evaluator_configs["stability"] = EvaluatorConfig(
            name="Stability Analysis",
            description="Evaluates stability from multi-MLIP e_above_hull predictions",
            metrics={"stability": stability_metric},
            weights={"stability": 1.0},
            aggregation_method="weighted_mean",
        )

        # Metastability evaluator
        metastability_metric = MetastabilityMetric(
            metastable_threshold=self.metastable_threshold, **metric_params
        )
        evaluator_configs["metastability"] = EvaluatorConfig(
            name="Metastability Analysis",
            description="Evaluates metastability from multi-MLIP e_above_hull predictions",
            metrics={"metastability": metastability_metric},
            weights={"metastability": 1.0},
            aggregation_method="weighted_mean",
        )

        # Mean E_hull evaluator
        e_hull_metric = E_HullMetric(**metric_params)
        evaluator_configs["mean_e_above_hull"] = EvaluatorConfig(
            name="Mean Energy Above Hull",
            description="Evaluates mean energy above hull from multi-MLIP predictions",
            metrics={"mean_e_above_hull": e_hull_metric},
            weights={"mean_e_above_hull": 1.0},
            aggregation_method="weighted_mean",
        )

        # Formation energy evaluator
        formation_energy_metric = FormationEnergyMetric(**metric_params)
        evaluator_configs["formation_energy"] = EvaluatorConfig(
            name="Formation Energy Analysis",
            description="Evaluates formation energy from multi-MLIP predictions",
            metrics={"formation_energy": formation_energy_metric},
            weights={"formation_energy": 1.0},
            aggregation_method="weighted_mean",
        )

        # Relaxation stability evaluator
        relaxation_stability_metric = RelaxationStabilityMetric(**metric_params)
        evaluator_configs["relaxation_stability"] = EvaluatorConfig(
            name="Relaxation Stability Analysis",
            description="Evaluates relaxation stability from multi-MLIP RMSE predictions",
            metrics={"relaxation_stability": relaxation_stability_metric},
            weights={"relaxation_stability": 1.0},
            aggregation_method="weighted_mean",
        )

        # Relaxed stability evaluator
        relaxed_stability_metric = RelaxedStabilityMetric(**metric_params)
        evaluator_configs["relaxed_stability"] = EvaluatorConfig(
            name="Relaxed Stability Analysis",
            description="Evaluates stability after relaxation from multi-MLIP relaxed_e_above_hull predictions",
            metrics={"relaxed_stability": relaxed_stability_metric},
            weights={"relaxed_stability": 1.0},
            aggregation_method="weighted_mean",
        )

        # Relaxed metastability evaluator
        relaxed_metastability_metric = RelaxedMetastabilityMetric(
            metastable_threshold=self.metastable_threshold, **metric_params
        )
        evaluator_configs["relaxed_metastability"] = EvaluatorConfig(
            name="Relaxed Metastability Analysis",
            description="Evaluates metastability after relaxation from multi-MLIP relaxed_e_above_hull predictions",
            metrics={"relaxed_metastability": relaxed_metastability_metric},
            weights={"relaxed_metastability": 1.0},
            aggregation_method="weighted_mean",
        )

        # Relaxed mean E_hull evaluator
        relaxed_e_hull_metric = RelaxedE_HullMetric(**metric_params)
        evaluator_configs["relaxed_mean_e_above_hull"] = EvaluatorConfig(
            name="Relaxed Mean Energy Above Hull",
            description="Evaluates mean energy above hull after relaxation from multi-MLIP predictions",
            metrics={"relaxed_mean_e_above_hull": relaxed_e_hull_metric},
            weights={"relaxed_mean_e_above_hull": 1.0},
            aggregation_method="weighted_mean",
        )

        # Relaxed formation energy evaluator
        relaxed_formation_energy_metric = RelaxedFormationEnergyMetric(**metric_params)
        evaluator_configs["relaxed_formation_energy"] = EvaluatorConfig(
            name="Relaxed Formation Energy Analysis",
            description="Evaluates formation energy after relaxation from multi-MLIP predictions",
            metrics={"relaxed_formation_energy": relaxed_formation_energy_metric},
            weights={"relaxed_formation_energy": 1.0},
            aggregation_method="weighted_mean",
        )

        return evaluator_configs

    def aggregate_evaluator_results(
        self, evaluator_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """Aggregate results from multiple evaluators into final scores.

        Parameters
        ----------
        evaluator_results : Dict[str, Dict[str, Any]]
            Results from each evaluator (as dictionaries)

        Returns
        -------
        Dict[str, float]
            Final aggregated scores containing primary metrics and optionally individual results
        """
        final_scores = {}

        # Extract primary metrics from each evaluator
        stability_results = evaluator_results.get("stability")
        if stability_results:
            final_scores["stable_ratio"] = safe_float(
                stability_results.get("combined_value")
            )

            # Extract additional metrics from stability results - access MetricResult.metrics
            stability_metric_result = stability_results.get("metric_results", {}).get(
                "stability"
            )
            if stability_metric_result:
                stability_metrics = stability_metric_result.metrics
                final_scores["stable_count"] = safe_int(  # Add count
                    stability_metrics.get("stable_count")
                )
                final_scores["stability_mean_e_above_hull"] = safe_float(
                    stability_metrics.get("mean_e_above_hull")
                )
                final_scores["stability_std_e_above_hull"] = safe_float(
                    stability_metrics.get("std_e_above_hull")
                )
                final_scores["stability_total_structures"] = safe_int(  # Add total count
                    stability_metrics.get("total_structures_evaluated")
                )

                # Add ensemble uncertainty if available
                if self.use_ensemble:
                    final_scores["stability_mean_ensemble_std"] = safe_float(
                        stability_metrics.get("mean_ensemble_std")
                    )

        metastability_results = evaluator_results.get("metastability")
        if metastability_results:
            final_scores["metastable_ratio"] = safe_float(
                metastability_results.get("combined_value")
            )

            # Extract metastability count
            metastability_metric_result = metastability_results.get(
                "metric_results", {}
            ).get("metastability")
            if metastability_metric_result:
                metastability_metrics = metastability_metric_result.metrics
                final_scores["metastable_count"] = safe_int(  # Add count
                    metastability_metrics.get("metastable_count")
                )
                final_scores["metastability_total_structures"] = safe_int(  # Add total count
                    metastability_metrics.get("total_structures_evaluated")
                )

        e_hull_results = evaluator_results.get("mean_e_above_hull")
        if e_hull_results:
            final_scores["mean_e_above_hull"] = safe_float(
                e_hull_results.get("combined_value")
            )

            e_hull_metric_result = e_hull_results.get("metric_results", {}).get(
                "mean_e_above_hull"
            )
            if e_hull_metric_result:
                e_hull_metrics = e_hull_metric_result.metrics
                final_scores["e_hull_std"] = safe_float(
                    e_hull_metrics.get("std_e_above_hull")
                )
                final_scores["e_hull_total_structures"] = safe_int(  # Add total count
                    e_hull_metrics.get("total_structures_evaluated")
                )

        formation_energy_results = evaluator_results.get("formation_energy")
        if formation_energy_results:
            final_scores["mean_formation_energy"] = safe_float(
                formation_energy_results.get("combined_value")
            )

            fe_metric_result = formation_energy_results.get("metric_results", {}).get(
                "formation_energy"
            )
            if fe_metric_result:
                fe_metrics = fe_metric_result.metrics
                final_scores["formation_energy_std"] = safe_float(
                    fe_metrics.get("std_formation_energy")
                )
                final_scores["formation_energy_total_structures"] = safe_int(  # Add total count
                    fe_metrics.get("total_structures_evaluated")
                )

        relaxation_stability_results = evaluator_results.get("relaxation_stability")
        if relaxation_stability_results:
            final_scores["mean_relaxation_RMSE"] = safe_float(
                relaxation_stability_results.get("combined_value")
            )

            rs_metric_result = relaxation_stability_results.get(
                "metric_results", {}
            ).get("relaxation_stability")
            if rs_metric_result:
                rs_metrics = rs_metric_result.metrics
                final_scores["relaxation_RMSE_std"] = safe_float(
                    rs_metrics.get("std_relaxation_RMSE")
                )
                final_scores["relaxation_stability_total_structures"] = safe_int(  # Add total count
                    rs_metrics.get("total_structures_evaluated")
                )

        # Relaxed stability results
        relaxed_stability_results = evaluator_results.get("relaxed_stability")
        if relaxed_stability_results:
            final_scores["relaxed_stable_ratio"] = safe_float(
                relaxed_stability_results.get("combined_value")
            )

            # Extract additional metrics from relaxed stability results
            relaxed_stability_metric_result = relaxed_stability_results.get("metric_results", {}).get(
                "relaxed_stability"
            )
            if relaxed_stability_metric_result:
                relaxed_stability_metrics = relaxed_stability_metric_result.metrics
                final_scores["relaxed_stable_count"] = safe_int(
                    relaxed_stability_metrics.get("stable_count")
                )
                final_scores["relaxed_stability_mean_e_above_hull"] = safe_float(
                    relaxed_stability_metrics.get("mean_e_above_hull")
                )
                final_scores["relaxed_stability_std_e_above_hull"] = safe_float(
                    relaxed_stability_metrics.get("std_e_above_hull")
                )
                final_scores["relaxed_stability_total_structures"] = safe_int(
                    relaxed_stability_metrics.get("total_structures_evaluated")
                )

                # Add ensemble uncertainty if available
                if self.use_ensemble:
                    final_scores["relaxed_stability_mean_ensemble_std"] = safe_float(
                        relaxed_stability_metrics.get("mean_ensemble_std")
                    )

        # Relaxed metastability results
        relaxed_metastability_results = evaluator_results.get("relaxed_metastability")
        if relaxed_metastability_results:
            final_scores["relaxed_metastable_ratio"] = safe_float(
                relaxed_metastability_results.get("combined_value")
            )

            # Extract relaxed metastability count
            relaxed_metastability_metric_result = relaxed_metastability_results.get(
                "metric_results", {}
            ).get("relaxed_metastability")
            if relaxed_metastability_metric_result:
                relaxed_metastability_metrics = relaxed_metastability_metric_result.metrics
                final_scores["relaxed_metastable_count"] = safe_int(
                    relaxed_metastability_metrics.get("metastable_count")
                )
                final_scores["relaxed_metastability_total_structures"] = safe_int(
                    relaxed_metastability_metrics.get("total_structures_evaluated")
                )

        # Relaxed mean E_hull results
        relaxed_e_hull_results = evaluator_results.get("relaxed_mean_e_above_hull")
        if relaxed_e_hull_results:
            final_scores["relaxed_mean_e_above_hull"] = safe_float(
                relaxed_e_hull_results.get("combined_value")
            )

            relaxed_e_hull_metric_result = relaxed_e_hull_results.get("metric_results", {}).get(
                "relaxed_mean_e_above_hull"
            )
            if relaxed_e_hull_metric_result:
                relaxed_e_hull_metrics = relaxed_e_hull_metric_result.metrics
                final_scores["relaxed_e_hull_std"] = safe_float(
                    relaxed_e_hull_metrics.get("std_e_above_hull")
                )
                final_scores["relaxed_e_hull_total_structures"] = safe_int(
                    relaxed_e_hull_metrics.get("total_structures_evaluated")
                )

        # Relaxed formation energy results
        relaxed_formation_energy_results = evaluator_results.get("relaxed_formation_energy")
        if relaxed_formation_energy_results:
            final_scores["relaxed_mean_formation_energy"] = safe_float(
                relaxed_formation_energy_results.get("combined_value")
            )

            relaxed_fe_metric_result = relaxed_formation_energy_results.get("metric_results", {}).get(
                "relaxed_formation_energy"
            )
            if relaxed_fe_metric_result:
                relaxed_fe_metrics = relaxed_fe_metric_result.metrics
                final_scores["relaxed_formation_energy_std"] = safe_float(
                    relaxed_fe_metrics.get("std_formation_energy")
                )
                final_scores["relaxed_formation_energy_total_structures"] = safe_int(
                    relaxed_fe_metrics.get("total_structures_evaluated")
                )

        # Add individual MLIP results if requested
        if self.include_individual_results:
            for evaluator_name, result in evaluator_results.items():
                if result:
                    metric_result = result.get("metric_results", {}).get(evaluator_name)
                    if metric_result:
                        metrics = metric_result.metrics

                        # Extract individual MLIP metrics including counts
                        for mlip_name in self.mlip_names:
                            for metric_key, metric_value in metrics.items():
                                if metric_key.endswith(f"_{mlip_name}"):
                                    # Handle both float and int values appropriately
                                    if "count" in metric_key:
                                        final_scores[f"{evaluator_name}_{metric_key}"] = (
                                            safe_int(metric_value)
                                        )
                                    else:
                                        final_scores[f"{evaluator_name}_{metric_key}"] = (
                                            safe_float(metric_value)
                                        )

        return final_scores


# Factory functions for easy creation
def create_benchmark_from_config(config_path: Union[str, Path]) -> StabilityBenchmark:
    """Create stability benchmark from YAML configuration file.

    Parameters
    ----------
    config_path : Union[str, Path]
        Path to YAML configuration file

    Returns
    -------
    StabilityBenchmark
        Configured stability benchmark
    """
    return StabilityBenchmark(config=config_path)


def create_ensemble_stability_benchmark(**kwargs) -> StabilityBenchmark:
    """Create stability benchmark using ensemble predictions.

    Parameters
    ----------
    **kwargs
        Additional parameters for StabilityBenchmark initialization

    Returns
    -------
    StabilityBenchmark
        Configured ensemble stability benchmark
    """
    return StabilityBenchmark(use_ensemble=True, **kwargs)


def create_individual_mlip_stability_benchmark(
    mlip_names: List[str], **kwargs
) -> StabilityBenchmark:
    """Create stability benchmark using individual MLIP predictions.

    Parameters
    ----------
    mlip_names : List[str]
        Names of MLIPs to use for evaluation
    **kwargs
        Additional parameters for StabilityBenchmark initialization

    Returns
    -------
    StabilityBenchmark
        Configured individual MLIP stability benchmark
    """
    return StabilityBenchmark(use_ensemble=False, mlip_names=mlip_names, **kwargs)


def create_comprehensive_benchmark(**kwargs) -> StabilityBenchmark:
    """Create benchmark with both ensemble and individual results.

    Parameters
    ----------
    **kwargs
        Additional parameters for StabilityBenchmark initialization

    Returns
    -------
    StabilityBenchmark
        Comprehensive benchmark showing both ensemble and individual results
    """
    return StabilityBenchmark(
        use_ensemble=True, include_individual_results=True, **kwargs
    )


# Example usage functions
def example_config_based_evaluation():
    """Example of using the benchmark with YAML configuration."""
    # Load benchmark from config file
    config_path = "src/config/multi_mlip_stability.yaml"
    benchmark = create_benchmark_from_config(config_path)

    print("Benchmark created with config:")
    print(f"  - use_ensemble: {benchmark.use_ensemble}")
    print(f"  - mlip_names: {benchmark.mlip_names}")
    print(f"  - metastable_threshold: {benchmark.metastable_threshold}")
    print(f"  - min_mlips_required: {benchmark.min_mlips_required}")
    print(f"  - include_individual_results: {benchmark.include_individual_results}")

    # Usage:
    # structures = load_preprocessed_structures()  # Must have multi-MLIP properties
    # result = benchmark.evaluate(structures)
    #
    # Access results:
    # stable_ratio = result.final_scores["stable_ratio"]
    # stable_count = result.final_scores["stable_count"]  # NEW: count of stable structures
    # metastable_count = result.final_scores["metastable_count"]  # NEW: count of metastable structures
    # mean_e_hull = result.final_scores["mean_e_above_hull"]
    # e_hull_std = result.final_scores["e_hull_std"]
    #
    # If include_individual_results=True:
    # orb_stable_ratio = result.final_scores["stability_stable_ratio_orb"]
    # orb_stable_count = result.final_scores["stability_stable_count_orb"]  # NEW: per-MLIP counts

    return benchmark


def example_individual_vs_ensemble():
    """Example comparing individual and ensemble modes."""
    # Individual mode - shows per-MLIP results
    individual_benchmark = create_individual_mlip_stability_benchmark(
        mlip_names=["orb", "mace", "uma"], name="Individual MLIP Benchmark"
    )

    # Ensemble mode - shows ensemble statistics
    ensemble_benchmark = create_ensemble_stability_benchmark(name="Ensemble Benchmark")

    # Comprehensive mode - shows both ensemble and individual
    comprehensive_benchmark = create_comprehensive_benchmark(
        name="Comprehensive Benchmark"
    )

    print("Created three benchmark configurations:")
    print(
        f"Individual: {individual_benchmark.use_ensemble=}, {individual_benchmark.include_individual_results=}"
    )
    print(
        f"Ensemble: {ensemble_benchmark.use_ensemble=}, {ensemble_benchmark.include_individual_results=}"
    )
    print(
        f"Comprehensive: {comprehensive_benchmark.use_ensemble=}, {comprehensive_benchmark.include_individual_results=}"
    )

    return individual_benchmark, ensemble_benchmark, comprehensive_benchmark


# Test runner for development
if __name__ == "__main__":
    """Test the multi-MLIP stability benchmark implementation."""
    print("Testing multi-MLIP stability benchmark...")

    try:
        # Test basic initialization
        benchmark = StabilityBenchmark()
        print("✓ Basic initialization successful")

        # Test with custom configuration
        custom_benchmark = StabilityBenchmark(
            use_ensemble=False,
            mlip_names=["orb", "mace"],
            metastable_threshold=0.05,
            include_individual_results=True,
        )
        print("✓ Custom configuration successful")

        # Test factory functions
        ensemble_bench = create_ensemble_stability_benchmark()
        individual_bench = create_individual_mlip_stability_benchmark(["orb", "mace"])
        comprehensive_bench = create_comprehensive_benchmark()

        print("✓ All factory functions working")

        # Test that we have the expected evaluators
        expected_evaluators = [
            "stability",
            "metastability",
            "mean_e_above_hull",
            "formation_energy",
            "relaxation_stability",
        ]

        for eval_name in expected_evaluators:
            assert eval_name in benchmark.evaluators

        print("✓ All evaluators configured correctly")

        # Test that metrics have the right configuration
        stability_metric = benchmark.evaluators["stability"].metrics["stability"]
        assert stability_metric.use_ensemble == benchmark.use_ensemble
        assert stability_metric.mlip_names == benchmark.mlip_names
        assert stability_metric.min_mlips_required == benchmark.min_mlips_required
        assert (
            stability_metric.include_individual_results
            == benchmark.include_individual_results
        )

        print("✓ Metrics configured with benchmark settings")

        # Test metadata structure
        metadata = benchmark.config.metadata
        required_fields = [
            "version",
            "category",
            "use_ensemble",
            "mlip_names",
            "metastable_threshold",
            "min_mlips_required",
            "include_individual_results",
        ]
        for field in required_fields:
            assert field in metadata

        print("✓ Metadata structure correct")

        # Test example functions
        example_config_based_evaluation()
        example_individual_vs_ensemble()

        print("✓ Example functions working")

        print("\n🎉 Multi-MLIP stability benchmark implemented successfully!")
        print("\nKey features:")
        print("  ✓ Supports both individual and ensemble modes")
        print("  ✓ Configurable individual MLIP result inclusion")
        print("  ✓ Uses all config sections meaningfully")
        print("  ✓ Reports standard deviations at structure and sample levels")
        print("  ✓ Reports structure counts alongside ratios")  # NEW
        print("  ✓ Comprehensive factory functions")
        print("  ✓ Config file and programmatic configuration support")

        print("\nUsage:")
        print("  1. Create from config: create_benchmark_from_config('config.yaml')")
        print("  2. Ensemble mode: create_ensemble_stability_benchmark()")
        print(
            "  3. Individual mode: create_individual_mlip_stability_benchmark(['orb', 'mace'])"
        )
        print("  4. Comprehensive: create_comprehensive_benchmark() (shows both)")
        print("  5. Run benchmark.evaluate(structures) on preprocessed structures")
        print("  6. Access metrics from result.final_scores (includes counts and std deviations)")  # UPDATED

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()


# Alias for backward compatibility and expected import name
MultiMLIPStabilityBenchmark = StabilityBenchmark