"""New SUN (Stable, Unique, Novel) benchmark using augmented fingerprinting for material structures.

This module implements a benchmark that evaluates the proportion of
generated material structures that are simultaneously stable, unique,
and novel using the new augmented fingerprinting approach with improved
reference datasets and fingerprint handling.
"""

from typing import Any, Dict, Optional

import numpy as np

from lemat_genbench.benchmarks.base import BaseBenchmark
from lemat_genbench.evaluator import EvaluationResult, EvaluatorConfig
from lemat_genbench.metrics.sun_new_metric import MetaSUNNewMetric, SUNNewMetric


class SUNNewBenchmark(BaseBenchmark):
    """Benchmark for evaluating SUN (Stable, Unique, Novel) rate using augmented fingerprinting.

    This benchmark uses the SUNNewMetric to evaluate the proportion of generated
    structures that are simultaneously:
    1. Stable (e_above_hull <= stability_threshold)
    2. Unique (not duplicated within the generated set using augmented fingerprints)
    3. Novel (not present in reference dataset using augmented fingerprints)

    The benchmark also evaluates MetaSUN rate for metastable structures.
    """

    def __init__(
        self,
        stability_threshold: float = 0.0,
        metastability_threshold: float = 0.1,
        reference_fingerprints_path: Optional[str] = None,
        reference_dataset_name: str = "LeMat-Bulk",
        fingerprint_source: str = "auto",
        symprec: float = 0.01,
        angle_tolerance: float = 5.0,
        fallback_to_computation: bool = True,
        include_metasun: bool = True,
        name: str = "SUNNewBenchmark",
        description: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ):
        """Initialize the new SUN benchmark.

        Parameters
        ----------
        stability_threshold : float, default=0.0
            Energy above hull threshold for stability (eV/atom).
        metastability_threshold : float, default=0.1
            Energy above hull threshold for metastability (eV/atom).
        reference_fingerprints_path : str or None, default=None
            Path to file containing reference fingerprints. If None, uses default path.
        reference_dataset_name : str, default="LeMat-Bulk"
            Name of the reference dataset for logging purposes.
        fingerprint_source : str, default="auto"
            Source of fingerprints: "property", "compute", or "auto".
        symprec : float, default=0.01
            Symmetry precision for fingerprint computation.
        angle_tolerance : float, default=5.0
            Angle tolerance for fingerprint computation.
        fallback_to_computation : bool, default=True
            Whether to compute fingerprints if not found in structure properties.
        include_metasun : bool, default=True
            Whether to include MetaSUN evaluation alongside SUN.
        name : str
            Name of the benchmark.
        description : str, optional
            Description of the benchmark.
        metadata : dict, optional
            Additional metadata for the benchmark.
        """
        if description is None:
            description = (
                "Enhanced SUN benchmark that evaluates the proportion of crystal structures "
                "that are simultaneously Stable, Unique, and Novel using improved augmented "
                "fingerprinting for enhanced accuracy and robustness in structural comparison."
            )

        # Initialize the main SUN metric with augmented fingerprinting
        sun_metric = SUNNewMetric(
            stability_threshold=stability_threshold,
            metastability_threshold=metastability_threshold,
            reference_fingerprints_path=reference_fingerprints_path,
            reference_dataset_name=reference_dataset_name,
            fingerprint_source=fingerprint_source,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
            fallback_to_computation=fallback_to_computation,
        )

        # Set up evaluator configs
        evaluator_configs = {
            "sun": EvaluatorConfig(
                name="sun",
                description=("Evaluates structures that are Stable, Unique, and Novel using augmented fingerprinting"),
                metrics={"sun": sun_metric},
                weights={"sun": 1.0},
                aggregation_method="weighted_mean",
            ),
        }

        # Add MetaSUN evaluator if requested
        if include_metasun:
            metasun_metric = MetaSUNNewMetric(
                metastability_threshold=metastability_threshold,
                reference_fingerprints_path=reference_fingerprints_path,
                reference_dataset_name=reference_dataset_name,
                fingerprint_source=fingerprint_source,
                symprec=symprec,
                angle_tolerance=angle_tolerance,
                fallback_to_computation=fallback_to_computation,
            )

            evaluator_configs["metasun"] = EvaluatorConfig(
                name="metasun",
                description=(
                    "Evaluates structures that are Metastable, Unique, and Novel using augmented fingerprinting"
                ),
                metrics={"metasun": metasun_metric},
                weights={"metasun": 1.0},
                aggregation_method="weighted_mean",
            )

        # Create benchmark metadata
        benchmark_metadata = {
            "version": "0.2.0",
            "category": "sun",
            "fingerprinting_method": "augmented",
            "stability_threshold": stability_threshold,
            "metastability_threshold": metastability_threshold,
            "reference_dataset_name": reference_dataset_name,
            "reference_fingerprints_path": reference_fingerprints_path,
            "fingerprint_source": fingerprint_source,
            "symprec": symprec,
            "angle_tolerance": angle_tolerance,
            "fallback_to_computation": fallback_to_computation,
            "include_metasun": include_metasun,
            **(metadata or {}),
        }

        super().__init__(
            name=name,
            description=description,
            evaluator_configs=evaluator_configs,
            metadata=benchmark_metadata,
        )

    def aggregate_evaluator_results(
        self, evaluator_results: Dict[str, EvaluationResult]
    ) -> Dict[str, float]:
        """Aggregate results from SUN evaluators into final scores.

        Parameters
        ----------
        evaluator_results : dict[str, EvaluationResult]
            Results from each evaluator.

        Returns
        -------
        dict[str, float]
            Final aggregated scores containing SUN metrics.
        """
        # Initialize default scores
        final_scores = {
            "sun_rate": np.nan,
            "sun_count": 0,
            "msun_rate": np.nan,
            "msun_count": 0,
            "combined_sun_msun_rate": np.nan,
            "unique_count": 0,
            "unique_rate": np.nan,
            "total_structures_evaluated": 0,
            "failed_count": 0,
        }

        # Extract SUN results
        sun_results = evaluator_results.get("sun")
        if sun_results:
            # Get the combined score (should be same as sun_rate)
            combined_value = sun_results.get("combined_value")
            if combined_value is not None:
                final_scores["sun_rate"] = float(combined_value)

            # Extract detailed metrics from the metric results
            metric_results = sun_results.get("metric_results", {})
            sun_metric_result = metric_results.get("sun", {})

            if hasattr(sun_metric_result, "metrics"):
                metrics = sun_metric_result.metrics
            elif isinstance(sun_metric_result, dict):
                metrics = sun_metric_result.get("metrics", {})
            else:
                metrics = {}

            # Extract detailed SUN metrics
            for key in [
                "sun_count",
                "msun_count",
                "combined_sun_msun_rate",
                "unique_count",
                "unique_rate",
                "total_structures_evaluated",
                "failed_count",
                "msun_rate",
            ]:
                if key in metrics:
                    final_scores[key] = metrics[key]

        # Extract MetaSUN results if available
        metasun_results = evaluator_results.get("metasun")
        if metasun_results:
            # Get the combined score for MetaSUN
            metasun_combined_value = metasun_results.get("combined_value")
            if metasun_combined_value is not None:
                final_scores["metasun_rate"] = float(metasun_combined_value)

            # Extract detailed MetaSUN metrics
            metric_results = metasun_results.get("metric_results", {})
            metasun_metric_result = metric_results.get("metasun", {})

            if hasattr(metasun_metric_result, "metrics"):
                metasun_metrics = metasun_metric_result.metrics
            elif isinstance(metasun_metric_result, dict):
                metasun_metrics = metasun_metric_result.get("metrics", {})
            else:
                metasun_metrics = {}

            # Add MetaSUN specific metrics with prefix
            for key, value in metasun_metrics.items():
                if key.startswith("sun_"):
                    # Replace sun_ prefix with metasun_
                    metasun_key = key.replace("sun_", "metasun_", 1)
                    final_scores[metasun_key] = value

        return final_scores


# Factory functions for common configurations
def create_sun_new_benchmark(
    reference_fingerprints_path: Optional[str] = None,
    fingerprint_source: str = "auto",
    include_metasun: bool = True,
    **kwargs
) -> SUNNewBenchmark:
    """Factory function to create SUN benchmark with new augmented fingerprinting.

    Parameters
    ----------
    reference_fingerprints_path : str or None, default=None
        Path to reference fingerprints file.
    fingerprint_source : str, default="auto"
        Source of fingerprints.
    include_metasun : bool, default=True
        Whether to include MetaSUN evaluation.
    **kwargs
        Additional arguments for the benchmark.

    Returns
    -------
    SUNNewBenchmark
        Configured SUN benchmark with augmented fingerprinting.
    """
    return SUNNewBenchmark(
        reference_fingerprints_path=reference_fingerprints_path,
        fingerprint_source=fingerprint_source,
        include_metasun=include_metasun,
        **kwargs,
    )


def create_property_based_sun_benchmark(**kwargs) -> SUNNewBenchmark:
    """Create benchmark that only uses preprocessed fingerprints from properties."""
    return create_sun_new_benchmark(
        fingerprint_source="property",
        fallback_to_computation=False,
        name="PropertyBasedSUNBenchmark",
        description="SUN benchmark using only preprocessed augmented fingerprints",
        **kwargs
    )


def create_computation_based_sun_benchmark(**kwargs) -> SUNNewBenchmark:
    """Create benchmark that computes fingerprints on-demand."""
    return create_sun_new_benchmark(
        fingerprint_source="compute",
        name="ComputationBasedSUNBenchmark", 
        description="SUN benchmark that computes augmented fingerprints on-demand",
        **kwargs
    )


def create_robust_sun_benchmark(**kwargs) -> SUNNewBenchmark:
    """Create benchmark with robust settings for most use cases."""
    return create_sun_new_benchmark(
        fingerprint_source="auto",
        symprec=0.1,
        angle_tolerance=10.0,
        fallback_to_computation=True,
        name="RobustSUNBenchmark",
        description="Robust SUN benchmark with fallback computation and relaxed parameters",
        **kwargs
    )


def create_high_precision_sun_benchmark(**kwargs) -> SUNNewBenchmark:
    """Create benchmark with high precision settings."""
    return create_sun_new_benchmark(
        fingerprint_source="auto",
        symprec=0.001,
        angle_tolerance=1.0,
        fallback_to_computation=True,
        name="HighPrecisionSUNBenchmark",
        description="High precision SUN benchmark with strict tolerances",
        **kwargs
    )