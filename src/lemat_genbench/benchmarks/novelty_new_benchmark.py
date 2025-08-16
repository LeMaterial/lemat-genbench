"""Enhanced novelty benchmark using augmented fingerprinting for material structures.

This module implements an enhanced benchmark that evaluates the novelty of
generated material structures using the new augmented fingerprinting approach
with improved reference datasets and fingerprint handling.
"""

from typing import Any, Dict, Optional

import numpy as np

from lemat_genbench.benchmarks.base import BaseBenchmark
from lemat_genbench.evaluator import EvaluationResult, EvaluatorConfig
from lemat_genbench.metrics.novelty_new_metric import AugmentedNoveltyMetric


class AugmentedNoveltyBenchmark(BaseBenchmark):
    """Enhanced benchmark for evaluating novelty using augmented fingerprinting.

    This benchmark uses the AugmentedNoveltyMetric to compare generated structures
    against reference datasets using the improved augmented fingerprinting approach.
    It supports both preprocessed fingerprints and on-demand computation.
    """

    def __init__(
        self,
        reference_fingerprints_path: Optional[str] = None,
        reference_dataset_name: str = "LeMat-Bulk",
        fingerprint_source: str = "auto",
        symprec: float = 0.01,
        angle_tolerance: float = 5.0,
        fallback_to_computation: bool = True,
        name: str = "AugmentedNoveltyBenchmark",
        description: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ):
        """Initialize the enhanced novelty benchmark.

        Parameters
        ----------
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
        name : str
            Name of the benchmark.
        description : str, optional
            Description of the benchmark.
        metadata : dict, optional
            Additional metadata for the benchmark.
        """
        if description is None:
            description = (
                "Enhanced novelty benchmark that evaluates crystal structures using "
                "improved augmented fingerprinting to identify truly novel materials "
                "compared to reference datasets with enhanced accuracy and robustness."
            )

        # Initialize the augmented novelty metric
        novelty_metric = AugmentedNoveltyMetric(
            reference_fingerprints_path=reference_fingerprints_path,
            reference_dataset_name=reference_dataset_name,
            fingerprint_source=fingerprint_source,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
            fallback_to_computation=fallback_to_computation,
        )

        # Set up evaluator configs
        evaluator_configs = {
            "augmented_novelty": EvaluatorConfig(
                name="augmented_novelty",
                description=(
                    "Evaluates structural novelty using enhanced augmented fingerprinting"
                ),
                metrics={"augmented_novelty": novelty_metric},
                weights={"augmented_novelty": 1.0},
                aggregation_method="weighted_mean",
            ),
        }

        # Create benchmark metadata
        benchmark_metadata = {
            "version": "0.2.0",
            "category": "novelty",
            "fingerprinting_method": "augmented",
            "reference_dataset": reference_dataset_name,
            "reference_fingerprints_path": reference_fingerprints_path,
            "fingerprint_source": fingerprint_source,
            "symprec": symprec,
            "angle_tolerance": angle_tolerance,
            "fallback_to_computation": fallback_to_computation,
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
        """Aggregate results from the augmented novelty evaluator into final scores.

        Parameters
        ----------
        evaluator_results : dict[str, EvaluationResult]
            Results from each evaluator.

        Returns
        -------
        dict[str, float]
            Final aggregated scores.
        """
        # Initialize default scores
        final_scores = {
            "novelty_score": np.nan,
            "novel_structures_count": 0,
            "total_structures_evaluated": 0,
            "total_structures_attempted": 0,
            "fingerprinting_success_rate": 0.0,
            "novelty_ratio": np.nan,  # Alias for novelty_score for clarity
        }

        # Extract augmented novelty results
        novelty_results = evaluator_results.get("augmented_novelty")
        if novelty_results:
            # Get the combined score (should be same as novelty_score)
            combined_value = novelty_results.get("combined_value")
            if combined_value is not None:
                final_scores["novelty_score"] = float(combined_value)
                final_scores["novelty_ratio"] = float(combined_value)

            # Extract detailed metrics from the metric results
            metric_results = novelty_results.get("metric_results", {})
            novelty_metric_result = metric_results.get("augmented_novelty", {})

            if hasattr(novelty_metric_result, "metrics"):
                metrics = novelty_metric_result.metrics
            elif isinstance(novelty_metric_result, dict):
                metrics = novelty_metric_result.get("metrics", {})
            else:
                metrics = {}

            # Extract count and success rate information
            final_scores["novel_structures_count"] = metrics.get(
                "novel_structures_count", 0
            )
            final_scores["total_structures_evaluated"] = metrics.get(
                "total_structures_evaluated", 0
            )
            final_scores["total_structures_attempted"] = metrics.get(
                "total_structures_attempted", 0
            )
            final_scores["fingerprinting_success_rate"] = metrics.get(
                "fingerprinting_success_rate", 0.0
            )

        return final_scores


# Factory functions for common configurations
def create_augmented_novelty_benchmark(
    reference_fingerprints_path: Optional[str] = None,
    fingerprint_source: str = "auto",
    **kwargs
) -> AugmentedNoveltyBenchmark:
    """Factory function to create augmented novelty benchmark with common configurations.

    Parameters
    ----------
    reference_fingerprints_path : str or None, default=None
        Path to reference fingerprints file.
    fingerprint_source : str, default="auto"
        Source of fingerprints.
    **kwargs
        Additional arguments for the benchmark.

    Returns
    -------
    AugmentedNoveltyBenchmark
        Configured augmented novelty benchmark.
    """
    return AugmentedNoveltyBenchmark(
        reference_fingerprints_path=reference_fingerprints_path,
        fingerprint_source=fingerprint_source,
        **kwargs,
    )


def create_property_based_novelty_benchmark(**kwargs) -> AugmentedNoveltyBenchmark:
    """Create benchmark that only uses preprocessed fingerprints from properties."""
    return create_augmented_novelty_benchmark(
        fingerprint_source="property",
        fallback_to_computation=False,
        name="PropertyBasedNoveltyBenchmark",
        description="Novelty benchmark using only preprocessed augmented fingerprints",
        **kwargs
    )


def create_computation_based_novelty_benchmark(**kwargs) -> AugmentedNoveltyBenchmark:
    """Create benchmark that computes fingerprints on-demand."""
    return create_augmented_novelty_benchmark(
        fingerprint_source="compute",
        name="ComputationBasedNoveltyBenchmark", 
        description="Novelty benchmark that computes augmented fingerprints on-demand",
        **kwargs
    )


def create_robust_novelty_benchmark(**kwargs) -> AugmentedNoveltyBenchmark:
    """Create benchmark with robust settings for most use cases."""
    return create_augmented_novelty_benchmark(
        fingerprint_source="auto",
        symprec=0.1,
        angle_tolerance=10.0,
        fallback_to_computation=True,
        name="RobustNoveltyBenchmark",
        description="Robust novelty benchmark with fallback computation and relaxed parameters",
        **kwargs
    )


def create_high_precision_novelty_benchmark(**kwargs) -> AugmentedNoveltyBenchmark:
    """Create benchmark with high precision settings."""
    return create_augmented_novelty_benchmark(
        fingerprint_source="auto",
        symprec=0.001,
        angle_tolerance=1.0,
        fallback_to_computation=True,
        name="HighPrecisionNoveltyBenchmark",
        description="High precision novelty benchmark with strict tolerances",
        **kwargs
    )