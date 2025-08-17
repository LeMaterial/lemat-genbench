"""New uniqueness benchmark for material structures using augmented fingerprints.

This module implements a benchmark that evaluates the uniqueness of
generated material structures by measuring the fraction of unique
structures within the generated set using augmented structure fingerprinting.
"""

from typing import Any, Dict

import numpy as np

from lemat_genbench.benchmarks.base import BaseBenchmark
from lemat_genbench.evaluator import EvaluationResult, EvaluatorConfig
from lemat_genbench.metrics.uniqueness_new_metric import UniquenessNewMetric


class UniquenessNewBenchmark(BaseBenchmark):
    """Benchmark for evaluating the uniqueness of generated material structures 
    using augmented fingerprints.

    This benchmark uses the UniquenessNewMetric to measure the fraction of 
    unique structures within a generated set, detecting duplicates using 
    augmented structure fingerprinting to assess the diversity of generated 
    materials.
    """

    def __init__(
        self,
        fingerprint_source: str = "auto",
        symprec: float = 0.01,
        angle_tolerance: float = 5.0,
        name: str = "UniquenessNewBenchmark",
        description: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ):
        """Initialize the new uniqueness benchmark.

        Parameters
        ----------
        fingerprint_source : str, default="auto"
            Source for structure fingerprints. Options:
            - "auto": Use preprocessed fingerprint if available, compute if not
            - "compute": Always compute fingerprints from structures
            - "property": Only use preprocessed fingerprints from structure 
               properties
        symprec : float, default=0.01
            Symmetry precision for crystallographic analysis when computing fingerprints.
        angle_tolerance : float, default=5.0
            Angle tolerance in degrees for crystallographic analysis when computing fingerprints.
        name : str
            Name of the benchmark.
        description : str, optional
            Description of the benchmark.
        metadata : dict, optional
            Additional metadata for the benchmark.
        """
        if description is None:
            description = (
                "Evaluates the uniqueness of crystal structures within a "
                "generated set by measuring the fraction of unique structures "
                "using augmented structure fingerprinting to detect duplicates and "
                "assess diversity."
            )

        # Initialize the uniqueness metric
        uniqueness_metric = UniquenessNewMetric(
            fingerprint_source=fingerprint_source,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        )

        # Set up evaluator configs
        evaluator_configs = {
            "uniqueness": EvaluatorConfig(
                name="uniqueness",
                description=("Evaluates structural uniqueness within generated set using augmented fingerprints"),
                metrics={"uniqueness": uniqueness_metric},
                weights={"uniqueness": 1.0},
                aggregation_method="weighted_mean",
            ),
        }

        # Create benchmark metadata
        benchmark_metadata = {
            "version": "0.1.0",
            "category": "uniqueness",
            "fingerprint_method": "augmented",
            "fingerprint_source": fingerprint_source,
            "symprec": symprec,
            "angle_tolerance": angle_tolerance,
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
        """Aggregate results from the uniqueness evaluator into final scores.

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
            "uniqueness_score": np.nan,
            "unique_structures_count": 0,
            "duplicate_structures_count": 0,
            "total_structures_evaluated": 0,
            "failed_fingerprinting_count": 0,
            "uniqueness_ratio": np.nan,  # Alias for uniqueness_score for clarity
        }

        # Extract uniqueness results
        uniqueness_results = evaluator_results.get("uniqueness")
        if uniqueness_results:
            # Get the combined score (should be same as uniqueness_score)
            combined_value = uniqueness_results.get("combined_value")
            if combined_value is not None:
                final_scores["uniqueness_score"] = float(combined_value)
                final_scores["uniqueness_ratio"] = float(combined_value)
            # If combined_value is None, keep the default np.nan values

            # Extract detailed metrics from the metric results
            metric_results = uniqueness_results.get("metric_results", {})
            uniqueness_metric_result = metric_results.get("uniqueness", {})

            if hasattr(uniqueness_metric_result, "metrics"):
                metrics = uniqueness_metric_result.metrics
            elif isinstance(uniqueness_metric_result, dict):
                metrics = uniqueness_metric_result.get("metrics", {})
            else:
                metrics = {}

            # Extract count information
            final_scores["unique_structures_count"] = metrics.get(
                "unique_structures_count", 0
            )
            final_scores["duplicate_structures_count"] = metrics.get(
                "duplicate_structures_count", 0
            )
            final_scores["total_structures_evaluated"] = metrics.get(
                "total_structures_evaluated", 0
            )
            final_scores["failed_fingerprinting_count"] = metrics.get(
                "failed_fingerprinting_count", 0
            )

        return final_scores