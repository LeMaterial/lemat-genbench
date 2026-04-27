"""Uniqueness benchmark for material structures.

This module implements a benchmark that evaluates the uniqueness of
generated material structures by measuring the fraction of unique
structures within the generated set using structure fingerprinting.
"""

from typing import Any, Dict

import numpy as np

from lemat_genbench.benchmarks.base import BaseBenchmark
from lemat_genbench.evaluator import EvaluationResult, EvaluatorConfig
from lemat_genbench.metrics.uniqueness_metric import UniquenessMetric


class UniquenessBenchmark(BaseBenchmark):
    """Benchmark for evaluating the uniqueness of generated material structures.

    This benchmark uses the UniquenessMetric to measure the fraction of unique
    structures within a generated set, detecting duplicates using structure
    fingerprinting to assess the diversity of generated materials.
    """

    def __init__(
        self,
        fingerprint_method: str = "bawl",
        name: str = "UniquenessBenchmark",
        description: str | None = None,
        metadata: Dict[str, Any] | None = None,
        n_jobs: int = 1,
    ):
        """Initialize the uniqueness benchmark.

        Parameters
        ----------
        fingerprint_method : str, default="bawl"
            Method to use for structure fingerprinting.
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
                "using structure fingerprinting to detect duplicates and "
                "assess diversity."
            )

        # Initialize the uniqueness metric
        uniqueness_metric = UniquenessMetric(
            fingerprint_method=fingerprint_method,
            n_jobs=n_jobs,
        )

        # Set up evaluator configs
        evaluator_configs = {
            "uniqueness": EvaluatorConfig(
                name="uniqueness",
                description=("Evaluates structural uniqueness within generated set"),
                metrics={"uniqueness": uniqueness_metric},
                weights={"uniqueness": 1.0},
                aggregation_method="weighted_mean",
            ),
        }

        # Create benchmark metadata
        benchmark_metadata = {
            "version": "0.1.0",
            "category": "uniqueness",
            "fingerprint_method": fingerprint_method,
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
        final_scores = {
            "uniqueness_score": np.nan,
            "unique_structures_count": 0,
            "duplicate_structures_count": 0,
            "total_structures_evaluated": 0,
            "failed_fingerprinting_count": 0,
            "uniqueness_ratio": np.nan,
            "unique_composition_count": 0,
            "unique_spacegroup_count": 0,
            "unique_structure_only_count": 0,
        }

        uniqueness_results = evaluator_results.get("uniqueness")
        if uniqueness_results:
            combined_value = uniqueness_results.get("combined_value")
            if combined_value is not None:
                final_scores["uniqueness_score"] = float(combined_value)
                final_scores["uniqueness_ratio"] = float(combined_value)

            metric_results = uniqueness_results.get("metric_results", {})
            uniqueness_metric_result = metric_results.get("uniqueness", {})

            if hasattr(uniqueness_metric_result, "metrics"):
                metrics = uniqueness_metric_result.metrics
            elif isinstance(uniqueness_metric_result, dict):
                metrics = uniqueness_metric_result.get("metrics", {})
            else:
                metrics = {}

            for key in (
                "unique_structures_count",
                "duplicate_structures_count",
                "total_structures_evaluated",
                "failed_fingerprinting_count",
                "unique_composition_count",
                "unique_spacegroup_count",
                "unique_structure_only_count",
            ):
                final_scores[key] = metrics.get(key, 0)

        return final_scores
