"""Novelty benchmark for material structures.

This module implements a benchmark that evaluates the novelty of
generated material structures by comparing them against known materials
in reference datasets using structure fingerprinting.
"""

from typing import Any, Dict

import numpy as np

from lematerial_forgebench.benchmarks.base import BaseBenchmark
from lematerial_forgebench.evaluator import EvaluationResult, EvaluatorConfig
from lematerial_forgebench.metrics.novelty_metric import NoveltyMetric
from lematerial_forgebench.utils.distribution_utils import safe_float


class NoveltyBenchmark(BaseBenchmark):
    """Benchmark for evaluating the novelty of generated material structures.

    This benchmark uses the NoveltyMetric to compare generated structures
    against reference datasets (like LeMat-Bulk) to determine how many
    structures are truly novel vs. known materials.
    """

    def __init__(
        self,
        reference_dataset: str = "LeMaterial/LeMat-Bulk",
        reference_config: str = "compatible_pbe",
        fingerprint_method: str = "bawl",
        cache_reference: bool = True,
        max_reference_size: int = None,
        name: str = "NoveltyBenchmark",
        description: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ):
        """Initialize the novelty benchmark.

        Parameters
        ----------
        reference_dataset : str, default="LeMaterial/LeMat-Bulk"
            HuggingFace dataset name to use as reference for known materials.
        reference_config : str, default="compatible_pbe"
            Configuration/subset of the reference dataset to use.
        fingerprint_method : str, default="bawl"
            Method to use for structure fingerprinting.
        cache_reference : bool, default=True
            Whether to cache the reference dataset fingerprints.
        max_reference_size : int | None, default=None
            Maximum number of structures to load from reference dataset.
        name : str
            Name of the benchmark.
        description : str, optional
            Description of the benchmark.
        metadata : dict, optional
            Additional metadata for the benchmark.
        """
        if description is None:
            description = (
                "Evaluates the novelty of crystal structures by comparing them "
                "against known materials in reference datasets using structure "
                "fingerprinting to identify truly novel materials."
            )

        # Initialize the novelty metric
        novelty_metric = NoveltyMetric(
            reference_dataset=reference_dataset,
            reference_config=reference_config,
            fingerprint_method=fingerprint_method,
            cache_reference=cache_reference,
            max_reference_size=max_reference_size,
        )

        # Set up evaluator configs
        evaluator_configs = {
            "novelty": EvaluatorConfig(
                name="Novelty Analysis",
                description=(
                    "Evaluates structural novelty against reference "
                    "datasets"
                ),
                metrics={"novelty": novelty_metric},
                weights={"novelty": 1.0},
                aggregation_method="weighted_mean",
            ),
        }

        # Create benchmark metadata
        benchmark_metadata = {
            "version": "0.1.0",
            "category": "novelty",
            "reference_dataset": reference_dataset,
            "reference_config": reference_config,
            "fingerprint_method": fingerprint_method,
            "max_reference_size": max_reference_size,
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
        """Aggregate results from the novelty evaluator into final scores.

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
            "novelty_ratio": np.nan,  # Alias for novelty_score for clarity
        }

        # Extract novelty results
        novelty_results = evaluator_results.get("novelty")
        if novelty_results:
            # Get the combined score (should be same as novelty_score)
            final_scores["novelty_score"] = safe_float(
                novelty_results.get("combined_value")
            )
            final_scores["novelty_ratio"] = final_scores["novelty_score"]

            # Extract detailed metrics from the metric results
            metric_results = novelty_results.get("metric_results", {})
            novelty_metric_result = metric_results.get("novelty", {})

            if hasattr(novelty_metric_result, "metrics"):
                metrics = novelty_metric_result.metrics
            elif isinstance(novelty_metric_result, dict):
                metrics = novelty_metric_result.get("metrics", {})
            else:
                metrics = {}

            # Extract count information
            final_scores["novel_structures_count"] = metrics.get(
                "novel_structures_count", 0
            )
            final_scores["total_structures_evaluated"] = metrics.get(
                "total_structures_evaluated", 0
            )

        return final_scores
