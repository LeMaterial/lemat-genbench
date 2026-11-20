"""Novelty benchmark for material structures.

This module implements a benchmark that evaluates the novelty of
generated material structures by comparing them against known materials
in reference datasets using structure fingerprinting.
"""

from typing import Any, Dict

import numpy as np

from lemat_genbench.benchmarks.base import BaseBenchmark
from lemat_genbench.evaluator import EvaluationResult, EvaluatorConfig
from lemat_genbench.metrics.novelty_metric import NoveltyMetric


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
        n_jobs: int = 1,
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
            n_jobs=n_jobs,
        )

        # Set up evaluator configs
        evaluator_configs = {
            "novelty": EvaluatorConfig(
                name="novelty",
                description=("Evaluates structural novelty against reference datasets"),
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
        final_scores = {
            "novelty_score": np.nan,
            "novel_structures_count": 0,
            "total_structures_evaluated": 0,
            "novelty_ratio": np.nan,
            "novel_composition_count": 0,
            "novel_spacegroup_count": 0,
            "novel_structure_only_count": 0,
            "framework_existing_anon_known_sg": 0,
            "framework_existing_anon_novel_sg": 0,
            "framework_novel_anon": 0,
        }

        novelty_results = evaluator_results.get("novelty")
        if novelty_results:
            combined_value = novelty_results.get("combined_value")
            if combined_value is not None:
                final_scores["novelty_score"] = float(combined_value)
                final_scores["novelty_ratio"] = float(combined_value)

            metric_results = novelty_results.get("metric_results", {})
            novelty_metric_result = metric_results.get("novelty", {})

            if hasattr(novelty_metric_result, "metrics"):
                metrics = novelty_metric_result.metrics
            elif isinstance(novelty_metric_result, dict):
                metrics = novelty_metric_result.get("metrics", {})
            else:
                metrics = {}

            for key in (
                "novel_structures_count",
                "total_structures_evaluated",
                "novel_composition_count",
                "novel_spacegroup_count",
                "novel_structure_only_count",
                "framework_existing_anon_known_sg",
                "framework_existing_anon_novel_sg",
                "framework_novel_anon",
            ):
                final_scores[key] = metrics.get(key, 0)

        return final_scores
