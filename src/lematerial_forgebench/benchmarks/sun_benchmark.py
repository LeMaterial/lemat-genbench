"""SUN (Stable, Unique, Novel) benchmark for material structures.

This module implements a benchmark that evaluates the proportion of
generated material structures that are simultaneously stable, unique,
and novel using the SUN metric framework.
"""

from typing import Any, Dict

import numpy as np

from lematerial_forgebench.benchmarks.base import BaseBenchmark
from lematerial_forgebench.evaluator import EvaluationResult, EvaluatorConfig
from lematerial_forgebench.metrics.sun_metric import MetaSUNMetric, SUNMetric


class SUNBenchmark(BaseBenchmark):
    """Benchmark for evaluating SUN (Stable, Unique, Novel) rate of structures.

    This benchmark uses the SUNMetric to evaluate the proportion of generated
    structures that are simultaneously:
    1. Stable (e_above_hull <= stability_threshold)
    2. Unique (not duplicated within the generated set)
    3. Novel (not present in reference dataset)

    The benchmark also evaluates MetaSUN rate for metastable structures.
    """

    def __init__(
        self,
        stability_threshold: float = 0.0,
        metastability_threshold: float = 0.1,
        reference_dataset: str = "LeMaterial/LeMat-Bulk",
        reference_config: str = "compatible_pbe",
        fingerprint_method: str = "bawl",
        cache_reference: bool = True,
        max_reference_size: int = None,
        include_metasun: bool = True,
        name: str = "SUNBenchmark",
        description: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ):
        """Initialize the SUN benchmark.

        Parameters
        ----------
        stability_threshold : float, default=0.0
            Energy above hull threshold for stability (eV/atom).
        metastability_threshold : float, default=0.1
            Energy above hull threshold for metastability (eV/atom).
        reference_dataset : str, default="LeMaterial/LeMat-Bulk"
            HuggingFace dataset name to use as reference for novelty.
        reference_config : str, default="compatible_pbe"
            Configuration/subset of the reference dataset to use.
        fingerprint_method : str, default="bawl"
            Method to use for structure fingerprinting.
        cache_reference : bool, default=True
            Whether to cache the reference dataset fingerprints.
        max_reference_size : int | None, default=None
            Maximum number of structures to load from reference dataset.
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
                "Evaluates the SUN (Stable, Unique, Novel) rate of crystal structures, "
                "measuring the proportion that are simultaneously stable, unique within "
                "the generated set, and novel compared to reference datasets."
            )

        # Initialize the main SUN metric
        sun_metric = SUNMetric(
            stability_threshold=stability_threshold,
            metastability_threshold=metastability_threshold,
            reference_dataset=reference_dataset,
            reference_config=reference_config,
            fingerprint_method=fingerprint_method,
            cache_reference=cache_reference,
            max_reference_size=max_reference_size,
        )

        # Set up evaluator configs
        evaluator_configs = {
            "sun": EvaluatorConfig(
                name="SUN",
                description=(
                    "Evaluates structures that are Stable, Unique, and Novel"
                ),
                metrics={"sun": sun_metric},
                weights={"sun": 1.0},
                aggregation_method="weighted_mean",
            ),
        }

        # Add MetaSUN evaluator if requested
        if include_metasun:
            metasun_metric = MetaSUNMetric(
                metastability_threshold=metastability_threshold,
                reference_dataset=reference_dataset,
                reference_config=reference_config,
                fingerprint_method=fingerprint_method,
                cache_reference=cache_reference,
                max_reference_size=max_reference_size,
            )

            evaluator_configs["metasun"] = EvaluatorConfig(
                name="MetaSUN",
                description=(
                    "Evaluates structures that are Metastable, Unique, and Novel"
                ),
                metrics={"metasun": metasun_metric},
                weights={"metasun": 1.0},
                aggregation_method="weighted_mean",
            )

        # Create benchmark metadata
        benchmark_metadata = {
            "version": "0.1.0",
            "category": "sun",
            "stability_threshold": stability_threshold,
            "metastability_threshold": metastability_threshold,
            "reference_dataset": reference_dataset,
            "reference_config": reference_config,
            "fingerprint_method": fingerprint_method,
            "max_reference_size": max_reference_size,
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
                "sun_count", "msun_count", "combined_sun_msun_rate",
                "unique_count", "unique_rate", "total_structures_evaluated",
                "failed_count", "msun_rate"
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