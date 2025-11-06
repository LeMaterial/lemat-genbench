"""SUN (Stable, Unique, Novel) benchmark for material structures.

This module implements a benchmark that evaluates the proportion of
generated material structures that are simultaneously stable, unique,
and novel using the SUN metric framework with the new hierarchical order:
Stability → Uniqueness → Novelty.
"""

from typing import Any, Dict, List

import numpy as np

from lemat_genbench.benchmarks.base import BaseBenchmark
from lemat_genbench.evaluator import EvaluationResult, EvaluatorConfig
from lemat_genbench.metrics.sun_metric import MetaSUNMetric, SUNMetric


class SUNBenchmark(BaseBenchmark):
    """Benchmark for evaluating SUN (Stable, Unique, Novel) rate of structures.

    This benchmark uses the SUNMetric to evaluate the proportion of generated
    structures that are simultaneously:
    1. Stable (e_above_hull <= stability_threshold) [computed first]
    2. Unique (not duplicated within the stable/metastable sets) [computed second]
    3. Novel (not present in reference dataset) [computed third]

    The benchmark also evaluates MetaSUN rate for metastable structures and
    provides detailed hierarchical reporting of counts at each filtering stage.
    """

    def __init__(
        self,
        stability_threshold: float = 0.0,
        metastability_threshold: float = 0.1,
        reference_dataset_path: str = "mp-20-data/mp_20.csv",
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
        reference_dataset_path : str, default="mp-20-data/mp_20.csv"
            Path to MP-20 CSV file to use as reference for novelty.
        fingerprint_method : str, default="bawl"
            Method to use for structure fingerprinting/comparison.
            Supports: "bawl", "short-bawl", "structure-matcher"
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
                "Evaluates the SUN (Stable, Unique, Novel) rate of crystal structures "
                "using the hierarchical order: Stability → Uniqueness → Novelty. "
                "Measures the proportion that are simultaneously stable, unique within "
                "the stable/metastable sets, and novel compared to MP-20 reference."
            )

        # Initialize the main SUN metric
        sun_metric = SUNMetric(
            stability_threshold=stability_threshold,
            metastability_threshold=metastability_threshold,
            reference_dataset_path=reference_dataset_path,
            fingerprint_method=fingerprint_method,
            cache_reference=cache_reference,
            max_reference_size=max_reference_size,
        )

        # Set up evaluator configs
        evaluator_configs = {
            "sun": EvaluatorConfig(
                name="SUN",
                description=("Evaluates structures that are Stable, Unique, and Novel"),
                metrics={"sun": sun_metric},
                weights={"sun": 1.0},
                aggregation_method="weighted_mean",
            ),
        }

        # Add MetaSUN evaluator if requested
        if include_metasun:
            metasun_metric = MetaSUNMetric(
                metastability_threshold=metastability_threshold,
                reference_dataset_path=reference_dataset_path,
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
            "version": "0.2.0",  # Updated version for new hierarchical order
            "category": "sun",
            "computation_order": "Stability → Uniqueness → Novelty",
            "stability_threshold": stability_threshold,
            "metastability_threshold": metastability_threshold,
            "reference_dataset_path": reference_dataset_path,
            "fingerprint_method": fingerprint_method,
            "max_reference_size": max_reference_size,
            "include_metasun": include_metasun,
            "supports_structure_matcher": True,
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

        This method extracts all the hierarchical metrics and indices from the
        updated SUN metric implementation.

        Parameters
        ----------
        evaluator_results : dict[str, EvaluationResult]
            Results from each evaluator.

        Returns
        -------
        dict[str, float]
            Final aggregated scores containing all SUN hierarchical metrics and indices.
        """
        # Initialize default scores with all hierarchical metrics
        final_scores = {
            # Primary SUN metrics
            "sun_rate": np.nan,
            "msun_rate": np.nan,
            "combined_sun_msun_rate": np.nan,
            
            # Hierarchical counts - Level 1: Stability
            "total_structures_evaluated": 0,
            "stable_count": 0,
            "metastable_count": 0,
            "stable_rate": np.nan,
            "metastable_rate": np.nan,
            
            # Hierarchical counts - Level 2: Uniqueness within stable/metastable
            "unique_in_stable_count": 0,
            "unique_in_metastable_count": 0,
            "unique_in_stable_rate": np.nan,
            "unique_in_metastable_rate": np.nan,
            
            # Hierarchical counts - Level 3: Novelty (final SUN/MetaSUN)
            "sun_count": 0,
            "msun_count": 0,
            
            # Error tracking
            "failed_count": 0,
            
            # Structure indices (will be populated if available)
            "sun_indices": [],
            "msun_indices": [],
            "stable_indices": [],
            "metastable_indices": [],
            "stable_unique_indices": [],
            "metastable_unique_indices": [],
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

            # Handle both MetricResult objects and dict representations
            if hasattr(sun_metric_result, "metrics"):
                metrics = sun_metric_result.metrics
                metric_result_obj = sun_metric_result
            elif isinstance(sun_metric_result, dict):
                metrics = sun_metric_result.get("metrics", {})
                metric_result_obj = sun_metric_result
            else:
                metrics = {}
                metric_result_obj = None

            # Extract all hierarchical metrics
            for key in [
                "sun_rate", "msun_rate", "combined_sun_msun_rate",
                "total_structures_evaluated", "stable_count", "metastable_count",
                "stable_rate", "metastable_rate",
                "unique_in_stable_count", "unique_in_metastable_count",
                "unique_in_stable_rate", "unique_in_metastable_rate",
                "sun_count", "msun_count", "failed_count"
            ]:
                if key in metrics:
                    final_scores[key] = metrics[key]

            # Extract indices if available (from MetricResult object)
            if metric_result_obj and hasattr(metric_result_obj, "sun_indices"):
                final_scores["sun_indices"] = getattr(metric_result_obj, "sun_indices", [])
                final_scores["msun_indices"] = getattr(metric_result_obj, "msun_indices", [])
                final_scores["stable_indices"] = getattr(metric_result_obj, "stable_indices", [])
                final_scores["metastable_indices"] = getattr(metric_result_obj, "metastable_indices", [])
                final_scores["stable_unique_indices"] = getattr(metric_result_obj, "stable_unique_indices", [])
                final_scores["metastable_unique_indices"] = getattr(metric_result_obj, "metastable_unique_indices", [])

        # Extract MetaSUN results if available
        metasun_results = evaluator_results.get("metasun")
        if metasun_results:
            # Get the combined score for MetaSUN
            metasun_combined_value = metasun_results.get("combined_value")
            if metasun_combined_value is not None:
                final_scores["metasun_primary_rate"] = float(metasun_combined_value)

            # Extract detailed MetaSUN metrics
            metric_results = metasun_results.get("metric_results", {})
            metasun_metric_result = metric_results.get("metasun", {})

            if hasattr(metasun_metric_result, "metrics"):
                metasun_metrics = metasun_metric_result.metrics
                metasun_result_obj = metasun_metric_result
            elif isinstance(metasun_metric_result, dict):
                metasun_metrics = metasun_metric_result.get("metrics", {})
                metasun_result_obj = metasun_metric_result
            else:
                metasun_metrics = {}
                metasun_result_obj = None

            # Add MetaSUN specific metrics with prefixes to avoid conflicts
            for key, value in metasun_metrics.items():
                if key in ["sun_rate", "sun_count"]:
                    # Map sun_* metrics to metasun_* for the MetaSUN evaluator
                    metasun_key = key.replace("sun_", "metasun_", 1)
                    final_scores[metasun_key] = value
                elif key.startswith("metastable_"):
                    # Keep metastable-specific metrics
                    final_scores[f"metasun_{key}"] = value

            # Extract MetaSUN indices if available
            if metasun_result_obj and hasattr(metasun_result_obj, "sun_indices"):
                # For MetaSUN, the "sun_indices" actually represent metasun structures
                final_scores["metasun_structure_indices"] = getattr(metasun_result_obj, "sun_indices", [])

        return final_scores

    def get_structure_indices(self, evaluator_results: Dict[str, EvaluationResult]) -> Dict[str, List[int]]:
        """Extract structure indices from evaluation results.
        
        This is a convenience method to get all the hierarchical structure indices
        for further analysis.

        Parameters
        ----------
        evaluator_results : dict[str, EvaluationResult]
            Results from each evaluator.

        Returns
        -------
        dict[str, List[int]]
            Dictionary containing all structure index lists.
        """
        indices = {
            "sun_indices": [],
            "msun_indices": [],
            "stable_indices": [],
            "metastable_indices": [],
            "stable_unique_indices": [],
            "metastable_unique_indices": [],
        }

        # Extract from SUN results
        sun_results = evaluator_results.get("sun")
        if sun_results:
            metric_results = sun_results.get("metric_results", {})
            sun_metric_result = metric_results.get("sun", {})

            if hasattr(sun_metric_result, "sun_indices"):
                indices["sun_indices"] = getattr(sun_metric_result, "sun_indices", [])
                indices["msun_indices"] = getattr(sun_metric_result, "msun_indices", [])
                indices["stable_indices"] = getattr(sun_metric_result, "stable_indices", [])
                indices["metastable_indices"] = getattr(sun_metric_result, "metastable_indices", [])
                indices["stable_unique_indices"] = getattr(sun_metric_result, "stable_unique_indices", [])
                indices["metastable_unique_indices"] = getattr(sun_metric_result, "metastable_unique_indices", [])

        return indices

    def get_hierarchical_summary(self, evaluator_results: Dict[str, EvaluationResult]) -> Dict[str, Any]:
        """Get a summary of the hierarchical filtering process.
        
        This method provides a clear view of how many structures survive each
        stage of the filtering process.

        Parameters
        ----------
        evaluator_results : dict[str, EvaluationResult]
            Results from each evaluator.

        Returns
        -------
        dict[str, Any]
            Summary of the hierarchical filtering process.
        """
        final_scores = self.aggregate_evaluator_results(evaluator_results)
        
        total = final_scores.get("total_structures_evaluated", 0)
        
        summary = {
            "total_structures": total,
            "filtering_stages": {
                "1_stability": {
                    "stable_count": final_scores.get("stable_count", 0),
                    "metastable_count": final_scores.get("metastable_count", 0),
                    "stable_rate": final_scores.get("stable_rate", 0.0),
                    "metastable_rate": final_scores.get("metastable_rate", 0.0),
                },
                "2_uniqueness": {
                    "unique_in_stable_count": final_scores.get("unique_in_stable_count", 0),
                    "unique_in_metastable_count": final_scores.get("unique_in_metastable_count", 0),
                    "unique_in_stable_rate": final_scores.get("unique_in_stable_rate", 0.0),
                    "unique_in_metastable_rate": final_scores.get("unique_in_metastable_rate", 0.0),
                },
                "3_novelty": {
                    "sun_count": final_scores.get("sun_count", 0),
                    "msun_count": final_scores.get("msun_count", 0),
                    "sun_rate": final_scores.get("sun_rate", 0.0),
                    "msun_rate": final_scores.get("msun_rate", 0.0),
                },
            },
            "final_metrics": {
                "sun_rate": final_scores.get("sun_rate", 0.0),
                "msun_rate": final_scores.get("msun_rate", 0.0),
                "combined_sun_msun_rate": final_scores.get("combined_sun_msun_rate", 0.0),
            },
            "filtering_efficiency": {
                "stability_survival_rate": (final_scores.get("stable_count", 0) + final_scores.get("metastable_count", 0)) / total if total > 0 else 0.0,
                "uniqueness_survival_rate": (final_scores.get("unique_in_stable_count", 0) + final_scores.get("unique_in_metastable_count", 0)) / total if total > 0 else 0.0,
                "novelty_survival_rate": (final_scores.get("sun_count", 0) + final_scores.get("msun_count", 0)) / total if total > 0 else 0.0,
            }
        }
        
        return summary