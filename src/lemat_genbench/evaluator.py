"""Metric evaluation

This module provides tools for running multiple
metrics and combining their results.
"""

import time
from dataclasses import dataclass

import numpy as np
from pymatgen.core.structure import Structure

from lemat_genbench.metrics.base import BaseMetric, MetricConfig, MetricResult
from lemat_genbench.utils.logging import logger


@dataclass
class EvaluatorConfig:
    """Configuration for the evaluator.

    Parameters
    ----------
    metrics : dict[str, MetricConfig]
        Dictionary of metrics to evaluate.
    weights : dict[str, float] | None
        Optional weights for each metric when combining results.
        Keys should be metric names, values are weights.
        If not provided, all metrics are weighted equally.
    aggregation_method : str
        How to combine metric values when weights are provided:
        - 'weighted_sum': Weighted sum of metric values
        - 'weighted_mean': Weighted mean of metric values
        - 'min': Minimum value across metrics
        - 'max': Maximum value across metrics
    """

    name: str
    description: str
    metrics: dict[str, MetricConfig]
    weights: dict[str, float] | None = None
    aggregation_method: str = "weighted_mean"


@dataclass
class EvaluationResult:
    """Result of running multiple metrics.

    Parameters
    ----------
    metric_results : dict[str, MetricResult]
        Results for each individual metric.
    combined_value : float | None
        Combined value if weights were provided.
    uncertainties : dict[str, dict[str, float]] | None
        Raw uncertainty measures from all metrics.
        Common keys include:
        - 'std': Standard deviation
    computation_time : float
        Total time taken for evaluation.
    config : EvaluatorConfig
        Configuration used for evaluation.
    """

    metric_results: dict[str, MetricResult]
    combined_value: float | None
    uncertainties: dict[str, dict[str, float]] | None
    computation_time: float
    config: EvaluatorConfig


class MetricEvaluator:
    """Evaluate and combine multiple metrics.

    It provides a way to:
    1. Run multiple metrics on the same set of structures
    2. Combine results using specific aggregation methods
    3. Provide error reporting

    Example
    -------

    .. code-block:: python

        metrics = {
            "StructuralDiversity": MetricConfig(
                name="StructuralDiversity",
                description="Structural diversity metric",
                metric=StructuralDiversityMetric(),
            ),
            "CompositionDiversity": MetricConfig(
                name="CompositionDiversity",
                description="Composition diversity metric",
                metric=CompositionDiversityMetric(),
            ),
            "Novelty": MetricConfig(
                name="Novelty",
                description="Novelty metric",
                metric=NoveltyMetric(),
            )
        }
        weights = {
            "StructuralDiversity": 0.4,
            "CompositionDiversity": 0.4,
            "Novelty": 0.2
        }
        evaluator = MetricEvaluator(
            metrics=metrics,
            weights=weights,
            aggregation_method="weighted_mean"
        )
        result = evaluator.evaluate(structures)
        print(f"Combined score: {result.combined_value:.3f}")
        for name, res in result.metric_results.items():
            print(f"{name}: {res.value:.3f}")
    """

    def __init__(
        self,
        metrics: dict[str, BaseMetric],
        weights: dict[str, float] | None = None,
        aggregation_method: str = "weighted_mean",
        name: str | None = None,
        description: str | None = None,
    ):
        """Initialize the evaluator.

        Parameters
        ----------
        metrics : dict[str, BaseMetric]
            Dictionary of metrics to evaluate.
        weights : dict[str, float], optional
            Weights for each metric when combining results.
            Keys should be metric names, values are weights.
        aggregation_method : str, default="weighted_mean"
            How to combine metric values when weights are provided.
        name : str, optional
            Name for this evaluator.
        description : str, optional
            Description of what this evaluator measures.
        """
        self.metrics = metrics
        metrics_config = {name: metric.config for name, metric in metrics.items()}

        self.config = EvaluatorConfig(
            name=name or "MetricEvaluator",
            description=description or "Combines multiple metrics",
            metrics=metrics_config,
            weights=weights,
            aggregation_method=aggregation_method,
        )

        # Validate weights if provided
        if weights is not None:
            missing = set(metrics.keys()) - set(weights.keys())
            if missing:
                raise ValueError(f"Missing weights for metrics: {missing}")
            extra = set(weights.keys()) - set(metrics.keys())
            if extra:
                raise ValueError(f"Weights provided for non-existent metrics: {extra}")

        # Validate aggregation method
        if self.config.aggregation_method not in [
            "weighted_sum",
            "weighted_mean",
            "min",
            "max",
        ]:
            raise ValueError(
                f"Unknown aggregation method: {self.config.aggregation_method}"
            )

    def evaluate(self, structures: list[Structure]) -> EvaluationResult:
        """Evaluate all metrics on the given structures.

        Parameters
        ----------
        structures : list[Structure]
            Structures to evaluate.

        Returns
        -------
        EvaluationResult
            Results from all metrics and their combination if weights
            were provided.
        """
        start_time = time.time()
        metric_results: dict[str, MetricResult] = {}

        # We run the metrics sequentially
        # each metric handles its own parallelization
        for metric_name, metric in self.metrics.items():
            try:
                result = metric.compute(
                    structures=structures, **metric._get_compute_attributes()
                )
                metric_results[metric_name] = result

            except Exception as e:
                logger.error(f"Failed to compute metric {metric_name}", exc_info=True)
                metric_results[metric_name] = MetricResult(
                    metrics={metric_name: float("nan")},
                    primary_metric=metric_name,
                    uncertainties=None,
                    config=metric.config,
                    computation_time=0.0,
                    individual_values=[],
                    n_structures=len(structures),
                    failed_indices=list(range(len(structures))),
                    warnings=[f"Failed to compute: {str(e)}"],
                )

        # Combine results if weights provided
        combined_value = None
        try:
            if self.config.weights is not None:
                values = []
                weights = []
                all_uncertainties = {}

                for metric_name, metric in self.config.metrics.items():
                    result = metric_results[metric_name]
                    if not np.isnan(result.value):
                        value = result.value

                        values.append(value)
                        weights.append(self.config.weights[metric_name])

                        if result.uncertainties:
                            for key, val in result.uncertainties.items():
                                if key not in all_uncertainties:
                                    all_uncertainties[key] = []
                                all_uncertainties[key].append(val)

                if values:
                    weights = np.array(weights) / np.sum(weights)  # Normalize weights

                    if self.config.aggregation_method == "weighted_sum":
                        combined_value = np.sum(np.array(values) * weights)
                    elif self.config.aggregation_method == "weighted_mean":
                        combined_value = np.average(values, weights=weights)
                    elif self.config.aggregation_method == "min":
                        combined_value = np.min(values)
                    elif self.config.aggregation_method == "max":
                        combined_value = np.max(values)
                    else:
                        raise ValueError(
                            f"Unknown aggregation method: {self.config.aggregation_method}"
                        )
        except TypeError:
            pass

        return EvaluationResult(
            metric_results=metric_results,
            combined_value=combined_value,
            uncertainties=all_uncertainties,
            computation_time=time.time() - start_time,
            config=self.config,
        )

    def __call__(
        self,
        structures: list[Structure],
    ) -> EvaluationResult:
        """Convenient callable interface."""
        return self.evaluate(structures)
