"""Base classes for material generation benchmarks.

This module allows to create standardized benchmarks
that can evaluate material generation models across
multiple metrics and aggregate results.

The benchmarks are composed of multiple evaluators,
each combining different metrics in specific ways to
assess different aspects of generated materials.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd
from lemat_genbench.evaluator import (
    EvaluationResult,
    EvaluatorConfig,
    MetricEvaluator,
)
from pymatgen.core.structure import Structure


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark.

    Parameters
    ----------
    name : str
        Name of the benchmark
    description : str
        Detailed description of what the benchmark evaluates
    evaluator_configs : dict[str, EvaluatorConfig]
        List of evaluator configurations, each defining a set of metrics
        and how they should be combined
    metadata : dict[str, any] | None
        Additional metadata about the benchmark (e.g., version, paper reference)
    """

    name: str
    description: str
    evaluator_configs: dict[str, EvaluatorConfig]
    metadata: dict[str, any] | None = None
    reference_df: pd.DataFrame | None = None
    mlips: list[str] | None = None 


@dataclass
class BenchmarkResult:
    """Results from running a benchmark.

    Parameters
    ----------
    evaluator_results : list[dict[str, float]]
        Results from each evaluator, containing metric values
    final_scores : dict[str, float]
        Aggregated final scores for the benchmark
    metadata : dict[str, any]
        Additional metadata about the evaluation
    """

    evaluator_results: list[dict[str, float]]
    final_scores: dict[str, float]
    metadata: dict[str, any]


class BaseBenchmark(ABC):
    """Base class for all material generation benchmarks.

    This class defines the interface that all benchmarks must implement.
    A benchmark typically consists of multiple evaluators, each combining
    different metrics in specific ways to assess different aspects of
    generated materials.

    Parameters
    ----------
    name : str
        Name of the benchmark
    description : str
        Detailed description of what the benchmark evaluates
    evaluator_configs : list[EvaluatorConfig]
        List of evaluator configurations
    metadata : dict[str, any], optional
        Additional metadata about the benchmark
        This is only used for logging and tracking purposes
        and does not affect the evaluation results
    """

    def __init__(
        self,
        name: str,
        description: str,
        evaluator_configs: dict[str, EvaluatorConfig],
        metadata: dict[str, any] | None = None,
        reference_df: pd.DataFrame | None = None,
        mlips: list[str] | None = None,
    ):
        self.config = BenchmarkConfig(
            name=name,
            description=description,
            evaluator_configs=evaluator_configs,
            metadata=metadata or {},
            reference_df=reference_df,
            mlips=mlips,
        )

        # Create evaluators from configs
        self.evaluators = {
            name: MetricEvaluator(
                metrics=config.metrics,
                weights=config.weights,
                aggregation_method=config.aggregation_method,
                name=name,
                description=f"{self.config.name} {name}",
            )
            for name, config in self.config.evaluator_configs.items()
        }

    @abstractmethod
    def aggregate_evaluator_results(
        self, evaluator_results: dict[str, EvaluationResult]
    ) -> dict[str, float]:
        """Aggregate results from multiple evaluators into final scores.

        This method defines how the results from different evaluators
        should be combined into the final benchmark scores.

        It should be implemented for every benchmark.

        Parameters
        ----------
        evaluator_results : dict[str, EvaluationResult]
            Results from each evaluator

        Returns
        -------
        dict[str, float]
            Final aggregated scores
        """
        pass

    def evaluate(self, structures: list[Structure]) -> BenchmarkResult:
        """Run the complete benchmark evaluation.

        Parameters
        ----------
        structures : list[Structure]
            Structures to evaluate

        Returns
        -------
        BenchmarkResult
            Complete benchmark results
        """
        evaluator_results = {}
        for name, evaluator in self.evaluators.items():
            result = evaluator.evaluate(
                structures=structures,
            )

            evaluator_results[name] = {
                "combined_value": result.combined_value,
                **{
                    f"{name}_value": res.value
                    for name, res in result.metric_results.items()
                },
                "metric_results": result.metric_results,
                **{
                    f"{name}_value": res.value
                    for name, res in result.metric_results.items()
                },
            }

        final_scores = self.aggregate_evaluator_results(evaluator_results)
        result_metadata = {
            "benchmark_name": self.config.name,
            "benchmark_description": self.config.description,
            "n_structures": len(structures),
            "n_evaluators": len(self.evaluators),
            **(self.config.metadata or {}),
        }

        return BenchmarkResult(
            evaluator_results=evaluator_results,
            final_scores=final_scores,
            metadata=result_metadata,
        )
