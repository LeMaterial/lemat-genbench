"""Validity benchmark for material structures.

This module implements a benchmark that evaluates the validity of
generated material structures using fundamental validity criteria.
"""

from typing import Any, Dict

from lemat_genbench.benchmarks.base import BaseBenchmark
from lemat_genbench.evaluator import EvaluationResult, EvaluatorConfig
from lemat_genbench.metrics.validity_metrics import (
    ChargeNeutralityMetric,
    CompositeValidityMetric,
    MinimumInteratomicDistanceMetric,
    PhysicalPlausibilityMetric,
)


class ValidityBenchmark(BaseBenchmark):
    """Benchmark for evaluating the validity of generated material structures."""

    def __init__(
        self,
        charge_weight: float = 0.25,
        distance_weight: float = 0.25,
        # coordination_weight: float = 0.25,
        plausibility_weight: float = 0.25,
        name: str = "ValidityBenchmark",
        description: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ):
        if description is None:
            description = (
                "Evaluates the validity of crystal structures based on physical and "
                "chemical principles including charge neutrality, interatomic distances, "
                "coordination environments, and physical plausibility."
            )

        # Initialize metrics
        charge_metric = ChargeNeutralityMetric()
        distance_metric = MinimumInteratomicDistanceMetric()
        # coordination_metric = CoordinationEnvironmentMetric()
        plausibility_metric = PhysicalPlausibilityMetric()

        # Set up evaluators - pass the metric objects, not their configs
        evaluator_configs = {
            "charge_neutrality": EvaluatorConfig(
                name="Charge Neutrality",
                description="Evaluates charge balance in structures",
                metrics={
                    "charge_neutrality": charge_metric
                },  # Pass metric object, not config
                weights={"charge_neutrality": 1.0},
                aggregation_method="weighted_mean",
            ),
            "interatomic_distance": EvaluatorConfig(
                name="Interatomic Distance",
                description="Evaluates minimum distances between atoms",
                metrics={
                    "min_distance": distance_metric
                },  # Pass metric object, not config
                weights={"min_distance": 1.0},
                aggregation_method="weighted_mean",
            ),
            # "coordination_environment": EvaluatorConfig(
            #     name="Coordination Environment",
            #     description="Evaluates chemical bonding environments",
            #     metrics={
            #         "coordination": coordination_metric
            #     },  # Pass metric object, not config
            #     weights={"coordination": 1.0},
            #     aggregation_method="weighted_mean",
            # ),
            "physical_plausibility": EvaluatorConfig(
                name="Physical Plausibility",
                description="Evaluates basic physical properties",
                metrics={
                    "plausibility": plausibility_metric
                },  # Pass metric object, not config
                weights={"plausibility": 1.0},
                aggregation_method="weighted_mean",
            ),
            "overall_validity": EvaluatorConfig(
                name="Overall Validity",
                description="Combined validity score",
                metrics={
                    "composite": CompositeValidityMetric(
                        metrics={
                            "charge_neutrality": charge_metric,
                            "min_distance": distance_metric,
                            # "coordination": coordination_metric,
                            "physical_plausibility": plausibility_metric,
                        },
                        weights={
                            "charge_neutrality": charge_weight,
                            "min_distance": distance_weight,
                            # "coordination": coordination_weight,
                            "physical_plausibility": plausibility_weight,
                        },
                    )
                },
                weights={"composite": 1.0},
                aggregation_method="weighted_mean",
            ),
        }

        # Create benchmark metadata
        benchmark_metadata = {
            "version": "0.1.0",
            "category": "validity",
            "weights": {
                "charge_neutrality": charge_weight,
                "interatomic_distance": distance_weight,
                # "coordination_environment": coordination_weight,
                "physical_plausibility": plausibility_weight,
            },
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
        """Aggregate results from multiple evaluators into final scores.

        For the validity benchmark, we use the overall validity score as the
        primary result, and include individual component scores.

        Parameters
        ----------
        evaluator_results : dict[str, EvaluationResult]
            Results from each evaluator.

        Returns
        -------
        dict[str, float]
            Final aggregated scores.
        """
        # Extract overall validity score
        overall_score = evaluator_results.get("overall_validity", {}).get(
            "combined_value", 0.0
        )

        # Extract individual component scores
        charge_score = evaluator_results.get("charge_neutrality", {}).get(
            "combined_value", 0.0
        )
        distance_score = evaluator_results.get("interatomic_distance", {}).get(
            "combined_value", 0.0
        )
        # coordination_score = evaluator_results.get("coordination_environment", {}).get(
        #     "combined_value", 0.0
        # )
        plausibility_score = evaluator_results.get("physical_plausibility", {}).get(
            "combined_value", 0.0
        )

        # Get the number of structures with perfect validity
        overall_validity_ratio = evaluator_results.get("overall_validity", {})
        overall_validity_ratio = overall_validity_ratio.get("metric_results", {})
        overall_validity_ratio = overall_validity_ratio.get("composite", {})
        if isinstance(overall_validity_ratio, dict):
            overall_validity_ratio = overall_validity_ratio.get("metrics", {})
        else:
            overall_validity_ratio = overall_validity_ratio.metrics
        overall_validity_ratio = overall_validity_ratio.get(
            "valid_structures_ratio", 0.0
        )

        return {
            "overall_validity_score": overall_score,
            "charge_neutrality_score": charge_score,
            "interatomic_distance_score": distance_score,
            # "coordination_environment_score": coordination_score,
            "physical_plausibility_score": plausibility_score,
            "valid_structures_ratio": overall_validity_ratio,
        }
