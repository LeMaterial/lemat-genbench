"""Fixed validity benchmark for material structures.

This module implements a benchmark that evaluates the validity of
generated material structures using fundamental validity criteria.
The benchmark reports ratios and counts for interpretability.
"""

from typing import Any, Dict

from lemat_genbench.benchmarks.base import BaseBenchmark
from lemat_genbench.evaluator import EvaluatorConfig
from lemat_genbench.metrics.validity_metrics import (
    ChargeNeutralityMetric,
    MinimumInteratomicDistanceMetric,
    OverallValidityMetric,
    PhysicalPlausibilityMetric,
)


class ValidityBenchmark(BaseBenchmark):
    """Benchmark for evaluating the validity of generated material structures.
    
    Reports validity ratios for each individual metric and overall validity.
    Overall validity requires passing ALL individual checks.
    """

    def __init__(
        self,
        charge_tolerance: float = 0.1,
        distance_scaling: float = 0.5,
        min_density: float = 1.0,
        max_density: float = 25.0,
        check_format: bool = True,
        check_symmetry: bool = True,
        name: str = "ValidityBenchmark",
        description: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ):
        if description is None:
            description = (
                "Evaluates the validity of crystal structures based on physical and "
                "chemical principles including charge neutrality, interatomic distances, "
                "and physical plausibility. Reports ratios and counts for interpretability."
            )

        # Initialize individual metrics
        charge_metric = ChargeNeutralityMetric(tolerance=charge_tolerance)
        distance_metric = MinimumInteratomicDistanceMetric(scaling_factor=distance_scaling)
        plausibility_metric = PhysicalPlausibilityMetric(
            min_density=min_density,
            max_density=max_density,
            check_format=check_format,
            check_symmetry=check_symmetry,
        )
        overall_metric = OverallValidityMetric(
            charge_tolerance=charge_tolerance,
            distance_scaling=distance_scaling,
            min_density=min_density,
            max_density=max_density,
            check_format=check_format,
            check_symmetry=check_symmetry,
        )

        # Set up evaluators - each metric gets its own evaluator
        evaluator_configs = {
            "charge_neutrality": EvaluatorConfig(
                name="Charge Neutrality",
                description="Evaluates charge balance in structures",
                metrics={"charge_neutrality": charge_metric},
                weights={"charge_neutrality": 1.0},
                aggregation_method="weighted_mean",
            ),
            "interatomic_distance": EvaluatorConfig(
                name="Interatomic Distance",
                description="Evaluates minimum distances between atoms",
                metrics={"min_distance": distance_metric},
                weights={"min_distance": 1.0},
                aggregation_method="weighted_mean",
            ),
            "physical_plausibility": EvaluatorConfig(
                name="Physical Plausibility",
                description="Evaluates physical plausibility of structures",
                metrics={"plausibility": plausibility_metric},
                weights={"plausibility": 1.0},
                aggregation_method="weighted_mean",
            ),
            "overall_validity": EvaluatorConfig(
                name="Overall Validity",
                description="Overall validity requiring all checks to pass",
                metrics={"overall": overall_metric},
                weights={"overall": 1.0},
                aggregation_method="weighted_mean",
            ),
        }

        super().__init__(
            name=name,
            description=description,
            evaluator_configs=evaluator_configs,
            metadata=metadata,
        )

    def aggregate_evaluator_results(
        self, evaluator_results: dict[str, dict]
    ) -> dict[str, float]:
        """Aggregate results from individual validity evaluators.
        
        Returns ratios and counts for each validity check and overall validity.
        """
        final_scores = {}
        
        # Helper function to safely extract metric values
        def extract_metric_value(evaluator_name: str, metric_key: str, default=0.0):
            try:
                evaluator_result = evaluator_results.get(evaluator_name, {})
                metric_results = evaluator_result.get("metric_results", {})
                
                # The metric_results should contain metric objects with .metrics attribute
                for metric_name, metric_result in metric_results.items():
                    if hasattr(metric_result, 'metrics'):
                        return metric_result.metrics.get(metric_key, default)
                    elif isinstance(metric_result, dict):
                        return metric_result.get(metric_key, default)
                return default
            except Exception:
                return default
        
        # Extract total structures count (should be same for all)
        total_structures = extract_metric_value("charge_neutrality", "total_structures", 0)
        if total_structures == 0:
            total_structures = extract_metric_value("interatomic_distance", "total_structures", 0)
        if total_structures == 0:
            total_structures = extract_metric_value("physical_plausibility", "total_structures", 0)
        if total_structures == 0:
            total_structures = extract_metric_value("overall_validity", "total_structures", 0)
        
        # Extract individual metric results
        charge_ratio = extract_metric_value("charge_neutrality", "charge_neutral_ratio", 0.0)
        charge_count = extract_metric_value("charge_neutrality", "charge_neutral_count", 0)
        avg_charge_deviation = extract_metric_value("charge_neutrality", "avg_charge_deviation", float("nan"))
        
        distance_ratio = extract_metric_value("interatomic_distance", "distance_valid_ratio", 0.0)
        distance_count = extract_metric_value("interatomic_distance", "distance_valid_count", 0)
        
        plausibility_ratio = extract_metric_value("physical_plausibility", "plausibility_valid_ratio", 0.0)
        plausibility_count = extract_metric_value("physical_plausibility", "plausibility_valid_count", 0)
        
        overall_ratio = extract_metric_value("overall_validity", "overall_valid_ratio", 0.0)
        overall_count = extract_metric_value("overall_validity", "overall_valid_count", 0)
        
        # Compile final scores
        final_scores.update({
            # Individual metric ratios and counts
            "charge_neutrality_ratio": charge_ratio,
            "charge_neutrality_count": int(charge_count),
            "avg_charge_deviation": avg_charge_deviation,
            
            "interatomic_distance_ratio": distance_ratio,
            "interatomic_distance_count": int(distance_count),
            
            "physical_plausibility_ratio": plausibility_ratio,
            "physical_plausibility_count": int(plausibility_count),
            
            # Overall validity (intersection of all)
            "overall_validity_ratio": overall_ratio,
            "overall_validity_count": int(overall_count),
            
            # Summary statistics
            "total_structures": int(total_structures),
            "any_invalid_count": int(total_structures - overall_count) if total_structures > 0 else 0,
            "any_invalid_ratio": (total_structures - overall_count) / total_structures if total_structures > 0 else 0.0,
        })

        return final_scores