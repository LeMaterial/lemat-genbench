"""HHI (Herfindahl-Hirschman Index) benchmark for material structures.

This module implements a benchmark that evaluates the supply risk concentration
of generated material structures using HHI metrics for both production and 
reserves.
"""

import math
from typing import Any, Dict

from lematerial_forgebench.benchmarks.base import BaseBenchmark
from lematerial_forgebench.evaluator import EvaluatorConfig
from lematerial_forgebench.metrics.hhi_metrics import (
    HHIProductionMetric,
    HHIReserveMetric,
)
from lematerial_forgebench.utils.distribution_utils import safe_float


class HHIBenchmark(BaseBenchmark):
    """Benchmark for evaluating the supply risk concentration of generated 
    material structures."""

    def __init__(
        self,
        production_weight: float = 0.25,
        reserve_weight: float = 0.75,
        # Default weights prioritize long-term supply security over short-term 
        # market dynamics. Reserve concentration (0.75) reflects fundamental 
        # geological availability and is harder to change, while production 
        # concentration (0.25) can be adjusted through investment and trade 
        # diversification. This weighting is optimal for materials discovery 
        # where ong-term element availability is more critical than current 
        # market conditions.
        scale_to_0_10: bool = True,
        name: str = "HHIBenchmark",
        description: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ):
        """Initialize the HHI benchmark.

        Parameters
        ----------
        production_weight : float, default=0.25
            Weight for production-based HHI metric.
        reserve_weight : float, default=0.75
            Weight for reserve-based HHI metric.
        scale_to_0_10 : bool, default=True
            Whether to scale HHI values to 0-10 range (True) or keep 0-10000 
            range (False).
        name : str
            Name of the benchmark.
        description : str, optional
            Description of the benchmark.
        metadata : dict, optional
            Additional metadata for the benchmark.
        """
        if description is None:
            description = (
                "Evaluates the supply risk concentration of crystal structures "
                "using Herfindahl-Hirschman Index (HHI) for both production "
                "and reserves"
            )

        # Normalize weights
        total_weight = production_weight + reserve_weight
        if total_weight > 0:
            production_weight = production_weight / total_weight
            reserve_weight = reserve_weight / total_weight
        else:
            production_weight = 0.25
            reserve_weight = 0.75

        # Initialize the HHI metrics
        production_metric = HHIProductionMetric(scale_to_0_10=scale_to_0_10)
        reserve_metric = HHIReserveMetric(scale_to_0_10=scale_to_0_10)

        # Set up evaluator configs
        evaluator_configs = {
            "hhi_production": EvaluatorConfig(
                name="hhi_production",
                description=(
                    "Evaluates supply risk based on production concentration"
                ),
                metrics={"hhi_production": production_metric},
                weights={"hhi_production": 1.0},
                aggregation_method="weighted_mean",
            ),
            "hhi_reserve": EvaluatorConfig(
                name="hhi_reserve",
                description=(
                    "Evaluates supply risk based on reserve concentration"
                ),
                metrics={"hhi_reserve": reserve_metric},
                weights={"hhi_reserve": 1.0},
                aggregation_method="weighted_mean",
            ),
        }

        # Initialize metadata
        if metadata is None:
            metadata = {}

        metadata.update(
            {
                "version": "0.1.0",
                "category": "supply_risk",
                "scale_to_0_10": scale_to_0_10,
                "production_weight": production_weight,
                "reserve_weight": reserve_weight,
            }
        )

        # Initialize base benchmark
        super().__init__(
            name=name,
            description=description,
            evaluator_configs=evaluator_configs,
            metadata=metadata,
        )

        # Store weights for aggregation
        self.production_weight = production_weight
        self.reserve_weight = reserve_weight

    def aggregate_evaluator_results(
        self, evaluator_results: dict[str, dict[str, Any]]
    ) -> dict[str, float]:
        """Aggregate results from HHI evaluators into final scores.

        Parameters
        ----------
        evaluator_results : dict[str, dict[str, Any]]
            Results from each evaluator containing combined_value and 
            metric_results in the format created by BaseBenchmark.evaluate()

        Returns
        -------
        dict[str, float]
            Final aggregated scores including individual and combined HHI 
            metrics
        """
        scores = {}

        # Extract production HHI results
        production_result = evaluator_results.get("hhi_production", {})
        production_combined = safe_float(
            production_result.get("combined_value")
        )

        # Get detailed metrics from the metric_results if available
        production_metrics = {}
        if "metric_results" in production_result:
            hhi_prod_result = production_result["metric_results"].get(
                "hhi_production"
            )
            if hhi_prod_result:
                if hasattr(hhi_prod_result, "metrics"):
                    production_metrics = hhi_prod_result.metrics
                elif (
                    isinstance(hhi_prod_result, dict)
                    and "metrics" in hhi_prod_result
                ):
                    production_metrics = hhi_prod_result["metrics"]

        # Extract reserve HHI results
        reserve_result = evaluator_results.get("hhi_reserve", {})
        reserve_combined = safe_float(reserve_result.get("combined_value"))

        # Get detailed metrics from the metric_results if available
        reserve_metrics = {}
        if "metric_results" in reserve_result:
            hhi_res_result = reserve_result["metric_results"].get(
                "hhi_reserve"
            )
            if hhi_res_result:
                if hasattr(hhi_res_result, "metrics"):
                    reserve_metrics = hhi_res_result.metrics
                elif (
                    isinstance(hhi_res_result, dict)
                    and "metrics" in hhi_res_result
                ):
                    reserve_metrics = hhi_res_result["metrics"]

        # Individual HHI scores
        scores["hhi_production_mean"] = production_combined
        scores["hhi_reserve_mean"] = reserve_combined

        # Add detailed production metrics if available
        if production_metrics:
            scores.update(
                {
                    f"hhi_production_{k}": v
                    for k, v in production_metrics.items()
                    if k != "hhiproduction_mean"  # Avoid duplication
                }
            )

        # Add detailed reserve metrics if available
        if reserve_metrics:
            scores.update(
                {
                    f"hhi_reserve_{k}": v
                    for k, v in reserve_metrics.items()
                    if k != "hhireserve_mean"  # Avoid duplication
                }
            )

        # Compute weighted average of production and reserve HHI
        # Handle None values properly
        def is_none_or_nan(value):
            """Check if value is None or NaN."""
            if value is None:
                return True
            if isinstance(value, float) and math.isnan(value):
                return True
            return False

        production_invalid = is_none_or_nan(production_combined)
        reserve_invalid = is_none_or_nan(reserve_combined)

        if not (production_invalid and reserve_invalid):
            if production_invalid:
                scores["hhi_combined_mean"] = reserve_combined
            elif reserve_invalid:
                scores["hhi_combined_mean"] = production_combined
            else:
                scores["hhi_combined_mean"] = (
                    self.production_weight * production_combined
                    + self.reserve_weight * reserve_combined
                )
        else:
            scores["hhi_combined_mean"] = None

        # Risk assessment categories (for scaled values)
        if self.config.metadata.get("scale_to_0_10", True):
            # Add risk category counts for combined HHI
            combined_hhi = scores["hhi_combined_mean"]
            if not is_none_or_nan(combined_hhi):
                scores["hhi_low_risk"] = 1.0 if combined_hhi <= 2.0 else 0.0
                scores["hhi_moderate_risk"] = (
                    1.0 if 2.0 < combined_hhi <= 5.0 else 0.0
                )
                scores["hhi_high_risk"] = 1.0 if combined_hhi > 5.0 else 0.0
            else:
                scores["hhi_low_risk"] = None
                scores["hhi_moderate_risk"] = None
                scores["hhi_high_risk"] = None

        return scores