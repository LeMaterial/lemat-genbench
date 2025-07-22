"""Diversity benchmark for material structures.

This module implements a benchmark that computes a series of diversity metrics for a sample of
structures.
"""

from typing import Any, Dict

import numpy as np

from lematerial_forgebench.benchmarks.base import BaseBenchmark
from lematerial_forgebench.evaluator import EvaluationResult, EvaluatorConfig
from lematerial_forgebench.metrics.diversity_metric import (
    ElementDiversityMetric,
    PhysicalSizeComponentMetric,
    SiteNumberComponentMetric,
    SpaceGroupDiversityMetric,
)
from lematerial_forgebench.utils.distribution_utils import safe_float


class DiversityBenchmark(BaseBenchmark):
    """Benchmark for evaluating metrics of structural diversity across a series of structural
    parameters."""

    def __init__(
        self,
        name: str = "DiversityBenchmark",
        description: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ):
        """Initialize the diversity benchmark.

        Parameters
        ----------
        name : str
            Name of the benchmark.
        description : str, optional
            Description of the benchmark.
        metadata : dict, optional
            Additional metadata for the benchmark.
        """
        if description is None:
            description = "Computes the diversity of a sample of structures."

        # Initialize the ElementDiversityMetric
        element_diversity = ElementDiversityMetric()
        # Set up evaluator configs
        evaluator_configs = {
            "element_diversity": EvaluatorConfig(
                name="element_diversity",
                description="Calculates the element diveristy of a sample of structures",
                metrics={"element_diversity": element_diversity},
                weights={"element_diversity": 1.0},
                aggregation_method="weighted_mean",
            )
        }

        # Initialize the SpaceGroupDiversityMetric
        space_group_diversity = SpaceGroupDiversityMetric()
        # Set up evaluator configs
        evaluator_configs["space_group_diversity"] = EvaluatorConfig(
            name="space_group_diversity",
            description="Calculates the space group diveristy of a sample of structures",
            metrics={"space_group_diversity": space_group_diversity},
            weights={"space_group_diversity": 1.0},
            aggregation_method="weighted_mean",
        )

        # Initialize the SiteNumberComponentMetric
        site_number_diversity = SiteNumberComponentMetric()
        # Set up evaluator configs
        evaluator_configs["site_number_diversity"] = EvaluatorConfig(
            name="site_number_diversity",
            description="Calculates the space group diveristy of a sample of structures",
            metrics={"site_number_diversity": site_number_diversity},
            weights={"site_number_diversity": 1.0},
            aggregation_method="weighted_mean",
        )

        # Initialize the PhysicalSizeComponentMetric
        physical_size_diversity = PhysicalSizeComponentMetric()
        physical_size_diversity._init_reference_packing_factor_histogram()
        # Set up evaluator configs
        evaluator_configs["physical_size_diversity"] = EvaluatorConfig(
            name="physical_size_diversity",
            description="Calculates the space group diveristy of a sample of structures",
            metrics={"physical_size_diversity": physical_size_diversity},
            weights={"physical_size_diversity": 1.0},
            aggregation_method="weighted_mean",
        )

        # Create benchmark metadata
        benchmark_metadata = {
            "version": "0.1.0",
            "category": "diversity",
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
            "element_diversity": np.nan,
            "space_group_diversity": np.nan,
            "site_number_diversity": np.nan,
            "physical_size_diversity": np.nan,
        }

        # Extract element_diversity results
        element_diversity_results = evaluator_results.get("element_diversity")
        if element_diversity_results:
            final_scores["element_diversity"] = element_diversity_results.get(
                "combined_value"
            )

        # Extract space_group_diversity results
        space_group_diversity_results = evaluator_results.get("space_group_diversity")
        if space_group_diversity_results:
            final_scores["space_group_diversity"] = safe_float(
                space_group_diversity_results.get("combined_value")
            )

        # Extract site_number_diversity results
        site_number_diversity_results = evaluator_results.get("site_number_diversity")
        if site_number_diversity_results:
            final_scores["site_number_diversity"] = safe_float(
                site_number_diversity_results.get("combined_value")
            )

        # Extract physical_size_diversity results
        physical_size_diversity_results = evaluator_results.get(
            "physical_size_diversity"
        )
        if physical_size_diversity_results:
            final_scores["physical_size_diversity"] = safe_float(
                physical_size_diversity_results.get("combined_value")
            )

        return final_scores


if __name__ == "__main__":
    from pymatgen.util.testing import PymatgenTest

    test = PymatgenTest()

    structures = [
        test.get_structure("Si"),
        test.get_structure("LiFePO4"),
    ]

    benchmark = DiversityBenchmark()
    benchmark_result = benchmark.evaluate(structures)

    print("element_diversity")
    print(
        benchmark_result.evaluator_results["element_diversity"]["metric_results"][
            "element_diversity"
        ].metrics
    )
    print("space_group_diversity")
    print(
        benchmark_result.evaluator_results["space_group_diversity"]["metric_results"][
            "space_group_diversity"
        ].metrics
    )
    print("site_number_diversity")
    print(
        benchmark_result.evaluator_results["site_number_diversity"]["metric_results"][
            "site_number_diversity"
        ].metrics
    )
    print("physical_size_diversity")
    print(
        benchmark_result.evaluator_results["physical_size_diversity"]["metric_results"][
            "physical_size_diversity"
        ].metrics
    )
