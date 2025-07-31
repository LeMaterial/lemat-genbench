"""Tests for the base benchmark interface."""

from typing import Any

import numpy as np
import pytest
from pymatgen.core.structure import Lattice, Structure

from lemat_genbench.benchmarks.base import BaseBenchmark, EvaluatorConfig
from lemat_genbench.metrics import BaseMetric


class DummyMetric(BaseMetric):
    """A dummy metric that always returns a fixed value."""

    def __init__(self, return_value: float = 1.0):
        self.return_value = return_value
        super().__init__(
            name="dummy_metric",
            description="A dummy metric for testing",
        )

    def _get_compute_attributes(self) -> dict[str, Any]:
        return {"return_value": self.return_value}

    @staticmethod
    def compute_structure(structure: Structure, **compute_args: Any) -> float:
        return compute_args["return_value"]

    def aggregate_results(self, values: list[float]) -> dict:
        valid_values = [v for v in values if not np.isnan(v)]
        if not valid_values:
            return {
                "metrics": {"mean_atoms": float("nan")},
                "primary_metric": "mean_atoms",
                "uncertainties": {},
            }

        mean_value = np.mean(valid_values)
        std_value = np.std(valid_values) if len(valid_values) > 1 else 0.0
        total_value = sum(valid_values)

        return {
            "metrics": {"mean_atoms": mean_value, "total_atoms": total_value},
            "primary_metric": "mean_atoms",
            "uncertainties": {"mean_atoms": {"std": std_value}},
        }


class SimpleBenchmark(BaseBenchmark):
    """A simple benchmark implementation for testing."""

    def evaluate(
        self,
        structures: list[Structure],
    ) -> dict:
        """Run the complete benchmark evaluation.

        Parameters
        ----------
        structures : list[Structure]
            Structures to evaluate

        Returns
        -------
        dict
            Complete benchmark results
        """
        if not structures:
            raise ValueError("Cannot evaluate empty list of structures")

        return super().evaluate(structures)

    def aggregate_evaluator_results(
        self, evaluator_results: dict[str, dict[str, float]]
    ) -> dict[str, float]:
        """Simple mean aggregation of evaluator results."""
        # Get valid combined values (not None)
        valid_values = [
            res["combined_value"]
            for res in evaluator_results.values()
            if res["combined_value"] is not None
        ]

        if not valid_values:
            return {"mean_score": float("nan"), "max_score": float("nan")}

        final_scores = {
            "mean_score": np.mean(valid_values),
            "max_score": np.max(valid_values),
        }
        return final_scores


def test_benchmark_initialization():
    """Test that benchmark initializes correctly with configs."""
    evaluator_configs = {
        "eval1": EvaluatorConfig(
            name="Evaluator 1",
            description="First test evaluator",
            metrics={"dummy": DummyMetric(1.0)},
            weights={"dummy": 1.0},
            aggregation_method="weighted_mean",
        ),
        "eval2": EvaluatorConfig(
            name="Evaluator 2",
            description="Second test evaluator",
            metrics={"dummy1": DummyMetric(0.5), "dummy2": DummyMetric(0.8)},
            weights={"dummy1": 0.6, "dummy2": 0.4},
            aggregation_method="weighted_mean",
        ),
    }

    benchmark = SimpleBenchmark(
        name="test_benchmark",
        description="A test benchmark",
        evaluator_configs=evaluator_configs,
        metadata={"version": "0.1.0"},
    )

    assert benchmark.config.name == "test_benchmark"
    assert len(benchmark.evaluators) == 2
    assert "eval1" in benchmark.evaluators
    assert "eval2" in benchmark.evaluators


def test_benchmark_evaluation():
    """Test the complete evaluation flow with dummy structures."""
    # Create simple evaluator config
    evaluator_configs = {
        "eval1": EvaluatorConfig(
            name="Evaluator 1",
            description="First test evaluator",
            metrics={"dummy": DummyMetric(0.7)},
            weights={"dummy": 1.0},
            aggregation_method="weighted_mean",
        ),
        "eval2": EvaluatorConfig(
            name="Evaluator 2",
            description="Second test evaluator",
            metrics={"dummy": DummyMetric(0.9)},
            weights={"dummy": 1.0},
            aggregation_method="weighted_mean",
        ),
    }

    benchmark = SimpleBenchmark(
        name="test_benchmark",
        description="A test benchmark",
        evaluator_configs=evaluator_configs,
    )

    # Create dummy structures
    lattice = Lattice.cubic(4.0)
    structure1 = Structure(lattice=lattice, species=["Si"], coords=[[0, 0, 0]])
    structure2 = Structure(lattice=lattice, species=["Ge"], coords=[[0, 0, 0]])

    # Run evaluation
    result = benchmark.evaluate([structure1, structure2])

    # Check results format and values
    assert isinstance(result.evaluator_results, dict)
    assert len(result.evaluator_results) == 2
    assert isinstance(result.final_scores, dict)
    assert "mean_score" in result.final_scores
    assert "max_score" in result.final_scores

    assert result.final_scores["mean_score"] == pytest.approx(0.8)
    assert result.final_scores["max_score"] == pytest.approx(0.9)

    assert result.metadata["benchmark_name"] == "test_benchmark"
    assert result.metadata["n_structures"] == 2
    assert result.metadata["n_evaluators"] == 2


def test_benchmark_with_empty_structures():
    """Test that benchmark handles empty structure list appropriately."""
    evaluator_configs = {
        "eval1": EvaluatorConfig(
            name="Evaluator 1",
            description="Test evaluator",
            metrics={"dummy": DummyMetric(1.0)},
            weights={"dummy": 1.0},
            aggregation_method="weighted_mean",
        )
    }

    benchmark = SimpleBenchmark(
        name="test_benchmark",
        description="A test benchmark",
        evaluator_configs=evaluator_configs,
    )

    with pytest.raises(ValueError):
        benchmark.evaluate([])
