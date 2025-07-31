import numpy as np
import pytest
from lemat_genbench.evaluator import MetricEvaluator
from lemat_genbench.metrics.base import BaseMetric
from pymatgen.core.structure import Structure


class DummyMetricA(BaseMetric):
    def __init__(self, n_jobs=1):
        super().__init__(
            name="DummyMetricA",
            description="A dummy metric that returns the number of atoms",
            lower_is_better=True,
            n_jobs=n_jobs,
        )

    @staticmethod
    def compute_structure(structure: Structure, **compute_args) -> float:
        return len(structure)

    def aggregate_results(self, values: list[float]) -> dict:
        valid_values = [v for v in values if not np.isnan(v)]
        mean_value = np.mean(valid_values) if valid_values else float("nan")
        std_value = np.std(valid_values) if len(valid_values) > 1 else 0.0

        return {
            "metrics": {"DummyMetricA": mean_value},
            "primary_metric": "DummyMetricA",
            "uncertainties": {"DummyMetricA": {"std": std_value}},
        }


class DummyMetricB(BaseMetric):
    def __init__(self, n_jobs=1):
        super().__init__(
            name="DummyMetricB",
            description="A dummy metric that returns the volume",
            lower_is_better=False,
            n_jobs=n_jobs,
        )

    @staticmethod
    def compute_structure(structure: Structure, **compute_args) -> float:
        return structure.volume

    def aggregate_results(self, values: list[float]) -> dict:
        valid_values = [v for v in values if not np.isnan(v)]
        mean_value = np.mean(valid_values) if valid_values else float("nan")
        std_value = np.std(valid_values) if len(valid_values) > 1 else 0.0

        return {
            "metrics": {"DummyMetricB": mean_value},
            "primary_metric": "DummyMetricB",
            "uncertainties": {"DummyMetricB": {"std": std_value}},
        }


class FailingMetric(BaseMetric):
    def __init__(self, n_jobs=1):
        super().__init__(
            name="FailingMetric",
            description="A metric that always fails",
            n_jobs=n_jobs,
        )

    @staticmethod
    def compute_structure(structure: Structure, **compute_args) -> float:
        raise ValueError("This metric always fails")

    def aggregate_results(self, values: list[float]) -> dict:
        return {
            "metrics": {"FailingMetric": float("nan")},
            "primary_metric": "FailingMetric",
            "uncertainties": {},
        }


def test_evaluator_basic():
    """Test basic functionality of the evaluator"""
    lattice = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    coords = [[0, 0, 0]]
    structure = Structure(lattice, ["H"], coords)
    structures = [structure] * 3

    metrics = {"DummyMetricA": DummyMetricA(), "DummyMetricB": DummyMetricB()}
    weights = {"DummyMetricA": 0.6, "DummyMetricB": 0.4}

    evaluator = MetricEvaluator(
        metrics=metrics,
        weights=weights,
        aggregation_method="weighted_mean",
    )

    result = evaluator.evaluate(structures)

    assert len(result.metric_results) == 2
    assert "DummyMetricA" in result.metric_results
    assert "DummyMetricB" in result.metric_results
    assert result.combined_value is not None
    assert result.uncertainties is not None
    assert result.computation_time > 0


def test_evaluator_failing_metric():
    """Test handling of failing metrics"""
    metrics = {"DummyMetricA": DummyMetricA(), "FailingMetric": FailingMetric()}
    weights = {"DummyMetricA": 0.7, "FailingMetric": 0.3}

    evaluator = MetricEvaluator(
        metrics=metrics, weights=weights, aggregation_method="weighted_mean"
    )

    structure = Structure([[1, 0, 0], [0, 1, 0], [0, 0, 1]], ["H"], [[0, 0, 0]])
    structures = [structure] * 2

    result = evaluator.evaluate(structures)

    assert len(result.metric_results) == 2
    assert np.isnan(result.metric_results["FailingMetric"].value)
    assert (
        len(result.metric_results["FailingMetric"].warnings) == 2
    )  # both structures failed


def test_evaluator_different_aggregations():
    """Test different aggregation methods"""
    metrics = {"DummyMetricA": DummyMetricA(), "DummyMetricB": DummyMetricB()}
    weights = {"DummyMetricA": 0.5, "DummyMetricB": 0.5}
    structure = Structure([[1, 0, 0], [0, 1, 0], [0, 0, 1]], ["H"], [[0, 0, 0]])
    structures = [structure] * 2

    for method in ["weighted_sum", "weighted_mean", "min", "max"]:
        evaluator = MetricEvaluator(
            metrics=metrics, weights=weights, aggregation_method=method
        )
        result = evaluator.evaluate(structures)
        assert result.combined_value is not None
        assert result.uncertainties is not None


def test_evaluator_validation():
    metrics = {"DummyMetricA": DummyMetricA(), "DummyMetricB": DummyMetricB()}

    # missing weight
    with pytest.raises(ValueError):
        MetricEvaluator(metrics=metrics, weights={"DummyMetricA": 0.5})

    # extra weight
    with pytest.raises(ValueError):
        MetricEvaluator(
            metrics=metrics,
            weights={
                "DummyMetricA": 0.5,
                "DummyMetricB": 0.3,
                "NonExistentMetric": 0.2,
            },
        )

    # invalid aggregation method
    with pytest.raises(ValueError):
        MetricEvaluator(
            metrics=metrics,
            weights={"DummyMetricA": 0.5, "DummyMetricB": 0.5},
            aggregation_method="invalid_method",
        )
