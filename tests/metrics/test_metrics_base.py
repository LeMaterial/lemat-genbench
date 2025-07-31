from typing import Any

import numpy as np
import pytest
from lemat_genbench.metrics.base import BaseMetric, MetricResult
from pymatgen.core.structure import Structure


class CountAtomsMetric(BaseMetric):
    """Simple dummy metric that counts atom number in a structure"""

    def __init__(self, n_jobs=1):
        super().__init__(
            name="Atom Counter",
            description="Counts atoms in structure",
            lower_is_better=True,
            n_jobs=n_jobs,
        )

    @staticmethod
    def compute_structure(structure: Structure, **compute_args: Any) -> float:
        return len(structure)

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


class FailingMetric(BaseMetric):
    def __init__(self, n_jobs=1):
        super().__init__(
            name="failing_metric", description="Always fails", n_jobs=n_jobs
        )

    def compute_structure(self, structure: Structure) -> float:
        raise ValueError("This metric always fails")

    def aggregate_results(self, values: list[float]) -> dict:
        # For failing metric, we should return the metric name as key
        return {
            "metrics": {self.name: float("nan")},
            "primary_metric": self.name,
            "uncertainties": {},
        }


@pytest.fixture
def test_structures():
    structures = []
    for n in range(2, 5):
        structure = Structure(
            lattice=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            species=["H"] * (n * n * n),
            coords=[
                [x / n, y / n, z / n]
                for x in range(n)
                for y in range(n)
                for z in range(n)
            ],
        )
        structures.append(structure)
    return structures


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_count_atoms_metric_jobs(test_structures, n_jobs):
    metric = CountAtomsMetric(n_jobs=n_jobs)
    result = metric.compute(test_structures)

    assert isinstance(result, MetricResult)
    assert result.n_structures == 3
    assert len(result.individual_values) == 3
    assert result.individual_values == [8, 27, 64]
    assert result.metrics["mean_atoms"] == np.mean([8, 27, 64])
    assert result.metrics["total_atoms"] == sum([8, 27, 64])
    assert len(result.failed_indices) == 0
    assert result.primary_metric == "mean_atoms"


def test_failing_metric(test_structures):
    metric = FailingMetric()
    result = metric.compute(test_structures)

    assert isinstance(result, MetricResult)
    assert result.n_structures == 3
    assert len(result.failed_indices) == 3
    assert all(np.isnan(v) for v in result.individual_values)
    assert np.isnan(result.metrics["failing_metric"])
    assert len(result.warnings) == 3


def test_empty_structures():
    metric = CountAtomsMetric()
    result = metric.compute([])

    assert isinstance(result, MetricResult)
    assert result.n_structures == 0
    assert len(result.individual_values) == 0
    assert np.isnan(result.metrics[result.primary_metric])
    assert len(result.failed_indices) == 0


def test_metric_properties():
    metric = CountAtomsMetric()

    assert metric.name == "Atom Counter"
    assert metric.description == "Counts atoms in structure"
    assert metric.config.lower_is_better is True

    config_dict = metric.config.to_dict()
    assert config_dict["name"] == "Atom Counter"
    assert config_dict["description"] == "Counts atoms in structure"
    assert config_dict["lower_is_better"] is True


def test_metric_call_interface(test_structures):
    metric = CountAtomsMetric()
    # Test the __call__ interface
    result = metric(test_structures)

    assert isinstance(result, MetricResult)
    assert result.n_structures == 3
    assert result.individual_values == [8, 27, 64]
