"""Tests for stability metrics implementation."""

import pytest
from lemat_genbench.metrics.stability_metrics import (
    MetastabilityMetric,
    StabilityMetric,
)
from pymatgen.util.testing import PymatgenTest


@pytest.fixture
def test_structures():
    """Create test structures for stability evaluation."""
    test = PymatgenTest()
    structures = [
        test.get_structure("Si"),
        test.get_structure("LiFePO4"),
    ]
    return structures


@pytest.fixture
def test_structures_with_precomputed_e_above_hull():
    """Create test structures with precomputed e_above_hull values."""
    test = PymatgenTest()
    structures = [
        test.get_structure("Si"),
        test.get_structure("LiFePO4"),
    ]

    # Add precomputed e_above_hull values to structure properties
    e_above_hull_values = [0.05, 0.0]  # Mix of stable and metastable

    for structure, e_above_hull in zip(structures, e_above_hull_values):
        structure.properties["e_above_hull"] = e_above_hull

    return structures


def test_stability_metric(test_structures_with_precomputed_e_above_hull):
    metric = StabilityMetric(
        # relaxer_config={"steps": 500, "fmax": 0.02},
    )
    result = metric(test_structures_with_precomputed_e_above_hull)

    # Check that computation ran (may have failed indices due to no MP entries)
    assert len(result.individual_values) == len(
        test_structures_with_precomputed_e_above_hull
    )

    # Check result structure
    assert "stable_ratio" in result.metrics
    assert result.primary_metric == "stable_ratio"

    assert result.metrics["stable_ratio"] >= 0.0


def test_metastability_metric(test_structures_with_precomputed_e_above_hull):
    """Test MetastabilityMetric with precomputed e_above_hull values."""
    metric = MetastabilityMetric()

    result = metric(test_structures_with_precomputed_e_above_hull)

    # Check that computation ran successfully
    assert len(result.individual_values) == len(
        test_structures_with_precomputed_e_above_hull
    )

    # Check result structure
    # assert "mean_e_above_hull" in result.metrics
    assert "metastable_ratio" in result.metrics
    assert result.primary_metric == "metastable_ratio"

    assert result.metrics["metastable_ratio"] == 1.0  # 2 out of 2 structures
    # assert (
    #     abs(result.metrics["mean_e_above_hull"] - 0.06) < 1e-6
    # )  # (0.05+0.0+0.15+0.02+0.08)/5 = 0.06

    # Check that all individual values were extracted correctly
    expected_values = [0.05, 0.0]
    assert result.individual_values == expected_values


def test_metastability_metric_missing_properties():
    """Test MetastabilityMetric with structures missing e_above_hull properties."""
    test = PymatgenTest()
    structures = [
        test.get_structure("Si"),
        test.get_structure("LiFePO4"),
    ]

    # Don't add e_above_hull properties - should return NaN values
    metric = MetastabilityMetric()
    result = metric(structures)

    # Should have NaN values for structures without properties
    import numpy as np

    assert all(np.isnan(val) for val in result.individual_values)

    # Metrics should be 0 when all values are NaN
    # assert result.metrics["metastable_ratio"] == 0.0
