"""Tests for validity metrics implementation."""

import numpy as np
import pytest
from pymatgen.core.structure import Structure
from pymatgen.util.testing import PymatgenTest

from lemat_genbench.metrics.validity_metrics import (
    ChargeNeutralityMetric,
    MinimumInteratomicDistanceMetric,
    OverallValidityMetric,
    PhysicalPlausibilityMetric,
)


@pytest.fixture
def valid_structures():
    """Create valid test structures."""
    test = PymatgenTest()
    structures = [
        test.get_structure("Si"),  # Silicon
        test.get_structure("LiFePO4"),  # Lithium iron phosphate
        test.get_structure("CsCl"),  # Cesium chloride
    ]
    return structures


@pytest.fixture
def invalid_structures():
    """Create structures with validity issues."""
    # Create Si structure with extremely compressed lattice
    test = PymatgenTest()
    si = test.get_structure("Si")

    # Create compressed structure - use proper method to scale the lattice
    compressed_lattice = si.lattice.scale(0.1)
    compressed_si = Structure(compressed_lattice, si.species, si.frac_coords)

    # Create structure with atoms too close
    overlapping_si = si.copy()
    for i in range(len(overlapping_si)):
        if i > 0:
            displacement = overlapping_si[i].coords - overlapping_si[0].coords
            displacement /= np.linalg.norm(displacement)
            overlapping_si[i].coords = overlapping_si[0].coords + displacement * 0.5
            break

    return [compressed_si, overlapping_si]


def test_charge_neutrality_metric(valid_structures):
    """Test ChargeNeutralityMetric on valid structures."""
    metric = ChargeNeutralityMetric()
    result = metric.compute(valid_structures)

    # Check computation didn't fail
    assert len(result.failed_indices) == 0

    # Check result structure - updated metric names
    assert "charge_neutral_ratio" in result.metrics
    assert "charge_neutral_count" in result.metrics
    assert "avg_charge_deviation" in result.metrics
    assert "total_structures" in result.metrics
    assert result.primary_metric == "charge_neutral_ratio"

    # Check values
    assert 0.0 <= result.metrics["charge_neutral_ratio"] <= 1.0
    assert result.metrics["charge_neutral_count"] <= result.metrics["total_structures"]
    assert result.metrics["total_structures"] == len(valid_structures)

    # Check one value per structure
    assert len(result.individual_values) == len(valid_structures)
    
    # Individual values should be charge deviations (not binary)
    assert all(v >= 0.0 for v in result.individual_values if not np.isnan(v))


def test_minimum_interatomic_distance_metric(valid_structures, invalid_structures):
    """Test MinimumInteratomicDistanceMetric."""
    metric = MinimumInteratomicDistanceMetric()

    # Test on valid structures
    valid_result = metric.compute(valid_structures)
    
    # Check updated metric names
    assert "distance_valid_ratio" in valid_result.metrics
    assert "distance_valid_count" in valid_result.metrics
    assert "total_structures" in valid_result.metrics
    assert valid_result.primary_metric == "distance_valid_ratio"
    
    # Check values
    assert 0.0 <= valid_result.metrics["distance_valid_ratio"] <= 1.0
    assert valid_result.metrics["total_structures"] == len(valid_structures)
    
    # Individual values should be binary (0.0 or 1.0)
    assert all(v in [0.0, 1.0] for v in valid_result.individual_values if not np.isnan(v))

    # Test on structure with overlapping atoms
    invalid_result = metric.compute([invalid_structures[1]])  # The overlapping structure

    # Check that overlapping structure gets lower score
    assert (
        invalid_result.metrics["distance_valid_ratio"]
        <= valid_result.metrics["distance_valid_ratio"]
    )


def test_physical_plausibility_metric(valid_structures, invalid_structures):
    """Test PhysicalPlausibilityMetric."""
    metric = PhysicalPlausibilityMetric()

    # Test on valid structures
    valid_result = metric.compute(valid_structures)
    
    # Check updated metric names
    assert "plausibility_valid_ratio" in valid_result.metrics
    assert "plausibility_valid_count" in valid_result.metrics
    assert "total_structures" in valid_result.metrics
    assert valid_result.primary_metric == "plausibility_valid_ratio"
    
    # Check values
    assert 0.0 <= valid_result.metrics["plausibility_valid_ratio"] <= 1.0
    assert valid_result.metrics["total_structures"] == len(valid_structures)
    
    # Individual values should be binary (0.0 or 1.0)
    assert all(v in [0.0, 1.0] for v in valid_result.individual_values if not np.isnan(v))

    # Test on structure with compressed lattice (unrealistic density)
    invalid_result = metric.compute([invalid_structures[0]])  # The compressed structure

    # Check that compressed structure gets lower score
    assert (
        invalid_result.metrics["plausibility_valid_ratio"]
        <= valid_result.metrics["plausibility_valid_ratio"]
    )


def test_overall_validity_metric(valid_structures, invalid_structures):
    """Test OverallValidityMetric requires ALL checks to pass."""
    metric = OverallValidityMetric(
        charge_tolerance=0.1,
        distance_scaling=0.5,
        min_density=1.0,
        max_density=25.0,
        check_format=False,  # Skip for speed
        check_symmetry=False,
    )

    # Test on valid structures
    valid_result = metric.compute(valid_structures)
    
    # Check result structure
    assert "overall_valid_ratio" in valid_result.metrics
    assert "overall_valid_count" in valid_result.metrics
    assert "total_structures" in valid_result.metrics
    assert valid_result.primary_metric == "overall_valid_ratio"
    
    # Check values
    assert 0.0 <= valid_result.metrics["overall_valid_ratio"] <= 1.0
    assert valid_result.metrics["total_structures"] == len(valid_structures)
    
    # Individual values should be binary (0.0 or 1.0)
    assert all(v in [0.0, 1.0] for v in valid_result.individual_values if not np.isnan(v))

    # Test on invalid structures
    invalid_result = metric.compute(invalid_structures)

    # Valid structures should have higher or equal overall validity
    assert (
        valid_result.metrics["overall_valid_ratio"]
        >= invalid_result.metrics["overall_valid_ratio"]
    )


def test_charge_neutrality_tolerance():
    """Test ChargeNeutralityMetric with different tolerance."""
    # Create a structure with slight charge imbalance
    test = PymatgenTest()
    structure = test.get_structure("LiFePO4")

    # Create metrics with different tolerances
    strict_metric = ChargeNeutralityMetric(tolerance=0.01)
    lenient_metric = ChargeNeutralityMetric(tolerance=1.0)

    # Evaluate structure
    strict_result = strict_metric.compute([structure])
    lenient_result = lenient_metric.compute([structure])

    # Lenient metric should classify more structures as charge neutral
    assert (
        lenient_result.metrics["charge_neutral_ratio"]
        >= strict_result.metrics["charge_neutral_ratio"]
    )


def test_interatomic_distance_scaling():
    """Test MinimumInteratomicDistanceMetric with different scaling factors."""
    test = PymatgenTest()
    structure = test.get_structure("Si")

    # Create metrics with different scaling factors
    strict_metric = MinimumInteratomicDistanceMetric(scaling_factor=0.9)
    lenient_metric = MinimumInteratomicDistanceMetric(scaling_factor=0.3)

    # Evaluate structure
    strict_result = strict_metric.compute([structure])
    lenient_result = lenient_metric.compute([structure])

    # Lenient metric should allow atoms to be closer
    assert (
        lenient_result.metrics["distance_valid_ratio"]
        >= strict_result.metrics["distance_valid_ratio"]
    )


def test_charge_neutrality_deviation_tracking():
    """Test that ChargeNeutralityMetric tracks deviations properly."""
    test = PymatgenTest()
    structures = [test.get_structure("Si")] * 3
    
    metric = ChargeNeutralityMetric(tolerance=0.1)
    result = metric.compute(structures)
    
    # Should have deviation information
    assert "avg_charge_deviation" in result.metrics
    assert not np.isnan(result.metrics["avg_charge_deviation"]) or result.metrics["avg_charge_deviation"] >= 0
    
    # Individual values should be charge deviations
    assert len(result.individual_values) == len(structures)
    assert all(isinstance(v, (int, float)) for v in result.individual_values if not np.isnan(v))


def test_overall_validity_intersection_logic():
    """Test that overall validity is intersection of all individual checks."""
    test = PymatgenTest()
    valid_si = test.get_structure("Si")
    
    # Create structure that might fail only one check
    compressed_si = Structure(
        valid_si.lattice.scale(0.1), 
        valid_si.species, 
        valid_si.frac_coords
    )
    
    structures = [valid_si, compressed_si]
    
    # Test individual metrics
    charge_metric = ChargeNeutralityMetric(tolerance=0.1)
    distance_metric = MinimumInteratomicDistanceMetric(scaling_factor=0.5)
    plausibility_metric = PhysicalPlausibilityMetric(check_format=False, check_symmetry=False)
    overall_metric = OverallValidityMetric(
        charge_tolerance=0.1,
        distance_scaling=0.5,
        check_format=False,
        check_symmetry=False
    )
    
    charge_result = charge_metric.compute(structures)
    distance_result = distance_metric.compute(structures)
    plausibility_result = plausibility_metric.compute(structures)
    overall_result = overall_metric.compute(structures)
    
    # Overall count should be <= min of individual counts
    overall_count = overall_result.metrics["overall_valid_count"]
    charge_count = charge_result.metrics["charge_neutral_count"]
    distance_count = distance_result.metrics["distance_valid_count"]
    plausibility_count = plausibility_result.metrics["plausibility_valid_count"]
    
    assert overall_count <= min(charge_count, distance_count, plausibility_count)


def test_count_ratio_consistency():
    """Test that counts and ratios are mathematically consistent."""
    test = PymatgenTest()
    structures = [test.get_structure("Si")] * 5  # 5 identical structures
    
    metric = ChargeNeutralityMetric()
    result = metric.compute(structures)
    
    total = result.metrics["total_structures"]
    count = result.metrics["charge_neutral_count"]
    ratio = result.metrics["charge_neutral_ratio"]
    
    assert total == 5
    assert abs(ratio - (count / total)) < 1e-10


def test_physical_plausibility_checks():
    """Test individual checks in PhysicalPlausibilityMetric."""
    test = PymatgenTest()
    structure = test.get_structure("Si")
    
    # Test with different check configurations
    density_only = PhysicalPlausibilityMetric(
        check_format=False, 
        check_symmetry=False
    )
    
    all_checks = PhysicalPlausibilityMetric(
        check_format=True,
        check_symmetry=True
    )
    
    density_result = density_only.compute([structure])
    all_result = all_checks.compute([structure])
    
    # Both should work, but all_checks might be more restrictive
    assert 0.0 <= density_result.metrics["plausibility_valid_ratio"] <= 1.0
    assert 0.0 <= all_result.metrics["plausibility_valid_ratio"] <= 1.0
    assert all_result.metrics["plausibility_valid_ratio"] <= density_result.metrics["plausibility_valid_ratio"]


def test_overall_validity_parameters():
    """Test OverallValidityMetric with different parameter sets."""
    test = PymatgenTest()
    structure = test.get_structure("Si")
    
    # Strict parameters
    strict_metric = OverallValidityMetric(
        charge_tolerance=0.001,
        distance_scaling=0.9,
        min_density=2.0,
        max_density=20.0,
        check_format=True,
        check_symmetry=True
    )
    
    # Lenient parameters
    lenient_metric = OverallValidityMetric(
        charge_tolerance=1.0,
        distance_scaling=0.1,
        min_density=0.1,
        max_density=50.0,
        check_format=False,
        check_symmetry=False
    )
    
    strict_result = strict_metric.compute([structure])
    lenient_result = lenient_metric.compute([structure])
    
    # Lenient should have higher or equal validity
    assert (lenient_result.metrics["overall_valid_ratio"] >= 
            strict_result.metrics["overall_valid_ratio"])