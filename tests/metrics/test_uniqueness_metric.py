"""Tests for uniqueness metrics implementation."""

import traceback

import numpy as np
import pytest
from lemat_genbench.metrics.base import MetricResult
from lemat_genbench.metrics.uniqueness_metric import UniquenessMetric
from pymatgen.core.structure import Structure


def create_test_structures():
    """Create test structures with known duplicates."""
    structures = []

    # Structure 1: Simple cubic NaCl
    lattice1 = [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]]
    structure1 = Structure(
        lattice=lattice1,
        species=["Na", "Cl"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structures.append(structure1)

    # Structure 2: Identical to Structure 1 (duplicate)
    structure2 = Structure(
        lattice=lattice1,
        species=["Na", "Cl"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structures.append(structure2)

    # Structure 3: Different structure - CsCl
    lattice3 = [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]]
    structure3 = Structure(
        lattice=lattice3,
        species=["Cs", "Cl"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structures.append(structure3)

    # Structure 4: Another copy of Structure 1 (duplicate)
    structure4 = Structure(
        lattice=lattice1,
        species=["Na", "Cl"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structures.append(structure4)

    # Structure 5: Unique structure - LiF
    lattice5 = [[4.2, 0, 0], [0, 4.2, 0], [0, 0, 4.2]]
    structure5 = Structure(
        lattice=lattice5,
        species=["Li", "F"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structures.append(structure5)

    return structures


def create_all_unique_structures():
    """Create test structures that are all unique."""
    structures = []

    # Different compositions and lattices
    compositions_and_lattices = [
        (["Na", "Cl"], [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]]),
        (["K", "Br"], [[6.0, 0, 0], [0, 6.0, 0], [0, 0, 6.0]]),
        (["Li", "F"], [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]]),
        (["Mg", "O"], [[4.2, 0, 0], [0, 4.2, 0], [0, 0, 4.2]]),
    ]

    for species, lattice in compositions_and_lattices:
        structure = Structure(
            lattice=lattice,
            species=species,
            coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
            coords_are_cartesian=False,
        )
        structures.append(structure)

    return structures


def create_all_identical_structures(n=4):
    """Create n identical structures."""
    lattice = [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]]
    structures = []

    for _ in range(n):
        structure = Structure(
            lattice=lattice,
            species=["Na", "Cl"],
            coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
            coords_are_cartesian=False,
        )
        structures.append(structure)

    return structures


class TestUniquenessMetric:
    """Test suite for UniquenessMetric class."""

    def test_initialization(self):
        """Test UniquenessMetric initialization."""
        metric = UniquenessMetric()
        assert metric.name == "Uniqueness"
        assert metric.config.fingerprint_method == "bawl"
        assert hasattr(metric, "fingerprinter")

    def test_custom_initialization(self):
        """Test UniquenessMetric with custom parameters."""
        metric = UniquenessMetric(
            name="Custom Uniqueness",
            description="Custom description",
            fingerprint_method="bawl",
        )

        assert metric.name == "Custom Uniqueness"
        assert metric.description == "Custom description"
        assert metric.config.fingerprint_method == "bawl"

    def test_invalid_fingerprint_method(self):
        """Test error handling for invalid fingerprint method."""
        with pytest.raises(ValueError, match="Unknown fingerprint method"):
            UniquenessMetric(fingerprint_method="invalid_method")

    def test_compute_structure_fingerprint(self):
        """Test fingerprint computation for single structure."""
        metric = UniquenessMetric()
        structures = create_test_structures()

        # Test fingerprint computation
        fingerprint1 = metric._compute_structure_fingerprint(
            structures[0], fingerprinter=metric.fingerprinter
        )
        fingerprint2 = metric._compute_structure_fingerprint(
            structures[1], fingerprinter=metric.fingerprinter
        )  # Duplicate
        fingerprint3 = metric._compute_structure_fingerprint(
            structures[2], fingerprinter=metric.fingerprinter
        )  # Different

        # Duplicates should have same fingerprint
        assert fingerprint1 == fingerprint2
        # Different structures should have different fingerprints
        assert fingerprint1 != fingerprint3

        # All should be strings
        assert isinstance(fingerprint1, str)
        assert isinstance(fingerprint2, str)
        assert isinstance(fingerprint3, str)

    def test_uniqueness_with_duplicates(self):
        """Test uniqueness calculation with known duplicates."""
        metric = UniquenessMetric()
        structures = (
            create_test_structures()
        )  # 5 structures: 3 NaCl (duplicates), 1 CsCl, 1 LiF

        result = metric.compute(structures)

        # Check result type
        assert isinstance(result, MetricResult)

        # Should have evaluated all structures
        assert result.n_structures == 5

        # Check metrics exist
        assert "uniqueness_score" in result.metrics
        assert "unique_structures_count" in result.metrics
        assert "duplicate_structures_count" in result.metrics

        # With 3 identical NaCl + 1 CsCl + 1 LiF = 3 unique structures out of 5
        total_evaluated = result.metrics["total_structures_evaluated"]
        expected_unique = 3  # NaCl, CsCl, LiF
        assert result.metrics["unique_structures_count"] == expected_unique
        assert result.metrics["uniqueness_score"] == expected_unique / total_evaluated
        assert result.metrics["duplicate_structures_count"] == 2  # 2 extra NaCl

    def test_all_unique_structures(self):
        """Test uniqueness with all unique structures."""
        metric = UniquenessMetric()
        structures = create_all_unique_structures()  # 4 different structures

        result = metric.compute(structures)

        # Check result type
        assert isinstance(result, MetricResult)
        assert result.n_structures == 4

        # All structures that can be fingerprinted should be unique
        total_evaluated = result.metrics["total_structures_evaluated"]
        successful_fingerprints = (
            total_evaluated - result.metrics["failed_fingerprinting_count"]
        )

        if successful_fingerprints > 0:
            assert result.metrics["uniqueness_score"] == 1.0
            assert result.metrics["unique_structures_count"] == successful_fingerprints
            assert result.metrics["duplicate_structures_count"] == 0

    def test_all_identical_structures(self):
        """Test uniqueness with all identical structures."""
        metric = UniquenessMetric()
        structures = create_all_identical_structures(4)  # 4 identical structures

        result = metric.compute(structures)

        # Check result type
        assert isinstance(result, MetricResult)
        assert result.n_structures == 4

        # All structures that can be fingerprinted should be identical (1 unique)
        successful_fingerprints = (
            result.metrics["total_structures_evaluated"]
            - result.metrics["failed_fingerprinting_count"]
        )

        if successful_fingerprints > 0:
            assert result.metrics["unique_structures_count"] == 1
            assert (
                result.metrics["duplicate_structures_count"]
                == successful_fingerprints - 1
            )
            assert result.metrics["uniqueness_score"] == 1.0 / successful_fingerprints

    def test_empty_structures(self):
        """Test behavior with empty structure list."""
        metric = UniquenessMetric()
        result = metric.compute([])

        assert isinstance(result, MetricResult)
        assert result.n_structures == 0
        assert np.isnan(result.metrics["uniqueness_score"])
        assert result.metrics["unique_structures_count"] == 0

    def test_individual_values_assignment(self):
        """Test individual values for structures."""
        metric = UniquenessMetric()
        structures = create_test_structures()  # Mix of duplicates and unique

        result = metric.compute(structures)

        # Check individual values exist and have correct length
        assert len(result.individual_values) == 5

        # Individual values should reflect uniqueness
        # Unique structures get 1.0, duplicates get 1/count
        for i, value in enumerate(result.individual_values):
            if not np.isnan(value):
                assert 0 < value <= 1.0

    def test_parallel_computation(self):
        """Test parallel computation with n_jobs > 1."""
        metric = UniquenessMetric(n_jobs=2)
        structures = create_test_structures()

        result = metric.compute(structures)

        # Should work the same as serial computation
        assert isinstance(result, MetricResult)
        assert result.n_structures == 5
        assert "uniqueness_score" in result.metrics
        assert 0 <= result.metrics["uniqueness_score"] <= 1.0

    def test_error_handling_with_problematic_structures(self):
        """Test error handling when BAWL fails on certain structures."""
        metric = UniquenessMetric()

        # Create structures that might cause BAWL to fail
        problematic_structures = []

        # Try with superheavy elements that might not be in BAWL's database
        try:
            lattice = [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]]
            structure = Structure(
                lattice=lattice,
                species=["Og", "Ts"],  # Superheavy elements
                coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
                coords_are_cartesian=False,
            )
            problematic_structures.append(structure)
        except Exception:
            # If structure creation fails, skip this test
            pytest.skip("Could not create test structure with problematic elements")

        # Add a normal structure too
        normal_structure = Structure(
            lattice=[[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]],
            species=["Na", "Cl"],
            coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
            coords_are_cartesian=False,
        )
        problematic_structures.append(normal_structure)

        result = metric.compute(problematic_structures)

        # Should handle errors gracefully
        assert isinstance(result, MetricResult)
        assert result.n_structures == 2

        # Some structures might fail, but result should be valid
        assert 0 <= result.metrics["failed_fingerprinting_count"] <= 2
        assert result.metrics["total_structures_evaluated"] == 2

    def test_fingerprint_consistency(self):
        """Test that fingerprinting is consistent across calls."""
        metric = UniquenessMetric()
        structures = create_test_structures()[:2]  # Just first two

        # Compute twice
        result1 = metric.compute(structures)
        result2 = metric.compute(structures)

        # Results should be identical (deterministic)
        assert (
            result1.metrics["uniqueness_score"] == result2.metrics["uniqueness_score"]
        )
        assert (
            result1.metrics["unique_structures_count"]
            == result2.metrics["unique_structures_count"]
        )

    def test_callable_interface(self):
        """Test the __call__ interface inherited from BaseMetric."""
        metric = UniquenessMetric()
        structures = create_test_structures()

        # Test callable interface
        result = metric(structures)

        # Should return MetricResult same as compute()
        assert isinstance(result, MetricResult)
        assert "uniqueness_score" in result.metrics

    def test_metric_result_properties(self):
        """Test that MetricResult has all expected properties."""
        metric = UniquenessMetric()
        structures = create_test_structures()

        result = metric.compute(structures)

        # Check MetricResult properties
        assert isinstance(result, MetricResult)
        assert hasattr(result, "metrics")
        assert hasattr(result, "primary_metric")
        assert hasattr(result, "uncertainties")
        assert hasattr(result, "config")
        assert hasattr(result, "computation_time")
        assert hasattr(result, "individual_values")
        assert hasattr(result, "n_structures")
        assert hasattr(result, "failed_indices")
        assert hasattr(result, "warnings")

        # Check specific content
        assert result.primary_metric == "uniqueness_score"
        assert result.computation_time > 0

    def test_edge_case_single_structure(self):
        """Test with single structure."""
        metric = UniquenessMetric()
        structures = [create_test_structures()[0]]  # Just one structure

        result = metric.compute(structures)

        assert isinstance(result, MetricResult)
        assert result.n_structures == 1

        # Single structure should be 100% unique if fingerprinting succeeds
        if result.metrics["failed_fingerprinting_count"] == 0:
            assert result.metrics["uniqueness_score"] == 1.0
            assert result.metrics["unique_structures_count"] == 1

    def test_large_set_with_many_duplicates(self):
        """Test with a larger set containing many duplicates."""
        metric = UniquenessMetric()

        # Create 10 structures: 5 identical + 3 identical + 2 unique
        structures = []

        # 5 identical NaCl structures
        for _ in range(5):
            structures.append(create_test_structures()[0])

        # 3 identical CsCl structures
        for _ in range(3):
            structures.append(create_test_structures()[2])

        # 2 unique structures
        structures.extend(create_all_unique_structures()[:2])

        result = metric.compute(structures)

        assert isinstance(result, MetricResult)
        assert result.n_structures == 10

        # Should have 4 unique structures total if all fingerprinting succeeds
        successful = (
            result.metrics["total_structures_evaluated"]
            - result.metrics["failed_fingerprinting_count"]
        )
        if successful == 10:
            assert result.metrics["unique_structures_count"] == 3
            assert result.metrics["uniqueness_score"] == 3.0 / 10


# Manual test function for development
def manual_test():
    """Manual test for development purposes."""
    print("Running manual uniqueness metric test...")

    try:
        # Test 1: Basic functionality
        print("1. Testing basic uniqueness calculation...")
        metric = UniquenessMetric()
        structures = create_test_structures()

        print(f"Created {len(structures)} test structures")
        print("Structure compositions:")
        for i, s in enumerate(structures):
            print(f"  {i + 1}: {s.composition.reduced_formula}")

        result = metric.compute(structures)

        print("Results:")
        print(f"  Total structures: {result.metrics['total_structures_evaluated']}")
        print(f"  Unique structures: {result.metrics['unique_structures_count']}")
        print(f"  Duplicate structures: {result.metrics['duplicate_structures_count']}")
        print(
            f"  Failed fingerprinting: {result.metrics['failed_fingerprinting_count']}"
        )
        print(f"  Uniqueness score: {result.metrics['uniqueness_score']:.3f}")

        # Test 2: All unique structures
        print("\n2. Testing all unique structures...")
        unique_structures = create_all_unique_structures()
        result_unique = metric.compute(unique_structures)
        print(
            f"  Uniqueness score (all unique): {result_unique.metrics['uniqueness_score']:.3f}"
        )

        # Test 3: All identical structures
        print("\n3. Testing all identical structures...")
        identical_structures = create_all_identical_structures(4)
        result_identical = metric.compute(identical_structures)
        print(
            f"  Uniqueness score (all identical): {result_identical.metrics['uniqueness_score']:.3f}"
        )

        # Test 4: Fingerprint consistency
        print("\n4. Testing fingerprint consistency...")
        test_structure = structures[0]
        fp1 = metric._compute_structure_fingerprint(
            test_structure, fingerprinter=metric.fingerprinter
        )
        fp2 = metric._compute_structure_fingerprint(
            test_structure, fingerprinter=metric.fingerprinter
        )
        print(f"  Fingerprint 1: {fp1[:30]}...")
        print(f"  Fingerprint 2: {fp2[:30]}...")
        print(f"  Consistent: {fp1 == fp2}")

        # Test 5: Individual values
        print("\n5. Testing individual values...")
        print("  Individual uniqueness values:")
        for i, val in enumerate(result.individual_values):
            if not np.isnan(val):
                print(f"    Structure {i + 1}: {val:.3f}")
            else:
                print(f"    Structure {i + 1}: NaN (failed)")

        print("\nAll manual tests completed successfully!")
        return True

    except Exception as e:
        print(f"Manual test failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    manual_test()
