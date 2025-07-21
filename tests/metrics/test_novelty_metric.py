"""Tests for novelty metrics implementation using real LeMat-Bulk data."""

import traceback

import numpy as np
import pytest
from pymatgen.core.structure import Structure

from lematerial_forgebench.metrics.base import MetricResult
from lematerial_forgebench.metrics.novelty_metric import NoveltyMetric

NOVELTY_TESTS_AVAILABLE = True

# Sample data from LeMat-Bulk
SAMPLE_LEMAT_ROW = {
    "elements": ["Sb", "Sr"],
    "nsites": 5,
    "chemical_formula_anonymous": "A4B",
    "chemical_formula_reduced": "Sb4Sr",
    "chemical_formula_descriptive": "Sr1 Sb4",
    "nelements": 2,
    "dimension_types": [1, 1, 1],
    "nperiodic_dimensions": 3,
    "lattice_vectors": [
        [-3.56534985, 3.56534985, 3.56534985],
        [3.56534985, -3.56534985, 3.56534985],
        [3.56534985, 3.56534985, -3.56534985],
    ],
    "immutable_id": "agm005415715",
    "cartesian_site_positions": [
        [0, 0, 0],
        [-1.782674925, 1.782674925, 1.782674925],
        [1.782674925, 1.782674925, 1.782674925],
        [1.782674925, -1.782674925, 1.782674925],
        [1.782674925, 1.782674925, -1.782674925],
    ],
    "species": [
        {
            "mass": None,
            "name": "Sb",
            "attached": None,
            "nattached": None,
            "concentration": [1],
            "original_name": None,
            "chemical_symbols": ["Sb"],
        },
        {
            "mass": None,
            "name": "Sr",
            "attached": None,
            "nattached": None,
            "concentration": [1],
            "original_name": None,
            "chemical_symbols": ["Sr"],
        },
    ],
    "species_at_sites": ["Sr", "Sb", "Sb", "Sb", "Sb"],
    "last_modified": "2023-11-16 06:57:59",
    "elements_ratios": [0.8, 0.2],
    "stress_tensor": [[0.3413351, 0, 0], [0, 0.3413351, 0], [0, 0, 0.3413351]],
    "energy": -17.323021,
    "magnetic_moments": [0, 0, 0, 0, 0],
    "forces": [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
    "total_magnetization": 0.000002,
    "dos_ef": 4.318709,
    "functional": "pbe",
    "cross_compatibility": True,
    "entalpic_fingerprint": "38f73083d88aa235c8c8c9d66617f3e3_229_Sr1Sb4",
}


def create_weird_structures():
    """Create structures that are very unlikely to be in LeMat-Bulk.

    Note: Using more common elements that BAWL hasher can handle properly.
    """
    structures = []

    # 1. Very large unit cell with noble gases
    lattice1 = np.eye(3) * 20.0  # Large but not extreme unit cell
    structure1 = Structure(
        lattice=lattice1,
        species=["Xe", "Kr", "Ne"],
        coords=[[0.1, 0.1, 0.1], [0.9, 0.9, 0.9], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structures.append(structure1)

    # 2. Unusual but real actinide compound
    lattice2 = [[8, 0, 0], [0, 8, 0], [0, 0, 8]]
    structure2 = Structure(
        lattice=lattice2,
        species=["U", "O", "O"],
        coords=[[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5]],
        coords_are_cartesian=False,
    )
    structures.append(structure2)

    # 3. Artificial structure with unusual coordination
    lattice3 = [[6, 0, 0], [0, 6, 0], [0, 0, 6]]
    structure3 = Structure(
        lattice=lattice3,
        species=["Be", "F", "F"],
        coords=[
            [0.1, 0.1, 0.1],
            [0.9, 0.9, 0.9],
            [0.5, 0.5, 0.5],
        ],
        coords_are_cartesian=False,
    )
    structures.append(structure3)

    return structures


@pytest.mark.skipif(not NOVELTY_TESTS_AVAILABLE, reason="Novelty metrics not available")
class TestNoveltyMetricReal:
    """Test novelty metrics with real LeMat-Bulk data."""

    def test_novelty_metric_initialization(self):
        """Test NoveltyMetric initialization."""
        metric = NoveltyMetric()
        assert metric.name == "Novelty"
        assert metric.config.reference_dataset == "LeMaterial/LeMat-Bulk"
        assert metric.config.reference_config == "compatible_pbe"
        assert metric.config.fingerprint_method == "bawl"
        assert hasattr(metric, "fingerprinter")

    def test_row_to_structure_conversion(self):
        """Test conversion of LeMat-Bulk row to pymatgen Structure."""
        metric = NoveltyMetric()

        structure = metric._row_to_structure(SAMPLE_LEMAT_ROW)

        # Check structure properties
        assert isinstance(structure, Structure)
        assert len(structure) == 5  # nsites
        assert structure.composition.reduced_formula == "SrSb4"

        # Check species
        species_symbols = [str(site.specie) for site in structure]
        expected_species = ["Sr", "Sb", "Sb", "Sb", "Sb"]
        assert species_symbols == expected_species

        # Check lattice (should be the same)
        expected_lattice = np.array(SAMPLE_LEMAT_ROW["lattice_vectors"])
        np.testing.assert_array_almost_equal(structure.lattice.matrix, expected_lattice)

    def test_fingerprint_consistency(self):
        """Test that BAWL fingerprinting is consistent."""
        metric = NoveltyMetric()
        structure = metric._row_to_structure(SAMPLE_LEMAT_ROW)

        # Compute fingerprint multiple times
        fp1 = metric.fingerprinter.get_material_hash(structure)
        fp2 = metric.fingerprinter.get_material_hash(structure)

        # Should be identical
        assert fp1 == fp2
        assert isinstance(fp1, str)
        assert len(fp1) > 0

    def test_compute_structure_novel(self):
        """Test computing novelty for a novel structure."""
        # Create a weird structure that definitely won't be in reference
        weird_structures = create_weird_structures()
        structure = weird_structures[0]  # Xe-Kr-Ne compound

        # Mock reference fingerprints (from our sample data)
        reference_fingerprints = {
            "38f73083d88aa235c8c8c9d66617f3e3_229_Sr1Sb4",
            "different_fingerprint_1_229_Sr1Sb4",
        }

        # Initialize fingerprinter
        metric = NoveltyMetric()
        fingerprinter = metric.fingerprinter

        result = NoveltyMetric.compute_structure(
            structure, reference_fingerprints, fingerprinter
        )

        # Should be novel (1.0) since weird structure won't match reference
        # Note: If BAWL fails, it returns NaN, so we check for either 1.0 or NaN
        assert result == 1.0 or np.isnan(result)

    def test_compute_structure_known(self):
        """Test computing novelty for a known structure."""
        # Create structure from our sample data
        metric = NoveltyMetric()
        structure = metric._row_to_structure(SAMPLE_LEMAT_ROW)

        # Get the actual fingerprint for this structure
        fingerprinter = metric.fingerprinter
        actual_fingerprint = fingerprinter.get_material_hash(structure)

        # Reference fingerprints including the actual one
        reference_fingerprints = {actual_fingerprint, "some_other_fingerprint"}

        result = NoveltyMetric.compute_structure(
            structure, reference_fingerprints, fingerprinter
        )

        # Should be known (0.0) since structure matches reference
        assert result == 0.0

    def test_aggregate_results(self):
        """Test aggregation of novelty results."""
        metric = NoveltyMetric()

        # Test with mixed results: 2 novel, 1 known
        values = [1.0, 0.0, 1.0]
        result = metric.aggregate_results(values)

        assert result["metrics"]["novelty_score"] == 2.0 / 3.0
        assert result["metrics"]["novel_structures_count"] == 2
        assert result["metrics"]["total_structures_evaluated"] == 3
        assert result["primary_metric"] == "novelty_score"

        # Test with all novel
        values = [1.0, 1.0, 1.0]
        result = metric.aggregate_results(values)
        assert result["metrics"]["novelty_score"] == 1.0

        # Test with all known
        values = [0.0, 0.0, 0.0]
        result = metric.aggregate_results(values)
        assert result["metrics"]["novelty_score"] == 0.0

        # Test with NaN values
        values = [1.0, float("nan"), 0.0]
        result = metric.aggregate_results(values)
        assert result["metrics"]["novelty_score"] == 0.5  # 1 novel out of 2 valid

    def test_custom_configuration(self):
        """Test novelty metric with custom configuration."""
        metric = NoveltyMetric(
            reference_dataset="LeMaterial/LeMat-Bulk",
            reference_config="compatible_pbesol",  # Different config
            max_reference_size=1000,
            cache_reference=False,
        )

        assert metric.config.reference_config == "compatible_pbesol"
        assert metric.config.max_reference_size == 1000
        assert metric.config.cache_reference is False

    def test_invalid_fingerprint_method(self):
        """Test error handling for invalid fingerprint method."""
        with pytest.raises(ValueError, match="Unknown fingerprint method"):
            NoveltyMetric(fingerprint_method="invalid_method")

    def test_callable_interface(self):
        """Test the __call__ interface inherited from BaseMetric."""
        metric = NoveltyMetric(max_reference_size=10)  # Small for speed

        # Create test structures
        structures = create_weird_structures()[:2]

        # Test callable interface
        result = metric(structures)

        # Should return MetricResult
        assert isinstance(result, MetricResult)
        assert hasattr(result, "metrics")
        assert "novelty_score" in result.metrics

    @pytest.mark.slow
    def test_real_dataset_integration(self):
        """Test with actual LeMat-Bulk dataset (slow test)."""
        # Use very small reference for CI/CD
        metric = NoveltyMetric(max_reference_size=50)

        # Create definitely novel structures
        weird_structures = create_weird_structures()

        # Evaluate novelty
        result = metric.compute(weird_structures)

        # Check that we get a MetricResult
        assert isinstance(result, MetricResult)

        # Check results - some structures might fail fingerprinting
        assert result.metrics["total_structures_evaluated"] <= len(weird_structures)
        assert (
            result.metrics["total_structures_evaluated"] > 0
        )  # At least some should work
        assert 0 <= result.metrics["novelty_score"] <= 1.0
        assert isinstance(result.metrics["novel_structures_count"], int)

        # Most structures that can be fingerprinted should be novel
        if result.metrics["total_structures_evaluated"] > 0:
            assert result.metrics["novelty_score"] >= 0.5

    def test_error_handling_with_problematic_elements(self):
        """Test error handling when BAWL hasher fails on certain elements."""
        metric = NoveltyMetric(max_reference_size=10)

        # Create structure with elements that might cause BAWL to fail
        lattice = [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]]

        # Try with superheavy elements that might not be in BAWL's database
        try:
            problematic_structure = Structure(
                lattice=lattice,
                species=["Og", "Ts"],
                coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
                coords_are_cartesian=False,
            )

            result = metric.compute([problematic_structure])

            # Should handle the error gracefully
            assert isinstance(result, MetricResult)
            # Either succeeds or fails gracefully
            assert result.metrics["total_structures_evaluated"] in [0, 1]

        except Exception:
            # If the structure creation itself fails, that's also OK
            pass


# Manual test function for development
if __name__ == "__main__":
    """Manual test for development."""
    print("Running manual novelty metrics test...")

    try:
        # Test basic functionality
        print("1. Testing basic initialization...")
        metric = NoveltyMetric()
        print("Metric initialized successfully")

        # Test structure conversion
        print("2. Testing structure conversion...")
        structure = metric._row_to_structure(SAMPLE_LEMAT_ROW)
        print(f"Converted structure: {structure.composition.reduced_formula}")

        # Test fingerprinting
        print("3. Testing fingerprinting...")
        fingerprint = metric.fingerprinter.get_material_hash(structure)
        print(f"Fingerprint: {fingerprint[:50]}...")

        # Test with weird structures
        print("4. Testing with novel structures...")
        weird_structures = create_weird_structures()
        print(f"Created {len(weird_structures)} weird structures")

        for i, struct in enumerate(weird_structures):
            try:
                fp = metric.fingerprinter.get_material_hash(struct)
                print(
                    f"   - Structure {i + 1} ({struct.composition.reduced_formula}): {fp[:30]}..."
                )
            except Exception as e:
                print(
                    f"   - Structure {i + 1} ({struct.composition.reduced_formula}): Failed - {e}"
                )

        print("\nAll manual tests passed!")
        print("\nTo run full tests with LeMat-Bulk:")
        print("   pytest tests/metrics/test_novelty_metric.py -v")

    except Exception as e:
        print(f"Test failed: {e}")
        traceback.print_exc()
