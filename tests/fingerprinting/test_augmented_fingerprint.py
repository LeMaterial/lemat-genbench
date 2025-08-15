"""Tests for augmented fingerprinting functionality."""

from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

from lemat_genbench.fingerprinting.augmented_fingerprint import (
    AugmentedFingerprinter,
    get_augmented_fingerprint,
    record_to_anonymous_fingerprint,
    record_to_augmented_fingerprint,
    record_to_relaxed_AFLOW_fingerprint,
    record_to_strict_AFLOW_fingerprint,
)
from lemat_genbench.fingerprinting.crystallographic_analyzer import (
    structure_to_crystallographic_dict,
)


def create_test_structures():
    """Create test structures for fingerprinting."""
    structures = {}

    # 1. Simple NaCl structure
    lattice_nacl = Lattice.cubic(5.64)
    nacl = Structure(
        lattice_nacl,
        ["Na", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structures["nacl"] = nacl

    # 2. CsCl structure
    lattice_cscl = Lattice.cubic(4.11)
    cscl = Structure(
        lattice_cscl, ["Cs", "Cl"], [[0, 0, 0], [0.5, 0, 0]], coords_are_cartesian=False
    )
    structures["cscl"] = cscl

    # 3. Identical copy of NaCl (should have same fingerprint)
    nacl_copy = Structure(
        lattice_nacl,
        ["Na", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structures["nacl_copy"] = nacl_copy

    return structures


def create_mock_crystallographic_data():
    """Create mock crystallographic data for testing fingerprint functions."""
    return {
        "spacegroup_number": 221,
        "elements": ["Na", "Cl"],
        "site_symmetries": ["1a", "1b"],
        "sites_enumeration_augmented": [[1, 2], [2, 1]],
        "multiplicity": [1, 1],
        "success": True,
        "error": None,
    }


class TestAugmentedFingerprinter:
    """Test suite for AugmentedFingerprinter class."""

    def test_initialization(self):
        """Test fingerprinter initialization."""
        fingerprinter = AugmentedFingerprinter()
        assert fingerprinter.analyzer is not None
        assert fingerprinter.analyzer.symprec == 0.01
        assert fingerprinter.analyzer.angle_tolerance == 5.0

        # Test custom parameters
        fingerprinter_custom = AugmentedFingerprinter(symprec=0.1, angle_tolerance=10.0)
        assert fingerprinter_custom.analyzer.symprec == 0.1
        assert fingerprinter_custom.analyzer.angle_tolerance == 10.0

    def test_get_structure_fingerprint(self):
        """Test fingerprint generation for structures."""
        structures = create_test_structures()
        fingerprinter = AugmentedFingerprinter()

        # Test NaCl fingerprint
        nacl_fp = fingerprinter.get_structure_fingerprint(structures["nacl"])
        assert nacl_fp is not None
        assert isinstance(nacl_fp, str)
        assert nacl_fp.startswith("AUG_")

        # Test CsCl fingerprint
        cscl_fp = fingerprinter.get_structure_fingerprint(structures["cscl"])
        assert cscl_fp is not None
        assert isinstance(cscl_fp, str)
        assert cscl_fp.startswith("AUG_")

        # Test identical structures have same fingerprint
        nacl_copy_fp = fingerprinter.get_structure_fingerprint(structures["nacl_copy"])
        assert nacl_fp == nacl_copy_fp

        # Different structures should have different fingerprints
        assert nacl_fp != cscl_fp

    def test_fingerprint_consistency(self):
        """Test that fingerprints are consistent across calls."""
        structures = create_test_structures()
        fingerprinter = AugmentedFingerprinter()

        # Generate fingerprint multiple times
        fp1 = fingerprinter.get_structure_fingerprint(structures["nacl"])
        fp2 = fingerprinter.get_structure_fingerprint(structures["nacl"])
        fp3 = fingerprinter.get_structure_fingerprint(structures["nacl"])

        # Should be identical
        assert fp1 == fp2 == fp3

    def test_fingerprint_to_string_conversion(self):
        """Test internal fingerprint to string conversion."""
        fingerprinter = AugmentedFingerprinter()

        # Test with mock fingerprint tuple
        mock_fingerprint = (
            221,
            frozenset([frozenset([(("Na", "1a", 1), 1), (("Cl", "1b", 2), 1)])]),
        )

        result = fingerprinter._fingerprint_to_string(mock_fingerprint)
        assert isinstance(result, str)
        assert result.startswith("AUG_221_")


class TestFingerprintFunctions:
    """Test individual fingerprint functions."""

    def test_record_to_augmented_fingerprint(self):
        """Test augmented fingerprint computation."""
        mock_data = create_mock_crystallographic_data()

        fingerprint = record_to_augmented_fingerprint(mock_data)

        # Should return tuple with spacegroup and wyckoff variants
        assert isinstance(fingerprint, tuple)
        assert len(fingerprint) == 2

        spacegroup, variants = fingerprint
        assert spacegroup == 221
        assert isinstance(variants, frozenset)
        assert len(variants) > 0

    def test_record_to_anonymous_fingerprint(self):
        """Test anonymous fingerprint computation."""
        mock_data = create_mock_crystallographic_data()

        fingerprint = record_to_anonymous_fingerprint(mock_data)

        # Should return tuple with spacegroup and structural variants
        assert isinstance(fingerprint, tuple)
        assert len(fingerprint) == 2

        spacegroup, variants = fingerprint
        assert spacegroup == 221
        assert isinstance(variants, frozenset)

    def test_record_to_relaxed_AFLOW_fingerprint(self):
        """Test relaxed AFLOW fingerprint computation."""
        mock_data = create_mock_crystallographic_data()

        fingerprint = record_to_relaxed_AFLOW_fingerprint(mock_data)

        # Should return tuple with spacegroup, stoichiometry, and sites
        assert isinstance(fingerprint, tuple)
        assert len(fingerprint) == 3

        spacegroup, stochio, sites = fingerprint
        assert spacegroup == 221
        assert isinstance(stochio, frozenset)
        assert isinstance(sites, frozenset)

    def test_record_to_strict_AFLOW_fingerprint(self):
        """Test strict AFLOW fingerprint computation."""
        mock_data = create_mock_crystallographic_data()

        fingerprint = record_to_strict_AFLOW_fingerprint(mock_data)

        # Should return tuple with spacegroup and variants
        assert isinstance(fingerprint, tuple)
        assert len(fingerprint) == 2

        spacegroup, variants = fingerprint
        assert spacegroup == 221
        assert isinstance(variants, frozenset)

    def test_fingerprint_functions_with_invalid_data(self):
        """Test fingerprint functions handle invalid data gracefully."""
        invalid_data = {}

        # All functions should handle missing data gracefully
        fp1 = record_to_augmented_fingerprint(invalid_data)
        fp2 = record_to_anonymous_fingerprint(invalid_data)
        fp3 = record_to_relaxed_AFLOW_fingerprint(invalid_data)
        fp4 = record_to_strict_AFLOW_fingerprint(invalid_data)

        # Should return fallback fingerprints
        assert fp1 == (0, frozenset())
        assert fp2 == (0, frozenset())
        assert fp3 == (0, frozenset(), frozenset())
        assert fp4 == (0, frozenset())


class TestConvenienceFunction:
    """Test convenience function."""

    def test_get_augmented_fingerprint(self):
        """Test the convenience function."""
        structures = create_test_structures()

        # Test with NaCl
        nacl_fp = get_augmented_fingerprint(structures["nacl"])
        assert nacl_fp is not None
        assert isinstance(nacl_fp, str)
        assert nacl_fp.startswith("AUG_")

        # Test consistency
        nacl_fp2 = get_augmented_fingerprint(structures["nacl"])
        assert nacl_fp == nacl_fp2

        # Test with custom parameters
        nacl_fp_custom = get_augmented_fingerprint(
            structures["nacl"], symprec=0.1, angle_tolerance=10.0
        )
        assert nacl_fp_custom is not None
        assert isinstance(nacl_fp_custom, str)


class TestIntegration:
    """Integration tests combining crystallographic analysis and fingerprinting."""

    def test_full_fingerprinting_workflow(self):
        """Test complete workflow from structure to fingerprint."""
        structures = create_test_structures()

        for name, structure in structures.items():
            # Step 1: Analyze crystallography
            crystal_data = structure_to_crystallographic_dict(structure)
            assert crystal_data["success"] is True

            # Step 2: Generate fingerprint
            fingerprint = record_to_augmented_fingerprint(crystal_data)
            assert isinstance(fingerprint, tuple)
            assert len(fingerprint) == 2

            # Step 3: Convert to string via fingerprinter
            fingerprinter = AugmentedFingerprinter()
            fp_string = fingerprinter._fingerprint_to_string(fingerprint)
            assert isinstance(fp_string, str)
            assert fp_string.startswith("AUG_")

    def test_fingerprint_comparison_scenarios(self):
        """Test various fingerprint comparison scenarios."""
        structures = create_test_structures()

        # Generate all fingerprints
        fingerprints = {}
        for name, structure in structures.items():
            fp = get_augmented_fingerprint(structure)
            fingerprints[name] = fp

        # Test identical structures
        assert fingerprints["nacl"] == fingerprints["nacl_copy"]

        # Test different structures
        assert fingerprints["nacl"] != fingerprints["cscl"]

        # All fingerprints should be valid strings
        for name, fp in fingerprints.items():
            assert fp is not None
            assert isinstance(fp, str)
            assert len(fp) > 0


if __name__ == "__main__":
    """Manual test runner for development."""
    print("🧪 Running augmented fingerprinting tests...")

    try:
        # Test basic functionality
        print("1. Testing fingerprint generation...")
        structures = create_test_structures()

        for name, structure in structures.items():
            fp = get_augmented_fingerprint(structure)
            print(f"   {name}: {fp}")

        # Test consistency
        print("2. Testing fingerprint consistency...")
        nacl_fp1 = get_augmented_fingerprint(structures["nacl"])
        nacl_fp2 = get_augmented_fingerprint(structures["nacl"])
        print(f"   Consistency check: {nacl_fp1 == nacl_fp2}")

        print("✅ All manual tests passed!")
        print("\nTo run full test suite:")
        print("   uv run pytest tests/fingerprinting/test_augmented_fingerprint.py -v")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
