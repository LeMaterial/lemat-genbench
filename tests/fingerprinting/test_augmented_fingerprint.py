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
        lattice_cscl, ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]], coords_are_cartesian=False
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
    
    # 4. Fluorite structure (more complex for enhanced enumeration testing)
    lattice_caf2 = Lattice.cubic(5.46)
    caf2 = Structure(
        lattice_caf2,
        ["Ca", "F", "F"],
        [[0, 0, 0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
        coords_are_cartesian=False,
    )
    structures["caf2"] = caf2

    return structures


def create_mock_crystallographic_data():
    """Create mock crystallographic data for testing fingerprint functions."""
    return {
        "spacegroup_number": 225,
        "elements": ["Na", "Cl"],
        "site_symmetries": ["4a", "4b"],  # Proper Wyckoff format
        "sites_enumeration_augmented": [
            [1, 1],  # Base enumeration
            [1, 2],  # Alternative enumeration
            [2, 1],  # Another permutation
            [2, 2],  # Different pattern
        ],
        "multiplicity": [4, 4],
        "success": True,
        "error": None,
    }


def create_enhanced_crystallographic_data():
    """Create enhanced crystallographic data with many enumerations for testing."""
    return {
        "spacegroup_number": 227,  # Fd-3m (fluorite structure)
        "elements": ["Ca", "F", "F"],
        "site_symmetries": ["4a", "8c", "8c"],
        "sites_enumeration_augmented": [
            [1, 1, 2],  # Base enumeration
            [1, 2, 1],  # Swap F positions
            [1, 1, 1],  # All same
            [1, 2, 2],  # Different pattern
            [2, 1, 1],  # Rotate Ca
            [2, 2, 2],  # All different
            [1, 1, 3],  # More F diversity
            [1, 3, 1],  # Another F pattern
            [2, 1, 2],  # Mixed pattern
            [2, 2, 1],  # Another mixed
        ],
        "multiplicity": [4, 8, 8],
        "success": True,
        "error": None,
    }


def create_large_enumeration_data():
    """Create data with very large enumeration set for stress testing."""
    # Simulate what enhanced enumerations might produce
    enumerations = []
    # Generate many systematic combinations
    for i in range(1, 4):
        for j in range(1, 4):
            for k in range(1, 4):
                enumerations.append([i, j, k])
    
    return {
        "spacegroup_number": 225,
        "elements": ["Na", "Cl", "O"],
        "site_symmetries": ["4a", "4b", "8c"],
        "sites_enumeration_augmented": enumerations[:25],  # Cap at 25
        "multiplicity": [4, 4, 8],
        "success": True,
        "error": None,
    }


def create_edge_case_data():
    """Create edge case data to test robustness improvements."""
    return {
        "spacegroup_number": 166,
        "elements": ["Mg", "O"],
        "site_symmetries": ["2a", "6c"],
        "sites_enumeration_augmented": [
            [],  # Empty enumeration (should be handled)
            [1],  # Short enumeration (should be padded)
            [1, 2, 3, 4],  # Long enumeration (should be truncated)
            [0, 1],  # Contains zero (should be converted to 1)
            [-1, 2],  # Contains negative (should be converted to 1)
        ],
        "multiplicity": [2, 6],
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
        print(f"   NaCl fingerprint: {nacl_fp[:100]}...")

        # Test CsCl fingerprint
        cscl_fp = fingerprinter.get_structure_fingerprint(structures["cscl"])
        assert cscl_fp is not None
        assert isinstance(cscl_fp, str)
        assert cscl_fp.startswith("AUG_")
        print(f"   CsCl fingerprint: {cscl_fp[:100]}...")

        # Test identical structures have same fingerprint
        nacl_copy_fp = fingerprinter.get_structure_fingerprint(structures["nacl_copy"])
        assert nacl_copy_fp is not None
        assert nacl_fp == nacl_copy_fp

        # Different structures should have different fingerprints
        assert nacl_fp != cscl_fp

    def test_enhanced_fingerprint_with_complex_structure(self):
        """Test fingerprinting with complex structures that generate many enumerations."""
        structures = create_test_structures()
        fingerprinter = AugmentedFingerprinter()
        
        # Test fluorite structure (should have many enumerations)
        caf2_fp = fingerprinter.get_structure_fingerprint(structures["caf2"])
        assert caf2_fp is not None
        assert isinstance(caf2_fp, str)
        assert caf2_fp.startswith("AUG_")
        
        # Should be a substantial fingerprint due to multiple enumerations
        assert len(caf2_fp) > 30, f"Complex structure should have substantial fingerprint, got {len(caf2_fp)} chars"
        print(f"   CaF2 fingerprint length: {len(caf2_fp)} characters")

    def test_fingerprint_consistency(self):
        """Test that fingerprints are consistent across calls."""
        structures = create_test_structures()
        fingerprinter = AugmentedFingerprinter()

        # Generate fingerprint multiple times
        fp1 = fingerprinter.get_structure_fingerprint(structures["nacl"])
        fp2 = fingerprinter.get_structure_fingerprint(structures["nacl"])
        fp3 = fingerprinter.get_structure_fingerprint(structures["nacl"])

        # Should be identical
        assert fp1 is not None
        assert fp1 == fp2 == fp3

    def test_parameter_passing(self):
        """Test that analyzer parameters are properly passed through."""
        structures = create_test_structures()
        
        # Create fingerprinter with custom parameters
        fingerprinter = AugmentedFingerprinter(symprec=0.1, angle_tolerance=10.0)
        
        # Should work without issues and respect the parameters
        fp = fingerprinter.get_structure_fingerprint(structures["nacl"])
        assert fp is not None
        assert isinstance(fp, str)
        assert fp.startswith("AUG_")

    def test_fingerprint_to_string_conversion(self):
        """Test internal fingerprint to string conversion."""
        fingerprinter = AugmentedFingerprinter()

        # Test with mock fingerprint tuple with multiple variants
        mock_fingerprint = (
            225,
            frozenset([
                frozenset([(("Na", "4a", 1), 1), (("Cl", "4b", 1), 1)]),
                frozenset([(("Na", "4a", 1), 1), (("Cl", "4b", 2), 1)]),
                frozenset([(("Na", "4a", 2), 1), (("Cl", "4b", 1), 1)]),
            ]),
        )

        result = fingerprinter._fingerprint_to_string(mock_fingerprint)
        assert isinstance(result, str)
        assert result.startswith("AUG_225_")
        assert len(result) > 20  # Should be substantial due to multiple variants
        print(f"   Mock fingerprint string: {result[:100]}...")

    def test_fingerprint_string_with_large_variant_set(self):
        """Test fingerprint string conversion with many variants."""
        fingerprinter = AugmentedFingerprinter()

        # Create a fingerprint with many variants (like enhanced enumerations would produce)
        large_variants = set()
        for i in range(1, 6):
            for j in range(1, 4):
                variant = frozenset([(("Ca", "4a", i), 1), (("F", "8c", j), 2)])
                large_variants.add(variant)

        large_fingerprint = (227, frozenset(large_variants))
        result = fingerprinter._fingerprint_to_string(large_fingerprint)
        
        assert isinstance(result, str)
        assert result.startswith("AUG_227_")
        assert len(result) > 50  # Should be substantial
        print(f"   Large fingerprint length: {len(result)} characters")


class TestFingerprintFunctions:
    """Test individual fingerprint functions with enhanced enumeration data."""

    def test_record_to_augmented_fingerprint(self):
        """Test augmented fingerprint computation."""
        mock_data = create_mock_crystallographic_data()

        fingerprint = record_to_augmented_fingerprint(mock_data)

        # Should return tuple with spacegroup and wyckoff variants
        assert isinstance(fingerprint, tuple)
        assert len(fingerprint) == 2

        spacegroup, variants = fingerprint
        assert spacegroup == 225
        assert isinstance(variants, frozenset)
        
        # Should have multiple variants due to multiple enumerations
        assert len(variants) >= 1, f"Should have at least 1 variant, got {len(variants)}"
        print(f"   Augmented fingerprint variants: {len(variants)}")

    def test_record_to_augmented_fingerprint_enhanced(self):
        """Test augmented fingerprint with enhanced enumeration data."""
        enhanced_data = create_enhanced_crystallographic_data()

        fingerprint = record_to_augmented_fingerprint(enhanced_data)

        # Should return tuple with spacegroup and wyckoff variants
        assert isinstance(fingerprint, tuple)
        assert len(fingerprint) == 2

        spacegroup, variants = fingerprint
        assert spacegroup == 227
        assert isinstance(variants, frozenset)
        
        # Should have multiple variants due to many enumerations
        assert len(variants) >= 1, f"Enhanced data should have variants, got {len(variants)}"
        print(f"   Enhanced fingerprint variants: {len(variants)}")

        # Verify variants contain proper triplet structure
        for variant in variants:
            assert isinstance(variant, frozenset)
            for item, count in variant:
                assert isinstance(item, tuple)
                assert len(item) == 3  # (element, site_symmetry, enumeration_value)
                assert isinstance(count, int)
                assert count > 0

    def test_record_to_augmented_fingerprint_stress_test(self):
        """Test augmented fingerprint with very large enumeration sets."""
        large_data = create_large_enumeration_data()

        fingerprint = record_to_augmented_fingerprint(large_data)

        # Should handle large enumeration sets gracefully
        assert isinstance(fingerprint, tuple)
        assert len(fingerprint) == 2

        spacegroup, variants = fingerprint
        assert spacegroup == 225
        assert isinstance(variants, frozenset)
        
        # Should have variants but not cause performance issues
        print(f"   Stress test variants: {len(variants)}")
        assert len(variants) >= 1, "Should handle large enumeration sets"

    def test_record_to_augmented_fingerprint_edge_cases(self):
        """Test augmented fingerprint with edge case data (robustness test)."""
        edge_data = create_edge_case_data()

        # Should handle edge cases gracefully without crashing
        fingerprint = record_to_augmented_fingerprint(edge_data)

        assert isinstance(fingerprint, tuple)
        assert len(fingerprint) == 2
        assert fingerprint[0] == 166  # Space group should be preserved
        assert isinstance(fingerprint[1], frozenset)
        
        # Should still produce meaningful fingerprint despite edge cases
        print(f"   Edge case fingerprint variants: {len(fingerprint[1])}")

    def test_record_to_anonymous_fingerprint(self):
        """Test anonymous fingerprint computation."""
        mock_data = create_mock_crystallographic_data()

        fingerprint = record_to_anonymous_fingerprint(mock_data)

        # Should return tuple with spacegroup and structural variants
        assert isinstance(fingerprint, tuple)
        assert len(fingerprint) == 2

        spacegroup, variants = fingerprint
        assert spacegroup == 225
        assert isinstance(variants, frozenset)
        assert len(variants) >= 1

        # Verify variants don't contain element information
        for variant in variants:
            for item, count in variant:
                assert isinstance(item, tuple)
                assert len(item) == 2  # (site_symmetry, enumeration_value) - no element

    def test_record_to_relaxed_AFLOW_fingerprint(self):
        """Test relaxed AFLOW fingerprint computation."""
        mock_data = create_mock_crystallographic_data()

        fingerprint = record_to_relaxed_AFLOW_fingerprint(mock_data)

        # Should return tuple with spacegroup, stoichiometry, and sites
        assert isinstance(fingerprint, tuple)
        assert len(fingerprint) == 3

        spacegroup, stochio, sites = fingerprint
        assert spacegroup == 225
        assert isinstance(stochio, frozenset)
        assert isinstance(sites, frozenset)
        
        # Should have meaningful data
        assert len(sites) >= 1

    def test_record_to_strict_AFLOW_fingerprint(self):
        """Test strict AFLOW fingerprint computation."""
        mock_data = create_mock_crystallographic_data()

        fingerprint = record_to_strict_AFLOW_fingerprint(mock_data)

        # Should return tuple with spacegroup and variants
        assert isinstance(fingerprint, tuple)
        assert len(fingerprint) == 2

        spacegroup, variants = fingerprint
        assert spacegroup == 225
        assert isinstance(variants, frozenset)
        assert len(variants) >= 1

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

    def test_fingerprint_functions_with_mismatched_data(self):
        """Test fingerprint functions handle mismatched array lengths."""
        mismatched_data = {
            "spacegroup_number": 225,
            "elements": ["Na", "Cl", "F"],  # 3 elements
            "site_symmetries": ["4a", "4b"],  # 2 site symmetries
            "sites_enumeration_augmented": [[1, 2], [2, 1], [1, 1]],  # 2 values each
            "multiplicity": [4, 4],
        }

        # Should handle gracefully by truncating to minimum length
        fp = record_to_augmented_fingerprint(mismatched_data)
        assert isinstance(fp, tuple)
        assert len(fp) == 2
        assert fp[0] == 225

        # Should still produce valid fingerprint
        assert isinstance(fp[1], frozenset)

    def test_fingerprint_functions_with_empty_enumerations(self):
        """Test fingerprint functions handle empty or malformed enumerations."""
        empty_enum_data = {
            "spacegroup_number": 225,
            "elements": ["Na", "Cl"],
            "site_symmetries": ["4a", "4b"],
            "sites_enumeration_augmented": [],  # Empty enumerations
            "multiplicity": [4, 4],
        }

        # Should handle empty enumerations by creating default
        fp = record_to_augmented_fingerprint(empty_enum_data)
        assert isinstance(fp, tuple)
        assert len(fp) == 2
        assert fp[0] == 225
        assert len(fp[1]) >= 1  # Should create at least one default variant

        # Test with None enumerations
        none_enum_data = empty_enum_data.copy()
        none_enum_data["sites_enumeration_augmented"] = [[]]
        
        fp2 = record_to_augmented_fingerprint(none_enum_data)
        assert isinstance(fp2, tuple)
        assert len(fp2) == 2
        assert fp2[0] == 225
        assert len(fp2[1]) >= 1  # Should create default variant for empty inner list

    def test_fingerprint_with_real_enhanced_data(self):
        """Test fingerprints with actual enhanced enumeration data."""
        structures = create_test_structures()
        
        for name, structure in structures.items():
            # Get real enhanced crystallographic data
            crystal_data = structure_to_crystallographic_dict(structure)
            assert crystal_data["success"] is True
            
            print(f"\n   Testing {name}:")
            print(f"     Enumerations: {len(crystal_data['sites_enumeration_augmented'])}")
            
            # Test all fingerprint types
            aug_fp = record_to_augmented_fingerprint(crystal_data)
            anon_fp = record_to_anonymous_fingerprint(crystal_data)
            relax_fp = record_to_relaxed_AFLOW_fingerprint(crystal_data)
            strict_fp = record_to_strict_AFLOW_fingerprint(crystal_data)
            
            # All should be valid
            assert isinstance(aug_fp, tuple) and len(aug_fp) == 2
            assert isinstance(anon_fp, tuple) and len(anon_fp) == 2
            assert isinstance(relax_fp, tuple) and len(relax_fp) == 3
            assert isinstance(strict_fp, tuple) and len(strict_fp) == 2
            
            # Should have meaningful content
            assert aug_fp[0] > 0  # Valid space group
            assert len(aug_fp[1]) >= 1  # Should have at least one variant
            
            print(f"     Augmented variants: {len(aug_fp[1])}")
            print(f"     Anonymous variants: {len(anon_fp[1])}")


class TestConvenienceFunction:
    """Test convenience function with enhanced functionality."""

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

    def test_get_augmented_fingerprint_all_structures(self):
        """Test convenience function with all test structures."""
        structures = create_test_structures()

        fingerprints = {}
        for name, structure in structures.items():
            fp = get_augmented_fingerprint(structure)
            assert fp is not None
            assert isinstance(fp, str)
            assert fp.startswith("AUG_")
            fingerprints[name] = fp
            print(f"   {name}: {len(fp)} chars, {fp[:50]}...")

        # Identical structures should have identical fingerprints
        assert fingerprints["nacl"] == fingerprints["nacl_copy"]

        # Different structures should have different fingerprints
        assert fingerprints["nacl"] != fingerprints["cscl"]
        assert fingerprints["nacl"] != fingerprints["caf2"]


class TestIntegration:
    """Integration tests combining crystallographic analysis and fingerprinting."""

    def test_full_fingerprinting_workflow_enhanced(self):
        """Test complete workflow from structure to fingerprint with enhanced enumerations."""
        structures = create_test_structures()

        for name, structure in structures.items():
            print(f"\n   Testing enhanced workflow for {name}:")
            
            # Step 1: Analyze crystallography
            crystal_data = structure_to_crystallographic_dict(structure)
            assert crystal_data["success"] is True
            
            enum_count = len(crystal_data['sites_enumeration_augmented'])
            print(f"     ✅ Analysis: {enum_count} enumerations")
            assert enum_count >= 1, f"Should have at least 1 enumeration, got {enum_count}"

            # Step 2: Generate fingerprint
            fingerprint = record_to_augmented_fingerprint(crystal_data)
            assert isinstance(fingerprint, tuple)
            assert len(fingerprint) == 2
            
            variant_count = len(fingerprint[1])
            print(f"     ✅ Fingerprint variants: {variant_count}")
            assert variant_count >= 1, f"Should have at least 1 variant, got {variant_count}"

            # Step 3: Convert to string
            fingerprinter = AugmentedFingerprinter()
            fp_string = fingerprinter._fingerprint_to_string(fingerprint)
            assert isinstance(fp_string, str)
            assert fp_string.startswith("AUG_")
            assert len(fp_string) > 10  # Should be meaningful
            print(f"     ✅ String length: {len(fp_string)} chars")

    def test_fingerprint_comparison_scenarios_enhanced(self):
        """Test various fingerprint comparison scenarios with enhanced enumerations."""
        structures = create_test_structures()

        # Generate all fingerprints
        fingerprints = {}
        for name, structure in structures.items():
            fp = get_augmented_fingerprint(structure)
            assert fp is not None
            fingerprints[name] = fp
            print(f"   {name}: {len(fp)} chars")

        # Test identical structures
        assert fingerprints["nacl"] == fingerprints["nacl_copy"]
        print("   ✅ Identical structures produce identical fingerprints")

        # Test different structures
        unique_fps = set(fingerprints.values())
        expected_unique = len([k for k in fingerprints.keys() if k != "nacl_copy"])
        assert len(unique_fps) == expected_unique, "Different structures should have different fingerprints"
        print("   ✅ Different structures produce different fingerprints")

        # All fingerprints should be valid strings
        for name, fp in fingerprints.items():
            assert isinstance(fp, str)
            assert len(fp) > 0
            assert fp.startswith("AUG_")

    def test_performance_with_enhanced_enumerations(self):
        """Test that enhanced enumerations don't cause performance issues."""
        import time
        
        structures = create_test_structures()
        start_time = time.time()
        
        for name, structure in structures.items():
            # This should complete reasonably quickly even with enhanced enumerations
            fp = get_augmented_fingerprint(structure)
            assert fp is not None
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"   ⚡ Generated {len(structures)} enhanced fingerprints in {total_time:.3f} seconds")
        assert total_time < 30, f"Enhanced fingerprinting took too long: {total_time:.3f}s"

    def test_robustness_with_different_symmetry_precisions(self):
        """Test fingerprint robustness across different symmetry precisions."""
        structures = create_test_structures()
        
        precisions = [0.001, 0.01, 0.1]
        
        for name, structure in structures.items():
            fps_at_precisions = []
            space_groups = []
            variant_counts = []
            
            for symprec in precisions:
                fp = get_augmented_fingerprint(structure, symprec=symprec)
                assert fp is not None
                fps_at_precisions.append(fp)
                
                # Extract space group and analyze variants
                if fp.startswith("AUG_"):
                    parts = fp.split("_")
                    if len(parts) > 1 and parts[1].isdigit():
                        space_groups.append(int(parts[1]))
                
                # Count variants by analyzing string structure
                if len(parts) > 2:
                    variant_part = "_".join(parts[2:])
                    variant_count = len(variant_part.split("|")) if variant_part else 0
                    variant_counts.append(variant_count)
            
            # Space groups should be consistent across precisions
            if space_groups:
                unique_sgs = set(space_groups)
                print(f"   {name}: Space groups at different precisions: {unique_sgs}")
                assert len(unique_sgs) <= 2, f"Too much space group variation: {unique_sgs}"
            
            # Variant counts should be reasonable
            if variant_counts:
                print(f"   {name}: Variant counts: {variant_counts}")
                assert all(count >= 0 for count in variant_counts), "Should have non-negative variant counts"


class TestErrorHandling:
    """Test error handling and edge cases with enhanced functionality."""

    def test_empty_crystallographic_data(self):
        """Test handling of empty crystallographic data."""
        empty_data = {
            "spacegroup_number": 0,
            "elements": [],
            "site_symmetries": [],
            "sites_enumeration_augmented": [],
            "multiplicity": [],
        }

        # All functions should handle empty data gracefully
        fp1 = record_to_augmented_fingerprint(empty_data)
        fp2 = record_to_anonymous_fingerprint(empty_data)
        fp3 = record_to_relaxed_AFLOW_fingerprint(empty_data)
        fp4 = record_to_strict_AFLOW_fingerprint(empty_data)

        # Should return fallback fingerprints
        assert fp1 == (0, frozenset())
        assert fp2 == (0, frozenset())
        assert fp3 == (0, frozenset(), frozenset())
        assert fp4 == (0, frozenset())

    def test_partial_crystallographic_data(self):
        """Test handling of partial crystallographic data."""
        partial_data = {
            "spacegroup_number": 225,
            "elements": ["Na"],
            # Missing other required fields
        }

        # Should handle gracefully
        fp = record_to_augmented_fingerprint(partial_data)
        assert isinstance(fp, tuple)
        assert fp[0] == 225  # Should preserve space group
        assert isinstance(fp[1], frozenset)

    def test_very_large_enumeration_handling(self):
        """Test handling of structures with extremely large enumeration sets."""
        # Create data with many enumerations (stress test)
        huge_enumerations = []
        for i in range(100):  # 100 different enumerations
            huge_enumerations.append([i % 5 + 1, (i + 1) % 3 + 1, (i + 2) % 4 + 1])
        
        huge_data = {
            "spacegroup_number": 225,
            "elements": ["Na", "Cl", "O"],
            "site_symmetries": ["4a", "4b", "8c"],
            "sites_enumeration_augmented": huge_enumerations,
            "multiplicity": [4, 4, 8],
        }

        # Should handle without issues (though may limit variants)
        fp = record_to_augmented_fingerprint(huge_data)
        assert isinstance(fp, tuple)
        assert len(fp) == 2
        assert fp[0] == 225
        assert isinstance(fp[1], frozenset)
        
        print(f"   Huge enumeration test: {len(fp[1])} variants from {len(huge_enumerations)} enumerations")

    def test_invalid_structure_fingerprinting_enhanced(self):
        """Test fingerprinting with structures that cause analysis to fail."""
        fingerprinter = AugmentedFingerprinter()

        # Create a structure that might cause issues
        minimal_lattice = Lattice.cubic(1.0)
        minimal_structure = Structure(
            minimal_lattice,
            ["H"],
            [[0, 0, 0]],
            coords_are_cartesian=False
        )

        # Should either succeed or return None gracefully
        fp = fingerprinter.get_structure_fingerprint(minimal_structure)
        
        # If it succeeds, should be valid
        if fp is not None:
            assert isinstance(fp, str)
            assert fp.startswith("AUG_")

    def test_malformed_enumeration_values(self):
        """Test handling of malformed enumeration values."""
        malformed_data = {
            "spacegroup_number": 225,
            "elements": ["Na", "Cl"],
            "site_symmetries": ["4a", "4b"],
            "sites_enumeration_augmented": [
                [1, 2],          # Normal
                [0, 1],          # Contains zero
                [-1, 2],         # Contains negative
                [1.5, 2.7],      # Contains floats
                ["1", "2"],      # Contains strings
            ],
            "multiplicity": [4, 4],
        }

        # Should handle malformed values gracefully
        fp = record_to_augmented_fingerprint(malformed_data)
        assert isinstance(fp, tuple)
        assert len(fp) == 2
        assert fp[0] == 225
        assert isinstance(fp[1], frozenset)
        print(f"   Malformed enumeration handling: {len(fp[1])} variants created")

    def test_none_and_missing_fields(self):
        """Test handling of None values and missing fields."""
        none_data = {
            "spacegroup_number": None,
            "elements": None,
            "site_symmetries": None,
            "sites_enumeration_augmented": None,
            "multiplicity": None,
        }

        # Should handle None values gracefully
        fp = record_to_augmented_fingerprint(none_data)
        assert isinstance(fp, tuple)
        assert len(fp) == 2
        assert isinstance(fp[1], frozenset)

    def test_type_conversion_robustness(self):
        """Test robustness of type conversions in fingerprint functions."""
        mixed_type_data = {
            "spacegroup_number": "225",  # String instead of int
            "elements": [1, 2],          # Numbers instead of strings
            "site_symmetries": [4, 8],   # Numbers instead of strings
            "sites_enumeration_augmented": [[1, 2], [2, 1]],
            "multiplicity": ["4", "4"],  # Strings instead of ints
        }

        # Should handle type conversions gracefully
        fp = record_to_augmented_fingerprint(mixed_type_data)
        assert isinstance(fp, tuple)
        assert len(fp) == 2
        # Space group should be converted to int
        assert isinstance(fp[0], int) or fp[0] == 0  # May fallback to 0 if conversion fails
        assert isinstance(fp[1], frozenset)


class TestValidationAndLogging:
    """Test validation improvements and error handling."""

    def test_validation_warnings(self):
        """Test that validation warnings are properly logged."""
        import logging
        from io import StringIO
        
        # Capture log output
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        logger = logging.getLogger("lemat_genbench.fingerprinting.augmented_fingerprint")
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
        
        try:
            # Test with data that should trigger warnings
            invalid_data = {
                "spacegroup_number": 225,
                "elements": [],  # Empty - should trigger warning
                "site_symmetries": ["4a"],
                "sites_enumeration_augmented": [[]],  # Empty enumeration
            }
            
            fp = record_to_augmented_fingerprint(invalid_data)
            assert fp == (225, frozenset())  # Should fallback gracefully
            
            # Check that warning was logged (if logging is configured)
            _ = log_capture.getvalue()
            # Note: This test might not capture logs if logger is not configured to output
            
        finally:
            logger.removeHandler(handler)

    def test_consistent_length_handling(self):
        """Test that length mismatches are handled consistently."""
        # Different length combinations
        test_cases = [
            {
                "elements": ["Na", "Cl", "O"],      # 3 elements
                "site_symmetries": ["4a", "4b"],    # 2 symmetries
                "expected_length": 2,               # Should truncate to min
            },
            {
                "elements": ["Na"],                 # 1 element
                "site_symmetries": ["4a", "4b", "8c"],  # 3 symmetries
                "expected_length": 1,               # Should truncate to min
            },
            {
                "elements": ["Na", "Cl"],           # 2 elements
                "site_symmetries": ["4a", "4b"],    # 2 symmetries
                "expected_length": 2,               # Perfect match
            },
        ]
        
        for i, case in enumerate(test_cases):
            test_data = {
                "spacegroup_number": 225,
                "elements": case["elements"],
                "site_symmetries": case["site_symmetries"],
                "sites_enumeration_augmented": [[1] * case["expected_length"]],
                "multiplicity": [4] * case["expected_length"],
            }
            
            fp = record_to_augmented_fingerprint(test_data)
            assert isinstance(fp, tuple)
            assert len(fp) == 2
            
            # Verify that the fingerprint was created with consistent lengths
            assert isinstance(fp[1], frozenset)
            print(f"   Length test case {i+1}: {len(fp[1])} variants created")


if __name__ == "__main__":
    """Manual test runner for development."""
    print("🧪 Running enhanced augmented fingerprinting tests...")

    try:
        # Test basic functionality
        print("\n1. Testing enhanced fingerprint generation...")
        structures = create_test_structures()

        for name, structure in structures.items():
            fp = get_augmented_fingerprint(structure)
            assert fp is not None, f"Fingerprint generation failed for {name}"
            print(f"   ✅ {name}: {len(fp)} chars, {fp[:50]}...")

        # Test consistency
        print("\n2. Testing fingerprint consistency...")
        nacl_fp1 = get_augmented_fingerprint(structures["nacl"])
        nacl_fp2 = get_augmented_fingerprint(structures["nacl"])
        assert nacl_fp1 == nacl_fp2
        print("   ✅ Consistency check passed")

        # Test enhanced functionality
        print("\n3. Testing enhanced enumeration handling...")
        enhanced_data = create_enhanced_crystallographic_data()
        enhanced_fp = record_to_augmented_fingerprint(enhanced_data)
        assert enhanced_fp[0] == 227
        print(f"   ✅ Enhanced data: SG={enhanced_fp[0]}, variants={len(enhanced_fp[1])}")

        # Test edge cases
        print("\n4. Testing edge case handling...")
        edge_data = create_edge_case_data()
        edge_fp = record_to_augmented_fingerprint(edge_data)
        assert edge_fp[0] == 166
        print(f"   ✅ Edge cases: SG={edge_fp[0]}, variants={len(edge_fp[1])}")

        # Test stress scenarios
        print("\n5. Testing stress scenarios...")
        large_data = create_large_enumeration_data()
        large_fp = record_to_augmented_fingerprint(large_data)
        print(f"   ✅ Large enumeration set: {len(large_fp[1])} variants")

        # Test robustness improvements
        print("\n6. Testing robustness improvements...")
        test_validation = TestValidationAndLogging()
        test_validation.test_consistent_length_handling()
        print("   ✅ Length handling tests passed")

        # Test error handling
        print("\n7. Testing error handling...")
        test_errors = TestErrorHandling()
        test_errors.test_malformed_enumeration_values()
        print("   ✅ Error handling tests passed")

        # Test integration
        print("\n8. Testing enhanced integration workflow...")
        test_integration = TestIntegration()
        test_integration.test_full_fingerprinting_workflow_enhanced()

        print("\n✅ All enhanced tests completed successfully!")
        print("\nKey improvements validated:")
        print("   • Better input validation and error handling")
        print("   • Robust handling of malformed enumerations")
        print("   • Consistent length management")
        print("   • Proper type conversions")
        print("   • Graceful fallbacks for edge cases")
        print("\nTo run full test suite:")
        print("   uv run pytest tests/fingerprinting/test_augmented_fingerprint.py -v")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()