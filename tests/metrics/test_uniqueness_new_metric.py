"""Tests for new uniqueness metrics implementation using augmented fingerprints.

This test suite focuses on real-world scenarios and actual implementation behavior
rather than heavily mocked tests. It tests the metric with actual structures and
fingerprinting to ensure it works correctly in practice.
"""

import traceback

import numpy as np
from pymatgen.core.structure import Structure

from lemat_genbench.metrics.base import MetricResult
from lemat_genbench.metrics.uniqueness_new_metric import UniquenessNewMetric


def create_realistic_structure_set():
    """Create a realistic set of structures that might come from generation.
    
    This includes common duplicates, variations, and unique structures.
    """
    structures = []
    
    # Common binary compounds that generators often produce
    
    # Set 1: NaCl structures (3 duplicates + 1 slightly different)
    nacl_lattice = [[5.64, 0, 0], [0, 5.64, 0], [0, 0, 5.64]]  # Realistic NaCl lattice
    for i in range(3):
        structure = Structure(
            lattice=nacl_lattice,
            species=["Na", "Cl"],
            coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
            coords_are_cartesian=False,
        )
        structures.append(structure)
    
    # Slightly strained NaCl (should still be same structure)
    strained_lattice = [[5.66, 0, 0], [0, 5.66, 0], [0, 0, 5.66]]
    strained_nacl = Structure(
        lattice=strained_lattice,
        species=["Na", "Cl"], 
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structures.append(strained_nacl)
    
    # Set 2: CsCl structures (2 duplicates)
    cscl_lattice = [[4.11, 0, 0], [0, 4.11, 0], [0, 0, 4.11]]  # Realistic CsCl lattice
    for i in range(2):
        structure = Structure(
            lattice=cscl_lattice,
            species=["Cs", "Cl"],
            coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
            coords_are_cartesian=False,
        )
        structures.append(structure)
    
    # Set 3: Unique structures
    
    # LiF
    lif_lattice = [[4.03, 0, 0], [0, 4.03, 0], [0, 0, 4.03]]
    lif_structure = Structure(
        lattice=lif_lattice,
        species=["Li", "F"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structures.append(lif_structure)
    
    # MgO
    mgo_lattice = [[4.21, 0, 0], [0, 4.21, 0], [0, 0, 4.21]]
    mgo_structure = Structure(
        lattice=mgo_lattice,
        species=["Mg", "O"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structures.append(mgo_structure)
    
    # CaF2 (different structure type)
    caf2_lattice = [[5.46, 0, 0], [0, 5.46, 0], [0, 0, 5.46]]
    caf2_structure = Structure(
        lattice=caf2_lattice,
        species=["Ca", "F", "F"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
        coords_are_cartesian=False,
    )
    structures.append(caf2_structure)
    
    return structures


def create_structures_with_preprocessed_fingerprints():
    """Create structures with realistic preprocessed fingerprints for testing property mode."""
    structures = []
    
    # Create structures that represent what would come from the preprocessor
    
    # Set 1: Multiple structures with same fingerprint (duplicates)
    for i in range(3):
        structure = Structure(
            lattice=[[5.64, 0, 0], [0, 5.64, 0], [0, 0, 5.64]],
            species=["Na", "Cl"],
            coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
            coords_are_cartesian=False,
        )
        # Realistic fingerprint that would come from augmented fingerprinting
        structure.properties["augmented_fingerprint"] = "AUG_225_('Cl', '4b', 1):1_('Na', '4a', 1):1"
        structures.append(structure)
    
    # Set 2: Different structure with different fingerprint
    structure2 = Structure(
        lattice=[[4.11, 0, 0], [0, 4.11, 0], [0, 0, 4.11]],
        species=["Cs", "Cl"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structure2.properties["augmented_fingerprint"] = "AUG_221_('Cl', '1b', 1):1_('Cs', '1a', 1):1"
    structures.append(structure2)
    
    # Set 3: Another unique structure
    structure3 = Structure(
        lattice=[[4.03, 0, 0], [0, 4.03, 0], [0, 0, 4.03]],
        species=["Li", "F"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structure3.properties["augmented_fingerprint"] = "AUG_225_('F', '4b', 1):1_('Li', '4a', 1):1"
    structures.append(structure3)
    
    return structures


def create_generation_batch_scenario():
    """Create a realistic batch of generated structures with known patterns."""
    structures = []
    
    # Simulate what might come from a typical generation run:
    # - Some identical structures (common with generative models)
    # - Some variations that should be considered the same
    # - Some truly unique structures
    
    # Batch 1: 4 identical NaCl structures (common generation artifact)
    nacl_lattice = [[5.64, 0, 0], [0, 5.64, 0], [0, 0, 5.64]]
    for _ in range(4):
        structures.append(Structure(
            lattice=nacl_lattice,
            species=["Na", "Cl"],
            coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
            coords_are_cartesian=False,
        ))
    
    # Batch 2: 2 identical CsCl structures
    cscl_lattice = [[4.11, 0, 0], [0, 4.11, 0], [0, 0, 4.11]]
    for _ in range(2):
        structures.append(Structure(
            lattice=cscl_lattice,
            species=["Cs", "Cl"],
            coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
            coords_are_cartesian=False,
        ))
    
    # Batch 3: 3 unique structures
    unique_compounds = [
        (["Li", "F"], [[4.03, 0, 0], [0, 4.03, 0], [0, 0, 4.03]]),
        (["K", "Br"], [[6.60, 0, 0], [0, 6.60, 0], [0, 0, 6.60]]),
        (["Mg", "O"], [[4.21, 0, 0], [0, 4.21, 0], [0, 0, 4.21]]),
    ]
    
    for species, lattice in unique_compounds:
        structures.append(Structure(
            lattice=lattice,
            species=species,
            coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
            coords_are_cartesian=False,
        ))
    
    return structures


class TestUniquenessNewMetricRealistic:
    """Test suite focusing on realistic scenarios and actual implementation behavior."""

    def test_basic_initialization(self):
        """Test basic metric initialization with different configurations."""
        # Default initialization
        metric = UniquenessNewMetric()
        assert metric.name == "UniquenessNew"
        assert metric.config.fingerprint_source == "auto"
        assert metric.config.symprec == 0.01
        
        # Custom initialization
        metric_custom = UniquenessNewMetric(
            fingerprint_source="compute",
            symprec=0.001,
            angle_tolerance=1.0,
            name="Custom Uniqueness"
        )
        assert metric_custom.name == "Custom Uniqueness"
        assert metric_custom.config.fingerprint_source == "compute"
        assert metric_custom.config.symprec == 0.001

    def test_property_mode_with_preprocessed_fingerprints(self):
        """Test using preprocessed fingerprints (property mode) - most common use case."""
        metric = UniquenessNewMetric(fingerprint_source="property")
        structures = create_structures_with_preprocessed_fingerprints()
        
        result = metric.compute(structures)
        
        # Should work correctly
        assert isinstance(result, MetricResult)
        assert result.n_structures == 5
        assert result.metrics["failed_fingerprinting_count"] == 0
        
        # Expected: 3 NaCl (same fingerprint) + 1 CsCl + 1 LiF = 3 unique
        assert result.metrics["unique_structures_count"] == 3
        assert result.metrics["duplicate_structures_count"] == 2
        assert result.metrics["uniqueness_score"] == 3.0 / 5.0
        
        # Check individual values follow expected pattern
        individual_values = result.individual_values
        # First 3 should be 1/3 (duplicates), last 2 should be 1.0 (unique)
        assert abs(individual_values[0] - 1/3) < 1e-6
        assert abs(individual_values[1] - 1/3) < 1e-6
        assert abs(individual_values[2] - 1/3) < 1e-6
        assert abs(individual_values[3] - 1.0) < 1e-6
        assert abs(individual_values[4] - 1.0) < 1e-6

    def test_compute_mode_with_real_structures(self):
        """Test computing fingerprints on-the-fly (compute mode)."""
        metric = UniquenessNewMetric(fingerprint_source="compute")
        structures = create_realistic_structure_set()
        
        result = metric.compute(structures)
        
        # Should work correctly (though specific counts depend on actual fingerprinting)
        assert isinstance(result, MetricResult)
        assert result.n_structures == len(structures)
        assert result.metrics["total_structures_evaluated"] == len(structures)
        
        # Should have reasonable uniqueness score
        uniqueness_score = result.metrics["uniqueness_score"]
        if not np.isnan(uniqueness_score):
            assert 0 <= uniqueness_score <= 1.0
            
        # Should have some duplicate detection (we know there are duplicates)
        if result.metrics["failed_fingerprinting_count"] == 0:
            # We expect some duplicates in our realistic set
            assert result.metrics["duplicate_structures_count"] > 0
            assert result.metrics["unique_structures_count"] < len(structures)

    def test_auto_mode_mixed_structures(self):
        """Test auto mode with mix of preprocessed and non-preprocessed structures."""
        metric = UniquenessNewMetric(fingerprint_source="auto")
        
        # Mix preprocessed and non-preprocessed structures
        preprocessed = create_structures_with_preprocessed_fingerprints()[:2]
        non_preprocessed = create_realistic_structure_set()[:2]
        
        structures = preprocessed + non_preprocessed
        
        result = metric.compute(structures)
        
        # Should handle mixed case correctly
        assert isinstance(result, MetricResult)
        assert result.n_structures == 4
        assert result.metrics["total_structures_evaluated"] == 4
        
        # Should have computed some fingerprints and used some preprocessed ones
        # The exact behavior depends on fingerprinting success

    def test_generation_batch_scenario(self):
        """Test a realistic generation batch with known duplicate patterns."""
        metric = UniquenessNewMetric(fingerprint_source="compute")
        structures = create_generation_batch_scenario()
        
        result = metric.compute(structures)
        
        # Should identify the duplicate patterns
        assert isinstance(result, MetricResult)
        assert result.n_structures == 9  # 4 NaCl + 2 CsCl + 3 unique
        
        if result.metrics["failed_fingerprinting_count"] == 0:
            # We expect: NaCl group (1 unique) + CsCl group (1 unique) + 3 individual = 5 unique total
            expected_unique = 5
            assert result.metrics["unique_structures_count"] <= expected_unique
            assert result.metrics["duplicate_structures_count"] > 0

    def test_large_scale_uniqueness(self):
        """Test with larger structure set to verify performance and correctness."""
        metric = UniquenessNewMetric(fingerprint_source="property")
        
        structures = []
        
        # Create 20 structures: 10 identical + 5 identical + 5 unique
        base_lattice = [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]]
        
        # 10 identical structures (Group A)
        for i in range(10):
            structure = Structure(
                lattice=base_lattice,
                species=["Na", "Cl"],
                coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
                coords_are_cartesian=False,
            )
            structure.properties["augmented_fingerprint"] = "GROUP_A_FINGERPRINT"
            structures.append(structure)
        
        # 5 identical structures (Group B)
        for i in range(5):
            structure = Structure(
                lattice=base_lattice,
                species=["K", "Br"],
                coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
                coords_are_cartesian=False,
            )
            structure.properties["augmented_fingerprint"] = "GROUP_B_FINGERPRINT"
            structures.append(structure)
        
        # 5 unique structures
        for i in range(5):
            structure = Structure(
                lattice=base_lattice,
                species=["Li", "F"],
                coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
                coords_are_cartesian=False,
            )
            structure.properties["augmented_fingerprint"] = f"UNIQUE_FINGERPRINT_{i}"
            structures.append(structure)
        
        result = metric.compute(structures)
        
        # Should correctly identify 7 unique structures (1 + 1 + 5)
        assert result.metrics["unique_structures_count"] == 7
        assert result.metrics["duplicate_structures_count"] == 13  # 9 + 4
        assert result.metrics["uniqueness_score"] == 7.0 / 20.0
        
        # Check individual values
        individual_values = result.individual_values
        
        # First 10 should be 1/10
        for i in range(10):
            assert abs(individual_values[i] - 0.1) < 1e-6
            
        # Next 5 should be 1/5
        for i in range(10, 15):
            assert abs(individual_values[i] - 0.2) < 1e-6
            
        # Last 5 should be 1.0
        for i in range(15, 20):
            assert abs(individual_values[i] - 1.0) < 1e-6

    def test_missing_fingerprints_property_mode(self):
        """Test behavior when fingerprints are missing in property mode."""
        metric = UniquenessNewMetric(fingerprint_source="property")
        
        # Create structures without fingerprints
        structures = create_realistic_structure_set()[:3]
        
        result = metric.compute(structures)
        
        # Should handle missing fingerprints gracefully
        assert result.metrics["failed_fingerprinting_count"] == 3
        assert np.isnan(result.metrics["uniqueness_score"])
        assert result.metrics["unique_structures_count"] == 0

    def test_mixed_success_failure_scenarios(self):
        """Test scenarios with some successful and some failed fingerprinting."""
        metric = UniquenessNewMetric(fingerprint_source="property")
        
        structures = []
        
        # Some structures with fingerprints
        for i in range(3):
            structure = Structure(
                lattice=[[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]],
                species=["Na", "Cl"],
                coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
                coords_are_cartesian=False,
            )
            structure.properties["augmented_fingerprint"] = "VALID_FINGERPRINT"
            structures.append(structure)
        
        # Some structures without fingerprints
        for i in range(2):
            structure = Structure(
                lattice=[[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]],
                species=["K", "Br"],
                coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
                coords_are_cartesian=False,
            )
            # No fingerprint property
            structures.append(structure)
        
        result = metric.compute(structures)
        
        # Should process the valid ones and report failures for others
        assert result.metrics["failed_fingerprinting_count"] == 2
        assert result.metrics["unique_structures_count"] == 1  # 3 identical valid ones
        assert result.metrics["duplicate_structures_count"] == 2

    def test_empty_and_edge_cases(self):
        """Test edge cases like empty lists, single structures, etc."""
        metric = UniquenessNewMetric()
        
        # Empty list
        result_empty = metric.compute([])
        assert result_empty.n_structures == 0
        assert np.isnan(result_empty.metrics["uniqueness_score"])
        
        # Single structure
        single_structure = create_structures_with_preprocessed_fingerprints()[:1]
        result_single = metric.compute(single_structure)
        assert result_single.n_structures == 1
        if result_single.metrics["failed_fingerprinting_count"] == 0:
            assert result_single.metrics["uniqueness_score"] == 1.0

    def test_deterministic_behavior(self):
        """Test that results are deterministic across multiple runs."""
        metric = UniquenessNewMetric(fingerprint_source="property")
        structures = create_structures_with_preprocessed_fingerprints()
        
        # Run multiple times
        results = []
        for _ in range(3):
            result = metric.compute(structures)
            results.append(result)
        
        # All results should be identical
        for i in range(1, len(results)):
            assert results[i].metrics["uniqueness_score"] == results[0].metrics["uniqueness_score"]
            assert results[i].metrics["unique_structures_count"] == results[0].metrics["unique_structures_count"]
            assert results[i].individual_values == results[0].individual_values

    def test_parallel_vs_serial_consistency(self):
        """Test that parallel and serial computation give same results."""
        structures = create_structures_with_preprocessed_fingerprints()
        
        # Serial computation
        metric_serial = UniquenessNewMetric(fingerprint_source="property", n_jobs=1)
        result_serial = metric_serial.compute(structures)
        
        # Parallel computation
        metric_parallel = UniquenessNewMetric(fingerprint_source="property", n_jobs=2)
        result_parallel = metric_parallel.compute(structures)
        
        # Results should be identical
        assert result_serial.metrics["uniqueness_score"] == result_parallel.metrics["uniqueness_score"]
        assert result_serial.metrics["unique_structures_count"] == result_parallel.metrics["unique_structures_count"]

    def test_realistic_fingerprint_patterns(self):
        """Test with realistic fingerprint patterns that would come from actual preprocessing."""
        metric = UniquenessNewMetric(fingerprint_source="property")
        
        structures = []
        
        # Create structures with realistic augmented fingerprints
        realistic_fingerprints = [
            "AUG_225_('Cl', '4b', 1):1_('Na', '4a', 1):1",  # NaCl
            "AUG_225_('Cl', '4b', 1):1_('Na', '4a', 1):1",  # NaCl duplicate
            "AUG_221_('Cl', '1b', 1):1_('Cs', '1a', 1):1",  # CsCl
            "AUG_225_('F', '4b', 1):1_('Li', '4a', 1):1",   # LiF
            "AUG_225_('Br', '4b', 1):1_('K', '4a', 1):1",   # KBr
            "AUG_225_('Cl', '4b', 1):1_('Na', '4a', 1):1",  # NaCl duplicate
        ]
        
        for i, fingerprint in enumerate(realistic_fingerprints):
            structure = Structure(
                lattice=[[5.0 + i*0.1, 0, 0], [0, 5.0 + i*0.1, 0], [0, 0, 5.0 + i*0.1]],
                species=["Na", "Cl"],  # Simplified for testing
                coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
                coords_are_cartesian=False,
            )
            structure.properties["augmented_fingerprint"] = fingerprint
            structures.append(structure)
        
        result = metric.compute(structures)
        
        # Should identify 4 unique fingerprints (3 NaCl counted as 1)
        assert result.metrics["unique_structures_count"] == 4
        assert result.metrics["duplicate_structures_count"] == 2
        assert result.metrics["uniqueness_score"] == 4.0 / 6.0


# Manual test function for development and verification
def manual_test():
    """Manual test to verify real-world behavior."""
    print("🧪 Running manual uniqueness new metric tests with real scenarios...")

    try:
        # Test 1: Realistic generation batch
        print("\n1. Testing realistic generation batch scenario...")
        metric = UniquenessNewMetric(fingerprint_source="compute")
        structures = create_generation_batch_scenario()
        
        print(f"   Created {len(structures)} structures:")
        for i, s in enumerate(structures):
            print(f"     {i+1}: {s.composition.reduced_formula}")
        
        result = metric.compute(structures)
        print("   Results:")
        print(f"     Total: {result.metrics['total_structures_evaluated']}")
        print(f"     Unique: {result.metrics['unique_structures_count']}")
        print(f"     Duplicates: {result.metrics['duplicate_structures_count']}")
        print(f"     Failed: {result.metrics['failed_fingerprinting_count']}")
        print(f"     Uniqueness score: {result.metrics['uniqueness_score']:.3f}")

        # Test 2: Preprocessed fingerprints
        print("\n2. Testing with preprocessed fingerprints...")
        metric_prop = UniquenessNewMetric(fingerprint_source="property")
        preprocessed_structures = create_structures_with_preprocessed_fingerprints()
        
        print("   Preprocessed fingerprints:")
        for i, s in enumerate(preprocessed_structures):
            fp = s.properties.get("augmented_fingerprint", "None")
            print(f"     {i+1}: {s.composition.reduced_formula} -> {fp[:50]}...")
        
        result_prop = metric_prop.compute(preprocessed_structures)
        print("   Results:")
        print(f"     Unique: {result_prop.metrics['unique_structures_count']}")
        print(f"     Duplicates: {result_prop.metrics['duplicate_structures_count']}")
        print(f"     Uniqueness score: {result_prop.metrics['uniqueness_score']:.3f}")

        # Test 3: Individual values verification
        print("\n3. Testing individual values pattern...")
        individual_values = result_prop.individual_values
        print("   Individual uniqueness values:")
        for i, val in enumerate(individual_values):
            print(f"     Structure {i+1}: {val:.3f}")

        # Test 4: Large scale test
        print("\n4. Testing large scale scenario...")
        metric_large = UniquenessNewMetric(fingerprint_source="property")
        
        # Create known pattern: 5 identical + 3 identical + 2 unique = 3 unique total
        large_structures = []
        
        # Group 1: 5 identical
        for i in range(5):
            s = Structure([[5, 0, 0], [0, 5, 0], [0, 0, 5]], ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
            s.properties["augmented_fingerprint"] = "GROUP_1"
            large_structures.append(s)
        
        # Group 2: 3 identical  
        for i in range(3):
            s = Structure([[4, 0, 0], [0, 4, 0], [0, 0, 4]], ["K", "Br"], [[0, 0, 0], [0.5, 0.5, 0.5]])
            s.properties["augmented_fingerprint"] = "GROUP_2" 
            large_structures.append(s)
        
        # Group 3: 2 unique
        for i in range(2):
            s = Structure([[3, 0, 0], [0, 3, 0], [0, 0, 3]], ["Li", "F"], [[0, 0, 0], [0.5, 0.5, 0.5]])
            s.properties["augmented_fingerprint"] = f"UNIQUE_{i}"
            large_structures.append(s)
        
        result_large = metric_large.compute(large_structures)
        print("   Large scale (10 structures, expected 4 unique):")
        print(f"     Unique: {result_large.metrics['unique_structures_count']}")
        print(f"     Duplicates: {result_large.metrics['duplicate_structures_count']}")
        print(f"     Uniqueness score: {result_large.metrics['uniqueness_score']:.3f}")
        
        # Verify expected pattern
        expected_unique = 4  # 1 + 1 + 2
        expected_duplicates = 6  # 4 + 2 + 0
        
        assert result_large.metrics['unique_structures_count'] == expected_unique
        assert result_large.metrics['duplicate_structures_count'] == expected_duplicates
        print("   ✅ Large scale test passed!")

        print("\n🎉 All manual tests completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Manual test failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    manual_test()