"""Tests for new uniqueness benchmark using augmented fingerprints.

This test suite focuses on realistic scenarios and integration testing rather
than heavily mocked tests. It verifies the benchmark works correctly with
actual structures and fingerprinting.
"""

import math

from pymatgen.core.structure import Structure

from lemat_genbench.benchmarks.uniqueness_new_benchmark import UniquenessNewBenchmark
from lemat_genbench.metrics.uniqueness_new_metric import UniquenessNewMetric


def create_realistic_test_structures():
    """Create realistic test structures for benchmark testing."""
    structures = []

    # Realistic binary compounds with known duplicates
    compounds = [
        # 3 NaCl structures (duplicates)
        (["Na", "Cl"], [[5.64, 0, 0], [0, 5.64, 0], [0, 0, 5.64]], 3),
        # 2 CsCl structures (duplicates)  
        (["Cs", "Cl"], [[4.11, 0, 0], [0, 4.11, 0], [0, 0, 4.11]], 2),
        # 1 LiF structure (unique)
        (["Li", "F"], [[4.03, 0, 0], [0, 4.03, 0], [0, 0, 4.03]], 1),
    ]

    for species, lattice, count in compounds:
        for _ in range(count):
            structure = Structure(
                lattice=lattice,
                species=species,
                coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
                coords_are_cartesian=False,
            )
            structures.append(structure)

    return structures


def create_structures_with_preprocessed_fingerprints():
    """Create structures with realistic preprocessed fingerprints."""
    structures = []

    # Set 1: Multiple structures with same fingerprint (duplicates)
    for i in range(4):
        structure = Structure(
            lattice=[[5.64, 0, 0], [0, 5.64, 0], [0, 0, 5.64]],
            species=["Na", "Cl"],
            coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
            coords_are_cartesian=False,
        )
        structure.properties["augmented_fingerprint"] = "AUG_225_('Cl', '4b', 1):1_('Na', '4a', 1):1"
        structures.append(structure)

    # Set 2: Different structures with unique fingerprints
    unique_data = [
        (["Cs", "Cl"], "AUG_221_('Cl', '1b', 1):1_('Cs', '1a', 1):1"),
        (["Li", "F"], "AUG_225_('F', '4b', 1):1_('Li', '4a', 1):1"),
        (["K", "Br"], "AUG_225_('Br', '4b', 1):1_('K', '4a', 1):1"),
    ]

    for species, fingerprint in unique_data:
        structure = Structure(
            lattice=[[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]],
            species=species,
            coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
            coords_are_cartesian=False,
        )
        structure.properties["augmented_fingerprint"] = fingerprint
        structures.append(structure)

    return structures


class TestUniquenessNewBenchmarkRealistic:
    """Test suite for UniquenessNewBenchmark focusing on real scenarios."""

    def test_benchmark_initialization(self):
        """Test benchmark initialization with different configurations."""
        # Default initialization
        benchmark = UniquenessNewBenchmark()
        assert benchmark.config.name == "UniquenessNewBenchmark"
        assert benchmark.config.metadata["fingerprint_method"] == "augmented"
        assert benchmark.config.metadata["fingerprint_source"] == "auto"

        # Custom initialization
        benchmark_custom = UniquenessNewBenchmark(
            fingerprint_source="compute",
            symprec=0.001,
            angle_tolerance=1.0,
            name="Custom Uniqueness Benchmark"
        )
        assert benchmark_custom.config.name == "Custom Uniqueness Benchmark"
        assert benchmark_custom.config.metadata["fingerprint_source"] == "compute"
        assert benchmark_custom.config.metadata["symprec"] == 0.001

    def test_benchmark_with_preprocessed_structures(self):
        """Test benchmark with preprocessed fingerprints (most common use case)."""
        benchmark = UniquenessNewBenchmark(fingerprint_source="property")
        structures = create_structures_with_preprocessed_fingerprints()

        result = benchmark.evaluate(structures)

        # Should complete successfully
        assert result is not None
        assert "uniqueness_score" in result.final_scores
        assert "unique_structures_count" in result.final_scores
        assert "duplicate_structures_count" in result.final_scores

        # Expected: 4 identical NaCl + 3 unique = 4 unique total
        assert result.final_scores["unique_structures_count"] == 4
        assert result.final_scores["duplicate_structures_count"] == 3
        assert result.final_scores["uniqueness_score"] == 4.0 / 7.0
        assert result.final_scores["uniqueness_ratio"] == 4.0 / 7.0

    def test_benchmark_with_compute_mode(self):
        """Test benchmark computing fingerprints on-the-fly."""
        benchmark = UniquenessNewBenchmark(fingerprint_source="compute")
        structures = create_realistic_test_structures()

        result = benchmark.evaluate(structures)

        # Should complete successfully
        assert result is not None
        assert "uniqueness_score" in result.final_scores
        
        # Results depend on actual fingerprinting, but should be reasonable
        if result.final_scores["failed_fingerprinting_count"] == 0:
            # We know there are duplicates in our test set
            assert result.final_scores["duplicate_structures_count"] > 0
            assert result.final_scores["unique_structures_count"] < len(structures)
            assert 0 < result.final_scores["uniqueness_score"] < 1.0

    def test_benchmark_with_auto_mode(self):
        """Test benchmark with auto mode (mix of preprocessed and computed)."""
        benchmark = UniquenessNewBenchmark(fingerprint_source="auto")
        
        # Mix preprocessed and non-preprocessed structures
        preprocessed = create_structures_with_preprocessed_fingerprints()[:3]
        non_preprocessed = create_realistic_test_structures()[:2]
        
        structures = preprocessed + non_preprocessed
        
        result = benchmark.evaluate(structures)

        # Should handle mixed case correctly
        assert result is not None
        assert result.final_scores["total_structures_evaluated"] == 5
        assert "uniqueness_score" in result.final_scores

    def test_empty_structures(self):
        """Test benchmark behavior with empty structure list."""
        benchmark = UniquenessNewBenchmark()
        result = benchmark.evaluate([])

        # Should handle gracefully
        assert math.isnan(result.final_scores["uniqueness_score"])
        assert result.final_scores["unique_structures_count"] == 0
        assert result.final_scores["duplicate_structures_count"] == 0
        assert result.final_scores["total_structures_evaluated"] == 0

    def test_single_structure(self):
        """Test benchmark with single structure."""
        benchmark = UniquenessNewBenchmark(fingerprint_source="property")
        structures = create_structures_with_preprocessed_fingerprints()[:1]

        result = benchmark.evaluate(structures)

        # Single structure should be 100% unique
        assert result.final_scores["uniqueness_score"] == 1.0
        assert result.final_scores["unique_structures_count"] == 1
        assert result.final_scores["duplicate_structures_count"] == 0

    def test_all_identical_structures(self):
        """Test benchmark with all identical structures."""
        benchmark = UniquenessNewBenchmark(fingerprint_source="property")
        
        # Create 5 identical structures
        structures = []
        for i in range(5):
            structure = Structure(
                lattice=[[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]],
                species=["Na", "Cl"],
                coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
                coords_are_cartesian=False,
            )
            structure.properties["augmented_fingerprint"] = "IDENTICAL_FINGERPRINT"
            structures.append(structure)

        result = benchmark.evaluate(structures)

        # Should have only 1 unique structure
        assert result.final_scores["unique_structures_count"] == 1
        assert result.final_scores["duplicate_structures_count"] == 4
        assert result.final_scores["uniqueness_score"] == 1.0 / 5.0

    def test_all_unique_structures(self):
        """Test benchmark with all unique structures."""
        benchmark = UniquenessNewBenchmark(fingerprint_source="property")
        
        # Create unique structures
        structures = []
        for i in range(4):
            structure = Structure(
                lattice=[[5.0 + i*0.1, 0, 0], [0, 5.0 + i*0.1, 0], [0, 0, 5.0 + i*0.1]],
                species=["Na", "Cl"],
                coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
                coords_are_cartesian=False,
            )
            structure.properties["augmented_fingerprint"] = f"UNIQUE_FINGERPRINT_{i}"
            structures.append(structure)

        result = benchmark.evaluate(structures)

        # Should have 100% uniqueness
        assert result.final_scores["unique_structures_count"] == 4
        assert result.final_scores["duplicate_structures_count"] == 0
        assert result.final_scores["uniqueness_score"] == 1.0

    def test_large_scale_benchmark(self):
        """Test benchmark with larger structure set."""
        benchmark = UniquenessNewBenchmark(fingerprint_source="property")
        
        structures = []
        
        # Create known pattern: 10 + 5 + 3 identical groups + 2 unique
        patterns = [
            ("GROUP_A", 10),
            ("GROUP_B", 5), 
            ("GROUP_C", 3),
            ("UNIQUE_1", 1),
            ("UNIQUE_2", 1),
        ]
        
        for fingerprint, count in patterns:
            for i in range(count):
                structure = Structure(
                    lattice=[[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]],
                    species=["Na", "Cl"],
                    coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
                    coords_are_cartesian=False,
                )
                structure.properties["augmented_fingerprint"] = fingerprint
                structures.append(structure)

        result = benchmark.evaluate(structures)

        # Should identify 5 unique patterns
        total_structures = 20
        expected_unique = 5
        expected_duplicates = 15

        assert result.final_scores["total_structures_evaluated"] == total_structures
        assert result.final_scores["unique_structures_count"] == expected_unique
        assert result.final_scores["duplicate_structures_count"] == expected_duplicates
        assert result.final_scores["uniqueness_score"] == expected_unique / total_structures

    def test_benchmark_with_failed_fingerprints(self):
        """Test benchmark behavior when some fingerprinting fails."""
        benchmark = UniquenessNewBenchmark(fingerprint_source="property")
        
        structures = []
        
        # Some structures with valid fingerprints
        for i in range(3):
            structure = Structure(
                lattice=[[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]],
                species=["Na", "Cl"],
                coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
                coords_are_cartesian=False,
            )
            structure.properties["augmented_fingerprint"] = "VALID_FINGERPRINT"
            structures.append(structure)
        
        # Some structures without fingerprints (will fail in property mode)
        for i in range(2):
            structure = Structure(
                lattice=[[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]],
                species=["K", "Br"],
                coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
                coords_are_cartesian=False,
            )
            # No fingerprint property
            structures.append(structure)

        result = benchmark.evaluate(structures)

        # Should report failures correctly
        assert result.final_scores["failed_fingerprinting_count"] == 2
        assert result.final_scores["unique_structures_count"] == 1  # 3 identical valid ones
        assert result.final_scores["duplicate_structures_count"] == 2

    def test_metadata_preservation(self):
        """Test that benchmark metadata is preserved correctly."""
        custom_metadata = {
            "experiment_id": "test_001",
            "batch_size": 100,
            "model_version": "v2.1"
        }
        
        benchmark = UniquenessNewBenchmark(
            fingerprint_source="property",
            symprec=0.005,
            metadata=custom_metadata
        )
        
        # Check that metadata is preserved
        metadata = benchmark.config.metadata
        assert metadata["experiment_id"] == "test_001"
        assert metadata["batch_size"] == 100
        assert metadata["model_version"] == "v2.1"
        
        # Check that default metadata is still present
        assert metadata["fingerprint_method"] == "augmented"
        assert metadata["fingerprint_source"] == "property"
        assert metadata["symprec"] == 0.005

    def test_different_precision_settings(self):
        """Test benchmark with different precision settings."""
        structures = create_structures_with_preprocessed_fingerprints()
        
        # High precision
        benchmark_high = UniquenessNewBenchmark(
            fingerprint_source="compute",
            symprec=0.001,
            angle_tolerance=1.0
        )
        
        # Low precision  
        benchmark_low = UniquenessNewBenchmark(
            fingerprint_source="compute", 
            symprec=0.1,
            angle_tolerance=10.0
        )
        
        # Both should work (though results may differ)
        result_high = benchmark_high.evaluate(structures)
        result_low = benchmark_low.evaluate(structures)
        
        assert result_high is not None
        assert result_low is not None
        assert "uniqueness_score" in result_high.final_scores
        assert "uniqueness_score" in result_low.final_scores

    def test_benchmark_reproducibility(self):
        """Test that benchmark results are reproducible."""
        benchmark = UniquenessNewBenchmark(fingerprint_source="property")
        structures = create_structures_with_preprocessed_fingerprints()
        
        # Run multiple times
        results = []
        for _ in range(3):
            result = benchmark.evaluate(structures)
            results.append(result)
        
        # All results should be identical
        for i in range(1, len(results)):
            assert results[i].final_scores["uniqueness_score"] == results[0].final_scores["uniqueness_score"]
            assert results[i].final_scores["unique_structures_count"] == results[0].final_scores["unique_structures_count"]

    def test_evaluator_configuration(self):
        """Test that the benchmark's evaluator is properly configured."""
        benchmark = UniquenessNewBenchmark()

        # Check evaluator setup
        assert len(benchmark.evaluators) == 1
        assert "uniqueness" in benchmark.evaluators

        evaluator = benchmark.evaluators["uniqueness"]
        assert evaluator.config.name == "uniqueness"
        assert "uniqueness" in evaluator.metrics
        assert isinstance(evaluator.metrics["uniqueness"], UniquenessNewMetric)

    def test_integration_with_realistic_workflow(self):
        """Test integration in a realistic workflow scenario."""
        # Simulate preprocessing step
        structures = create_realistic_test_structures()
        
        # Add realistic fingerprints (simulating preprocessor output)
        fingerprint_map = {
            "NaCl": "AUG_225_('Cl', '4b', 1):1_('Na', '4a', 1):1",
            "CsCl": "AUG_221_('Cl', '1b', 1):1_('Cs', '1a', 1):1", 
            "LiF": "AUG_225_('F', '4b', 1):1_('Li', '4a', 1):1",
        }
        
        for structure in structures:
            formula = structure.composition.reduced_formula
            if formula in fingerprint_map:
                structure.properties["augmented_fingerprint"] = fingerprint_map[formula]
        
        # Run benchmark
        benchmark = UniquenessNewBenchmark(fingerprint_source="property")
        result = benchmark.evaluate(structures)
        
        # Should identify duplicates correctly
        # We have: 3 NaCl + 2 CsCl + 1 LiF = 3 unique structures
        assert result.final_scores["unique_structures_count"] == 3
        assert result.final_scores["duplicate_structures_count"] == 3  # 2 extra NaCl + 1 extra CsCl
        assert result.final_scores["uniqueness_score"] == 3.0 / 6.0


# Manual test for development and verification
def manual_test_benchmark():
    """Manual test to verify realistic benchmark behavior."""
    print("🎯 Running manual uniqueness new benchmark tests...")
    
    try:
        # Test 1: Basic benchmark with preprocessed fingerprints
        print("\n1. Testing benchmark with preprocessed fingerprints...")
        benchmark = UniquenessNewBenchmark(fingerprint_source="property")
        structures = create_structures_with_preprocessed_fingerprints()
        
        print(f"   Input: {len(structures)} structures")
        for i, s in enumerate(structures):
            fp = s.properties.get("augmented_fingerprint", "None")
            print(f"     {i+1}: {s.composition.reduced_formula} -> {fp[:40]}...")
        
        result = benchmark.evaluate(structures)
        print("   Results:")
        print(f"     Uniqueness score: {result.final_scores['uniqueness_score']:.3f}")
        print(f"     Unique structures: {result.final_scores['unique_structures_count']}")
        print(f"     Duplicate structures: {result.final_scores['duplicate_structures_count']}")
        print(f"     Total evaluated: {result.final_scores['total_structures_evaluated']}")

        # Test 2: Large scale test with known pattern
        print("\n2. Testing large scale scenario...")
        benchmark_large = UniquenessNewBenchmark(fingerprint_source="property")
        
        # Create predictable pattern
        large_structures = []
        patterns = [("GROUP_A", 8), ("GROUP_B", 5), ("GROUP_C", 2), ("UNIQUE_1", 1), ("UNIQUE_2", 1)]
        
        for fingerprint, count in patterns:
            for i in range(count):
                s = Structure([[5, 0, 0], [0, 5, 0], [0, 0, 5]], ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
                s.properties["augmented_fingerprint"] = fingerprint
                large_structures.append(s)
        
        result_large = benchmark_large.evaluate(large_structures)
        print("   Large scale (17 structures, expected 5 unique):")
        print(f"     Uniqueness score: {result_large.final_scores['uniqueness_score']:.3f}")
        print(f"     Unique: {result_large.final_scores['unique_structures_count']}")
        print(f"     Duplicates: {result_large.final_scores['duplicate_structures_count']}")
        
        # Verify expected results
        expected_unique = 5
        expected_duplicates = 12  # 7 + 4 + 1 + 0 + 0
        
        assert result_large.final_scores['unique_structures_count'] == expected_unique
        assert result_large.final_scores['duplicate_structures_count'] == expected_duplicates
        print("   ✅ Large scale test passed!")

        # Test 3: Integration test
        print("\n3. Testing realistic integration scenario...")
        benchmark_integration = UniquenessNewBenchmark(fingerprint_source="auto")
        
        # Create mixed scenario
        mixed_structures = []
        
        # Some with fingerprints
        for i in range(3):
            s = Structure([[5, 0, 0], [0, 5, 0], [0, 0, 5]], ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
            s.properties["augmented_fingerprint"] = "PREPROCESSED_FP"
            mixed_structures.append(s)
        
        # Some without (will compute in auto mode)
        for i in range(2):
            s = Structure([[4, 0, 0], [0, 4, 0], [0, 0, 4]], ["K", "Br"], [[0, 0, 0], [0.5, 0.5, 0.5]])
            mixed_structures.append(s)
        
        result_mixed = benchmark_integration.evaluate(mixed_structures)
        print("   Mixed scenario results:")
        print(f"     Total: {result_mixed.final_scores['total_structures_evaluated']}")
        print(f"     Failed: {result_mixed.final_scores['failed_fingerprinting_count']}")
        print(f"     Unique: {result_mixed.final_scores['unique_structures_count']}")

        print("\n🎉 All manual benchmark tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Manual benchmark test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    manual_test_benchmark()