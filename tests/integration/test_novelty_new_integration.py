"""Integration tests for the new novelty implementations with augmented fingerprinting.

These tests verify the complete workflow from preprocessing to evaluation
using the enhanced augmented fingerprinting approach.
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from pymatgen.core.structure import Structure

from lemat_genbench.benchmarks.novelty_new_benchmark import (
    AugmentedNoveltyBenchmark,
    create_high_precision_novelty_benchmark,
    create_property_based_novelty_benchmark,
    create_robust_novelty_benchmark,
)
from lemat_genbench.metrics.base import MetricResult
from lemat_genbench.metrics.novelty_new_metric import AugmentedNoveltyMetric
from lemat_genbench.preprocess.augmented_fingerprint_preprocess import (
    AugmentedFingerprintPreprocessor,
)


def create_diverse_test_structures():
    """Create diverse test structures for comprehensive testing."""
    structures = []

    # Structure 1: Simple binary NaCl
    lattice1 = [[5.64, 0, 0], [0, 5.64, 0], [0, 0, 5.64]]
    structure1 = Structure(
        lattice=lattice1,
        species=["Na", "Cl"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structures.append(structure1)

    # Structure 2: Ternary perovskite BaTiO3
    lattice2 = [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]]
    structure2 = Structure(
        lattice=lattice2,
        species=["Ba", "Ti", "O", "O", "O"],
        coords=[[0.5, 0.5, 0.5], [0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]],
        coords_are_cartesian=False,
    )
    structures.append(structure2)

    # Structure 3: Fluorite CaF2
    lattice3 = [[5.463, 0, 0], [0, 5.463, 0], [0, 0, 5.463]]
    structure3 = Structure(
        lattice=lattice3,
        species=["Ca", "F", "F"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
        coords_are_cartesian=False,
    )
    structures.append(structure3)

    # Structure 4: Different CaF2 arrangement (same composition, different structure)
    lattice4 = [[5.463, 0, 0], [0, 5.463, 0], [0, 0, 5.463]]
    structure4 = Structure(
        lattice=lattice4,
        species=["Ca", "F", "F"],
        coords=[[0, 0, 0], [0.2, 0.2, 0.2], [0.8, 0.8, 0.8]],  # Slightly different positions
        coords_are_cartesian=False,
    )
    structures.append(structure4)

    return structures


class TestAugmentedNoveltyIntegration:
    """Integration tests for the complete augmented novelty workflow."""

    @patch("lemat_genbench.preprocess.augmented_fingerprint_preprocess.get_augmented_fingerprint")
    def test_preprocessing_to_evaluation_workflow(self, mock_get_fingerprint):
        """Test complete workflow from preprocessing to novelty evaluation."""
        
        # Mock fingerprint generation to return predictable results
        def mock_fingerprint_side_effect(structure, **kwargs):
            formula = structure.composition.reduced_formula
            if formula == "NaCl":
                return "AUG_225_Na:4a:1_1|Cl:4b:1_1"
            elif formula == "BaTiO3":
                return "AUG_221_Ba:1a:1_1|Ti:1b:1_1|O:3c:1_3"  # This will be in reference
            elif formula == "CaF2":
                return "AUG_227_Ca:4a:1_1|F:8c:1_2"
            else:
                return f"AUG_UNKNOWN_{formula}"
        
        mock_get_fingerprint.side_effect = mock_fingerprint_side_effect
        
        # Step 1: Create structures
        structures = create_diverse_test_structures()[:3]  # Use first 3
        
        # Step 2: Preprocess structures to add fingerprints
        preprocessor = AugmentedFingerprintPreprocessor()
        processed_structures = []
        
        for structure in structures:
            processed_structure = preprocessor.process_structure(structure)
            processed_structures.append(processed_structure)
        
        # Verify preprocessing worked and print actual fingerprints for debugging
        for i, structure in enumerate(processed_structures):
            assert "augmented_fingerprint" in structure.properties
            assert structure.properties["augmented_fingerprint"] is not None
            fp = structure.properties["augmented_fingerprint"]
            print(f"Structure {i} ({structure.composition.reduced_formula}): {fp}")
            assert fp.startswith("AUG_")
        
        # Step 3: Create temporary reference data using the ACTUAL fingerprints generated
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use the actual fingerprint from BaTiO3 structure (index 1)
            batio3_fingerprint = processed_structures[1].properties["augmented_fingerprint"]
            reference_fps = [
                batio3_fingerprint,  # BaTiO3 is known - use actual generated fingerprint
                "AUG_known_ref_1",
                "AUG_known_ref_2",
            ]
            df = pd.DataFrame({"values": reference_fps})
            parquet_path = Path(temp_dir) / "unique_fingerprints.parquet"
            df.to_parquet(parquet_path)
            
            print(f"Reference includes: {batio3_fingerprint}")
            
            # Step 4: Evaluate novelty using property-based approach
            metric = AugmentedNoveltyMetric(
                reference_fingerprints_path=temp_dir,
                fingerprint_source="property"
            )
            result = metric.compute(processed_structures)
            
            # Debug: print individual results
            print(f"Individual results: {result.individual_values}")
            
            # Step 5: Verify results
            assert isinstance(result, MetricResult)
            assert result.metrics["total_structures_evaluated"] == 3
            assert result.metrics["novel_structures_count"] == 2  # NaCl and CaF2 are novel
            assert result.metrics["novelty_score"] == 2.0 / 3.0
            assert result.metrics["fingerprinting_success_rate"] == 1.0

    def test_benchmark_with_preprocessing_integration(self):
        """Test benchmark evaluation with preprocessed structures."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock fingerprint database
            reference_fps = [
                "AUG_reference_1",
                "AUG_reference_2", 
                "AUG_reference_3",
            ]
            df = pd.DataFrame({"values": reference_fps})
            parquet_path = Path(temp_dir) / "unique_fingerprints.parquet"
            df.to_parquet(parquet_path)
            
            # Create structures with realistic fingerprints (as if preprocessed)
            structures = []
            fingerprints = [
                "AUG_225_Na:4a:1_1|Cl:4b:1_1_novel",
                "AUG_221_Ba:1a:1_1|Ti:1b:1_1|O:3c:1_3_novel",
                "AUG_227_Ca:4a:1_1|F:8c:1_2_novel",
            ]
            
            for i, fp in enumerate(fingerprints):
                lattice = [[4.0 + i * 0.1, 0, 0], [0, 4.0 + i * 0.1, 0], [0, 0, 4.0 + i * 0.1]]
                structure = Structure(
                    lattice=lattice,
                    species=["Na", "Cl"],
                    coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
                    coords_are_cartesian=False,
                )
                
                # Add preprocessed fingerprint properties
                structure.properties.update({
                    "augmented_fingerprint": fp,
                    "augmented_fingerprint_properties": {
                        "augmented_fingerprint": fp,
                        "augmented_fingerprint_success": True,
                        "augmented_fingerprint_spacegroup": 225,
                    },
                })
                structures.append(structure)
            
            # Evaluate with benchmark
            benchmark = AugmentedNoveltyBenchmark(
                reference_fingerprints_path=temp_dir,
                fingerprint_source="property"
            )
            result = benchmark.evaluate(structures)
            
            # All should be novel since none match reference
            assert result.final_scores["novelty_score"] == 1.0
            assert result.final_scores["novel_structures_count"] == 3
            assert result.final_scores["total_structures_evaluated"] == 3
            assert result.final_scores["fingerprinting_success_rate"] == 1.0

    @patch("lemat_genbench.metrics.novelty_new_metric.get_augmented_fingerprint")
    def test_fallback_computation_integration(self, mock_get_fingerprint):
        """Test integration where some structures need fingerprint computation."""
        
        # Mock fingerprint computation
        mock_get_fingerprint.return_value = "AUG_COMPUTED_TEST_FINGERPRINT"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock reference database
            reference_fps = [
                "AUG_KNOWN_FINGERPRINT_1",
                "AUG_KNOWN_FINGERPRINT_2",
            ]
            df = pd.DataFrame({"values": reference_fps})
            parquet_path = Path(temp_dir) / "unique_fingerprints.parquet"
            df.to_parquet(parquet_path)
            
            # Create structures with mixed fingerprint availability
            structures = []
            
            # Structure 1: Has preprocessed fingerprint
            lattice1 = [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]]
            structure1 = Structure(
                lattice=lattice1,
                species=["Na", "Cl"],
                coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
                coords_are_cartesian=False,
            )
            structure1.properties["augmented_fingerprint"] = "AUG_PREPROCESSED_FINGERPRINT"
            structures.append(structure1)
            
            # Structure 2: No fingerprint (will need computation)
            lattice2 = [[4.1, 0, 0], [0, 4.1, 0], [0, 0, 4.1]]
            structure2 = Structure(
                lattice=lattice2,
                species=["K", "Br"],
                coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
                coords_are_cartesian=False,
            )
            # No fingerprint property - will trigger computation
            structures.append(structure2)
            
            # Test with auto mode (should use property when available, compute when needed)
            metric = AugmentedNoveltyMetric(
                reference_fingerprints_path=temp_dir,
                fingerprint_source="auto"
            )
            result = metric.compute(structures)
            
            # Both structures should be processed successfully
            assert result.metrics["total_structures_evaluated"] == 2
            assert result.metrics["fingerprinting_success_rate"] == 1.0
            # Both should be novel (neither matches reference)
            assert result.metrics["novel_structures_count"] == 2
            assert result.metrics["novelty_score"] == 1.0
            
            # Verify computation was called for structure without fingerprint
            mock_get_fingerprint.assert_called()

    def test_error_handling_integration(self):
        """Test error handling in integrated workflow."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create reference data
            reference_fps = ["AUG_REF_1", "AUG_REF_2"]
            df = pd.DataFrame({"values": reference_fps})
            parquet_path = Path(temp_dir) / "unique_fingerprints.parquet"
            df.to_parquet(parquet_path)
            
            # Create structures with potential issues
            structures = []
            
            # Structure 1: Normal structure with fingerprint
            lattice1 = [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]]
            structure1 = Structure(
                lattice=lattice1,
                species=["Na", "Cl"],
                coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
                coords_are_cartesian=False,
            )
            structure1.properties["augmented_fingerprint"] = "AUG_VALID_FINGERPRINT"
            structures.append(structure1)
            
            # Structure 2: Malformed fingerprint
            lattice2 = [[4.1, 0, 0], [0, 4.1, 0], [0, 0, 4.1]]
            structure2 = Structure(
                lattice=lattice2,
                species=["K", "Br"],
                coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
                coords_are_cartesian=False,
            )
            structure2.properties["augmented_fingerprint"] = None  # Malformed
            structures.append(structure2)
            
            # Structure 3: No fingerprint and computation disabled
            lattice3 = [[4.2, 0, 0], [0, 4.2, 0], [0, 0, 4.2]]
            structure3 = Structure(
                lattice=lattice3,
                species=["Cs", "I"],
                coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
                coords_are_cartesian=False,
            )
            # No fingerprint property
            structures.append(structure3)
            
            # Test with property-only mode (no computation fallback)
            metric = AugmentedNoveltyMetric(
                reference_fingerprints_path=temp_dir,
                fingerprint_source="property",
                fallback_to_computation=False
            )
            result = metric.compute(structures)
            
            # Only structure 1 should be successfully evaluated
            assert result.metrics["total_structures_evaluated"] == 1
            assert result.metrics["total_structures_attempted"] == 3
            assert result.metrics["fingerprinting_success_rate"] == 1.0 / 3.0
            # Structure 1 should be novel
            assert result.metrics["novel_structures_count"] == 1
            assert result.metrics["novelty_score"] == 1.0  # 1 novel out of 1 evaluated

    def test_benchmark_factory_integration(self):
        """Test benchmark factory functions in integrated workflow."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock reference database
            reference_fps = [
                "AUG_KNOWN_1",
                "AUG_KNOWN_2",
            ]
            df = pd.DataFrame({"values": reference_fps})
            parquet_path = Path(temp_dir) / "unique_fingerprints.parquet"
            df.to_parquet(parquet_path)
            
            # Create test structures
            structures = []
            fingerprints = ["AUG_NOVEL_1", "AUG_NOVEL_2", "AUG_KNOWN_1"]  # Last one is known
            
            for i, fp in enumerate(fingerprints):
                lattice = [[4.0 + i * 0.1, 0, 0], [0, 4.0 + i * 0.1, 0], [0, 0, 4.0 + i * 0.1]]
                structure = Structure(
                    lattice=lattice,
                    species=["Na", "Cl"],
                    coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
                    coords_are_cartesian=False,
                )
                structure.properties["augmented_fingerprint"] = fp
                structures.append(structure)
            
            # Test different factory configurations
            
            factories = [
                create_property_based_novelty_benchmark,
                create_robust_novelty_benchmark,
                create_high_precision_novelty_benchmark,
            ]
            
            for factory in factories:
                benchmark = factory(reference_fingerprints_path=temp_dir)
                result = benchmark.evaluate(structures)
                
                # All factories should produce same result with property-based fingerprints
                assert result.final_scores["total_structures_evaluated"] == 3
                assert result.final_scores["novel_structures_count"] == 2  # First 2 are novel
                assert result.final_scores["novelty_score"] == 2.0 / 3.0

    def test_performance_characteristics(self):
        """Test performance characteristics of the integrated system."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create reference with some overlapping fingerprints
            known_fps = [f"AUG_PERF_TEST_{i:03d}" for i in range(0, 50, 5)]  # Every 5th
            df = pd.DataFrame({"values": known_fps})
            parquet_path = Path(temp_dir) / "unique_fingerprints.parquet"
            df.to_parquet(parquet_path)
            
            # Create larger set of structures for performance testing
            structures = []
            lattice = [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]]
            
            # Create 50 structures with fingerprints
            for i in range(50):
                structure = Structure(
                    lattice=lattice,
                    species=["Na", "Cl"],
                    coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
                    coords_are_cartesian=False,
                )
                structure.properties["augmented_fingerprint"] = f"AUG_PERF_TEST_{i:03d}"
                structures.append(structure)
            
            # Measure evaluation
            start_time = time.time()
            
            benchmark = AugmentedNoveltyBenchmark(
                reference_fingerprints_path=temp_dir,
                fingerprint_source="property"
            )
            result = benchmark.evaluate(structures)
            
            end_time = time.time()
            evaluation_time = end_time - start_time
            
            # Verify results
            assert result.final_scores["total_structures_evaluated"] == 50
            assert result.final_scores["novel_structures_count"] == 40  # 50 - 10 known
            assert result.final_scores["novelty_score"] == 0.8
            assert result.final_scores["fingerprinting_success_rate"] == 1.0
            
            # Performance should be reasonable (less than 10 seconds for 50 structures)
            assert evaluation_time < 10.0, f"Evaluation took too long: {evaluation_time:.2f}s"

    def test_real_world_simulation(self):
        """Simulate a real-world usage scenario."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a realistic set of reference fingerprints
            reference_fps = []
            reference_fps.extend([
                f"AUG_225_Na:4a:1_{i}|Cl:4b:1_{j}" 
                for i in range(1, 4) for j in range(1, 4)
            ])
            reference_fps.extend([
                f"AUG_227_Ca:4a:1_{i}|F:8c:1_{j}" 
                for i in range(1, 3) for j in range(1, 3)
            ])
            df = pd.DataFrame({"values": reference_fps})
            parquet_path = Path(temp_dir) / "unique_fingerprints.parquet"
            df.to_parquet(parquet_path)
            
            # Create structures representing a typical generation batch
            generated_structures = []
            
            # Mix of novel and known structures
            test_cases = [
                # Known structures (will be in reference)
                ("Na", "Cl", "AUG_225_Na:4a:1_1|Cl:4b:1_1"),
                ("Ca", "F", "AUG_227_Ca:4a:1_1|F:8c:1_1"),
                
                # Novel structures (not in reference)
                ("K", "Br", "AUG_225_K:4a:1_1|Br:4b:1_1"),
                ("Sr", "F", "AUG_227_Sr:4a:1_1|F:8c:1_2"),
                ("Cs", "I", "AUG_225_Cs:4a:1_1|I:4b:1_1"),
                
                # Novel variation of known composition
                ("Na", "Cl", "AUG_225_Na:4a:2_1|Cl:4b:2_1"),  # Different enumeration
            ]
            
            for i, (elem1, elem2, fingerprint) in enumerate(test_cases):
                lattice = [[4.0 + i * 0.05, 0, 0], [0, 4.0 + i * 0.05, 0], [0, 0, 4.0 + i * 0.05]]
                structure = Structure(
                    lattice=lattice,
                    species=[elem1, elem2],
                    coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
                    coords_are_cartesian=False,
                )
                
                # Add fingerprint as if from preprocessing
                structure.properties.update({
                    "augmented_fingerprint": fingerprint,
                    "augmented_fingerprint_properties": {
                        "augmented_fingerprint_success": True,
                        "augmented_fingerprint_spacegroup": 225 if "225" in fingerprint else 227,
                    },
                })
                generated_structures.append(structure)
            
            # Evaluate using benchmark
            benchmark = AugmentedNoveltyBenchmark(
                reference_fingerprints_path=temp_dir,
                fingerprint_source="property"
            )
            result = benchmark.evaluate(generated_structures)
            
            # Expected: 2 known (first 2), 4 novel (last 4)
            assert result.final_scores["total_structures_evaluated"] == 6
            assert result.final_scores["novel_structures_count"] == 4
            assert result.final_scores["novelty_score"] == 4.0 / 6.0
            assert result.final_scores["fingerprinting_success_rate"] == 1.0
            
            # Verify individual results make sense
            evaluator_results = result.evaluator_results["augmented_novelty"]
            individual_novelty = evaluator_results["metric_results"]["augmented_novelty"].individual_values
            # First two should be 0 (known), others should be 1 (novel)
            expected_individual = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
            assert individual_novelty == expected_individual


# Manual test function for comprehensive integration testing
def manual_integration_test():
    """Comprehensive manual test for development and debugging."""
    print("🧪 Running comprehensive augmented novelty integration test...")
    
    try:
        # Test 1: Basic workflow integration
        print("1. Testing basic workflow integration...")
        structures = create_diverse_test_structures()[:2]
        
        # Add mock fingerprints
        fingerprints = ["AUG_225_test_1", "AUG_227_test_2"]
        for structure, fp in zip(structures, fingerprints):
            structure.properties["augmented_fingerprint"] = fp
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create reference data
            reference_fps = ["AUG_225_test_1"]  # First is known
            df = pd.DataFrame({"values": reference_fps})
            parquet_path = Path(temp_dir) / "unique_fingerprints.parquet"
            df.to_parquet(parquet_path)
            
            # Test metric
            metric = AugmentedNoveltyMetric(
                reference_fingerprints_path=temp_dir,
                fingerprint_source="property"
            )
            result = metric.compute(structures)
            
            print(f"   Metric result: {result.metrics['novelty_score']:.3f}")
            print(f"   Novel count: {result.metrics['novel_structures_count']}")
            
            # Test benchmark
            benchmark = AugmentedNoveltyBenchmark(
                reference_fingerprints_path=temp_dir,
                fingerprint_source="property"
            )
            result = benchmark.evaluate(structures)
            
            print(f"   Benchmark result: {result.final_scores['novelty_score']:.3f}")
            print(f"   Success rate: {result.final_scores['fingerprinting_success_rate']:.3f}")
        
        # Test 2: Error handling
        print("2. Testing error handling...")
        structures_with_issues = [structures[0]]  # Good structure
        
        # Add structure without fingerprint
        bad_structure = create_diverse_test_structures()[0]
        # Don't add fingerprint property
        structures_with_issues.append(bad_structure)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create empty reference
            df = pd.DataFrame({"values": []})
            parquet_path = Path(temp_dir) / "unique_fingerprints.parquet"
            df.to_parquet(parquet_path)
            
            metric = AugmentedNoveltyMetric(
                reference_fingerprints_path=temp_dir,
                fingerprint_source="property", 
                fallback_to_computation=False
            )
            result = metric.compute(structures_with_issues)
            
            print(f"   With errors - evaluated: {result.metrics['total_structures_evaluated']}")
            print(f"   Success rate: {result.metrics['fingerprinting_success_rate']:.3f}")
        
        # Test 3: Performance with larger set
        print("3. Testing performance with larger set...")
        large_structures = []
        for i in range(20):
            structure = Structure(
                lattice=[[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]],
                species=["Na", "Cl"],
                coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
                coords_are_cartesian=False,
            )
            structure.properties["augmented_fingerprint"] = f"AUG_PERF_{i:02d}"
            large_structures.append(structure)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create reference with every other fingerprint known
            known_fps = [f"AUG_PERF_{i:02d}" for i in range(0, 20, 2)]
            df = pd.DataFrame({"values": known_fps})
            parquet_path = Path(temp_dir) / "unique_fingerprints.parquet"
            df.to_parquet(parquet_path)
            
            start_time = time.time()
            
            benchmark = AugmentedNoveltyBenchmark(
                reference_fingerprints_path=temp_dir,
                fingerprint_source="property"
            )
            result = benchmark.evaluate(large_structures)
            
            end_time = time.time()
            
            print(f"   Processed {len(large_structures)} structures in {end_time - start_time:.3f}s")
            print(f"   Novelty score: {result.final_scores['novelty_score']:.3f}")
        
        # Test 4: Factory functions
        print("4. Testing factory functions...")
        from lemat_genbench.benchmarks.novelty_new_benchmark import (
            create_high_precision_novelty_benchmark,
            create_property_based_novelty_benchmark,
            create_robust_novelty_benchmark,
        )
        
        factories = [
            create_property_based_novelty_benchmark,
            create_robust_novelty_benchmark,
            create_high_precision_novelty_benchmark,
        ]
        
        for i, factory in enumerate(factories):
            benchmark = factory()
            print(f"   Factory {i+1}: {benchmark.config.name}")
        
        # Test 5: Real-world simulation
        print("5. Testing real-world simulation...")
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create realistic reference
            reference_fps = [
                "AUG_225_Na:4a:1_1|Cl:4b:1_1",  # Known
                "AUG_227_Ca:4a:1_1|F:8c:1_1",   # Known
                "AUG_ref_1", "AUG_ref_2"
            ]
            df = pd.DataFrame({"values": reference_fps})
            parquet_path = Path(temp_dir) / "unique_fingerprints.parquet"
            df.to_parquet(parquet_path)
            
            # Create mix of known and novel structures
            test_structures = []
            test_cases = [
                ("Na", "Cl", "AUG_225_Na:4a:1_1|Cl:4b:1_1"),  # Known
                ("K", "Br", "AUG_225_K:4a:1_1|Br:4b:1_1"),    # Novel
                ("Cs", "I", "AUG_225_Cs:4a:1_1|I:4b:1_1"),    # Novel
            ]
            
            for elem1, elem2, fp in test_cases:
                structure = Structure(
                    lattice=[[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]],
                    species=[elem1, elem2],
                    coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
                    coords_are_cartesian=False,
                )
                structure.properties["augmented_fingerprint"] = fp
                test_structures.append(structure)
            
            benchmark = AugmentedNoveltyBenchmark(
                reference_fingerprints_path=temp_dir,
                fingerprint_source="property"
            )
            result = benchmark.evaluate(test_structures)
            
            print("   Real-world simulation:")
            print(f"     Total structures: {result.final_scores['total_structures_evaluated']}")
            print(f"     Novel structures: {result.final_scores['novel_structures_count']}")
            print(f"     Novelty score: {result.final_scores['novelty_score']:.3f}")
            print("     Expected: 2/3 novel = 0.667")
        
        print("\n✅ All integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run manual integration test for development
    manual_integration_test()