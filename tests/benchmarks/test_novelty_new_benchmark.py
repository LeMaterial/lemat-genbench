"""Tests for the enhanced novelty benchmark using augmented fingerprinting."""

import math
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
from pymatgen.core.structure import Structure

from lemat_genbench.benchmarks.novelty_new_benchmark import (
    AugmentedNoveltyBenchmark,
    create_augmented_novelty_benchmark,
    create_computation_based_novelty_benchmark,
    create_high_precision_novelty_benchmark,
    create_property_based_novelty_benchmark,
    create_robust_novelty_benchmark,
)
from lemat_genbench.metrics.novelty_new_metric import AugmentedNoveltyMetric


def create_test_structures_with_augmented_fingerprints():
    """Create test structures with augmented fingerprints for benchmarking."""
    structures = []

    # Structure 1: Novel NaCl structure
    lattice1 = [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]]
    structure1 = Structure(
        lattice=lattice1,
        species=["Na", "Cl"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structure1.properties["augmented_fingerprint"] = "AUG_225_Na:4a:1_1|Cl:4b:1_1_novel"
    structures.append(structure1)

    # Structure 2: Known CsCl structure (will be in reference)
    lattice2 = [[4.2, 0, 0], [0, 4.2, 0], [0, 0, 4.2]]
    structure2 = Structure(
        lattice=lattice2,
        species=["Cs", "Cl"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structure2.properties["augmented_fingerprint"] = "AUG_221_Cs:4a:1_1|Cl:4b:1_1_known"
    structures.append(structure2)

    # Structure 3: Novel complex structure
    lattice3 = [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]]
    structure3 = Structure(
        lattice=lattice3,
        species=["Ca", "F", "F"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
        coords_are_cartesian=False,
    )
    structure3.properties["augmented_fingerprint"] = "AUG_227_Ca:4a:1_1|F:8c:1_2_novel"
    structures.append(structure3)

    return structures


class TestAugmentedNoveltyBenchmark:
    """Test suite for AugmentedNoveltyBenchmark class."""

    def test_initialization_default(self):
        """Test benchmark initialization with default parameters."""
        benchmark = AugmentedNoveltyBenchmark()
        
        # Check basic properties
        assert benchmark.config.name == "AugmentedNoveltyBenchmark"
        assert "Enhanced novelty benchmark" in benchmark.config.description
        assert benchmark.config.metadata["version"] == "0.2.0"
        assert benchmark.config.metadata["category"] == "novelty"
        assert benchmark.config.metadata["fingerprinting_method"] == "augmented"
        
        # Check evaluator setup
        assert len(benchmark.evaluators) == 1
        assert "augmented_novelty" in benchmark.evaluators
        
        # Check metric configuration
        evaluator = benchmark.evaluators["augmented_novelty"]
        assert "augmented_novelty" in evaluator.metrics
        assert isinstance(evaluator.metrics["augmented_novelty"], AugmentedNoveltyMetric)

    def test_initialization_custom(self):
        """Test benchmark initialization with custom parameters."""
        benchmark = AugmentedNoveltyBenchmark(
            reference_fingerprints_path="/custom/path",
            reference_dataset_name="CustomDataset",
            fingerprint_source="property",
            symprec=0.1,
            angle_tolerance=10.0,
            fallback_to_computation=False,
            name="CustomBenchmark",
            description="Custom description",
            metadata={"test_key": "test_value"},
        )
        
        # Check custom configuration
        assert benchmark.config.name == "CustomBenchmark"
        assert benchmark.config.description == "Custom description"
        assert benchmark.config.metadata["test_key"] == "test_value"
        assert benchmark.config.metadata["reference_dataset"] == "CustomDataset"
        assert benchmark.config.metadata["reference_fingerprints_path"] == "/custom/path"
        assert benchmark.config.metadata["fingerprint_source"] == "property"
        assert benchmark.config.metadata["symprec"] == 0.1
        assert benchmark.config.metadata["angle_tolerance"] == 10.0
        assert benchmark.config.metadata["fallback_to_computation"] is False

    def test_evaluate_structures_with_temp_data(self):
        """Test benchmark evaluation with structures using temporary data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock reference fingerprints
            reference_fps = [
                "AUG_221_Cs:4a:1_1|Cl:4b:1_1_known",  # This matches structure 2
                "AUG_other_ref_1",
                "AUG_other_ref_2",
            ]
            df = pd.DataFrame({"values": reference_fps})
            parquet_path = Path(temp_dir) / "unique_fingerprints.parquet"
            df.to_parquet(parquet_path)
            
            # Create benchmark and test structures
            benchmark = AugmentedNoveltyBenchmark(
                reference_fingerprints_path=temp_dir,
                fingerprint_source="property"
            )
            structures = create_test_structures_with_augmented_fingerprints()
            
            # Run evaluation
            result = benchmark.evaluate(structures)
            
            # Check result structure
            assert len(result.evaluator_results) == 1
            assert "augmented_novelty" in result.evaluator_results
            
            # Check final scores
            final_scores = result.final_scores
            assert "novelty_score" in final_scores
            assert "novelty_ratio" in final_scores
            assert "novel_structures_count" in final_scores
            assert "total_structures_evaluated" in final_scores
            assert "total_structures_attempted" in final_scores
            assert "fingerprinting_success_rate" in final_scores
            
            # Check score values (2 novel out of 3 total)
            assert final_scores["novelty_score"] == 2.0 / 3.0
            assert final_scores["novelty_ratio"] == 2.0 / 3.0
            assert final_scores["novel_structures_count"] == 2
            assert final_scores["total_structures_evaluated"] == 3
            assert final_scores["fingerprinting_success_rate"] == 1.0

    def test_empty_structures_evaluation(self):
        """Test benchmark behavior with empty structure list."""
        benchmark = AugmentedNoveltyBenchmark()
        
        # Evaluate empty list
        result = benchmark.evaluate([])
        
        # Should handle gracefully
        final_scores = result.final_scores
        assert math.isnan(final_scores["novelty_score"])
        assert final_scores["novel_structures_count"] == 0
        assert final_scores["total_structures_evaluated"] == 0
        assert final_scores["total_structures_attempted"] == 0
        assert final_scores["fingerprinting_success_rate"] == 0.0

    def test_aggregate_evaluator_results(self):
        """Test result aggregation from evaluators."""
        benchmark = AugmentedNoveltyBenchmark()
        
        # Mock evaluator results
        mock_evaluator_results = {
            "augmented_novelty": {
                "combined_value": 0.6,
                "metric_results": {
                    "augmented_novelty": {
                        "metrics": {
                            "novelty_score": 0.6,
                            "novel_structures_count": 6,
                            "total_structures_evaluated": 10,
                            "total_structures_attempted": 12,
                            "fingerprinting_success_rate": 0.833,
                        }
                    }
                },
            }
        }
        
        # Aggregate results
        scores = benchmark.aggregate_evaluator_results(mock_evaluator_results)
        
        # Check aggregated scores
        assert scores["novelty_score"] == 0.6
        assert scores["novelty_ratio"] == 0.6
        assert scores["novel_structures_count"] == 6
        assert scores["total_structures_evaluated"] == 10
        assert scores["total_structures_attempted"] == 12
        assert scores["fingerprinting_success_rate"] == 0.833

    def test_aggregate_evaluator_results_with_metric_result_object(self):
        """Test aggregation with actual MetricResult objects."""
        benchmark = AugmentedNoveltyBenchmark()
        
        # Create mock MetricResult
        mock_metric_result = Mock()
        mock_metric_result.metrics = {
            "novelty_score": 0.8,
            "novel_structures_count": 8,
            "total_structures_evaluated": 10,
            "total_structures_attempted": 10,
            "fingerprinting_success_rate": 1.0,
        }
        
        mock_evaluator_results = {
            "augmented_novelty": {
                "combined_value": 0.8,
                "metric_results": {"augmented_novelty": mock_metric_result},
            }
        }
        
        scores = benchmark.aggregate_evaluator_results(mock_evaluator_results)
        
        assert scores["novelty_score"] == 0.8
        assert scores["novelty_ratio"] == 0.8
        assert scores["novel_structures_count"] == 8
        assert scores["total_structures_evaluated"] == 10
        assert scores["fingerprinting_success_rate"] == 1.0

    def test_aggregate_evaluator_results_missing_data(self):
        """Test aggregation with missing or malformed data."""
        benchmark = AugmentedNoveltyBenchmark()
        
        # Test with empty results
        empty_results = {}
        scores = benchmark.aggregate_evaluator_results(empty_results)
        
        assert math.isnan(scores["novelty_score"])
        assert scores["novel_structures_count"] == 0
        assert scores["total_structures_evaluated"] == 0
        assert scores["fingerprinting_success_rate"] == 0.0
        
        # Test with malformed results
        malformed_results = {
            "augmented_novelty": {
                "combined_value": None,
                "metric_results": {},
            }
        }
        scores = benchmark.aggregate_evaluator_results(malformed_results)
        
        assert math.isnan(scores["novelty_score"])
        assert scores["novel_structures_count"] == 0

    def test_benchmark_metadata_structure(self):
        """Test benchmark metadata contains expected information."""
        benchmark = AugmentedNoveltyBenchmark(
            reference_fingerprints_path="/test/path",
            reference_dataset_name="TestDataset",
            fingerprint_source="auto",
            symprec=0.05,
            angle_tolerance=8.0,
        )
        
        metadata = benchmark.config.metadata
        
        # Check required metadata fields
        assert metadata["version"] == "0.2.0"
        assert metadata["category"] == "novelty"
        assert metadata["fingerprinting_method"] == "augmented"
        assert metadata["reference_dataset"] == "TestDataset"
        assert metadata["reference_fingerprints_path"] == "/test/path"
        assert metadata["fingerprint_source"] == "auto"
        assert metadata["symprec"] == 0.05
        assert metadata["angle_tolerance"] == 8.0
        assert metadata["fallback_to_computation"] is True

    def test_evaluator_configuration(self):
        """Test that evaluator is properly configured."""
        benchmark = AugmentedNoveltyBenchmark()
        
        # Check evaluator exists and is configured correctly
        evaluator = benchmark.evaluators["augmented_novelty"]
        assert evaluator.config.name == "augmented_novelty"
        assert evaluator.config.weights == {"augmented_novelty": 1.0}
        assert evaluator.config.aggregation_method == "weighted_mean"
        
        # Check metric configuration
        metric = evaluator.metrics["augmented_novelty"]
        assert isinstance(metric, AugmentedNoveltyMetric)
        assert metric.name == "AugmentedNovelty"

    def test_factory_functions(self):
        """Test factory functions for creating benchmarks."""
        # Test basic factory
        benchmark1 = create_augmented_novelty_benchmark()
        assert isinstance(benchmark1, AugmentedNoveltyBenchmark)
        assert benchmark1.config.metadata["fingerprint_source"] == "auto"
        
        # Test property-based factory
        benchmark2 = create_property_based_novelty_benchmark()
        assert benchmark2.config.name == "PropertyBasedNoveltyBenchmark"
        assert benchmark2.config.metadata["fingerprint_source"] == "property"
        assert benchmark2.config.metadata["fallback_to_computation"] is False
        
        # Test computation-based factory
        benchmark3 = create_computation_based_novelty_benchmark()
        assert benchmark3.config.name == "ComputationBasedNoveltyBenchmark"
        assert benchmark3.config.metadata["fingerprint_source"] == "compute"
        
        # Test robust factory
        benchmark4 = create_robust_novelty_benchmark()
        assert benchmark4.config.name == "RobustNoveltyBenchmark"
        assert benchmark4.config.metadata["symprec"] == 0.1
        assert benchmark4.config.metadata["angle_tolerance"] == 10.0
        
        # Test high precision factory
        benchmark5 = create_high_precision_novelty_benchmark()
        assert benchmark5.config.name == "HighPrecisionNoveltyBenchmark"
        assert benchmark5.config.metadata["symprec"] == 0.001
        assert benchmark5.config.metadata["angle_tolerance"] == 1.0

    def test_benchmark_with_fingerprinting_failures(self):
        """Test benchmark handling of fingerprinting failures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create reference data
            reference_fps = ["AUG_ref1", "AUG_ref2"]
            df = pd.DataFrame({"values": reference_fps})
            parquet_path = Path(temp_dir) / "unique_fingerprints.parquet"
            df.to_parquet(parquet_path)
            
            # Create structures with some missing fingerprints
            structures = []
            
            # Structure 1: Has fingerprint
            lattice1 = [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]]
            structure1 = Structure(
                lattice=lattice1,
                species=["Na", "Cl"],
                coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
                coords_are_cartesian=False,
            )
            structure1.properties["augmented_fingerprint"] = "AUG_novel_1"
            structures.append(structure1)
            
            # Structure 2: No fingerprint (will fail in property mode)
            lattice2 = [[4.1, 0, 0], [0, 4.1, 0], [0, 0, 4.1]]
            structure2 = Structure(
                lattice=lattice2,
                species=["K", "Br"],
                coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
                coords_are_cartesian=False,
            )
            # No fingerprint property
            structures.append(structure2)
            
            # Test with property-only mode (no computation fallback)
            benchmark = AugmentedNoveltyBenchmark(
                reference_fingerprints_path=temp_dir,
                fingerprint_source="property",
                fallback_to_computation=False
            )
            
            result = benchmark.evaluate(structures)
            
            # Should handle failure gracefully
            final_scores = result.final_scores
            assert final_scores["total_structures_attempted"] == 2
            assert final_scores["total_structures_evaluated"] == 1  # Only first succeeded
            assert final_scores["fingerprinting_success_rate"] == 0.5  # 1/2

    def test_different_fingerprint_sources(self):
        """Test benchmark with different fingerprint sources."""
        sources = ["property", "compute", "auto"]
        
        for source in sources:
            benchmark = AugmentedNoveltyBenchmark(fingerprint_source=source)
            
            # Check that the source is correctly set in metadata
            assert benchmark.config.metadata["fingerprint_source"] == source
            
            # Check that the metric is configured correctly
            evaluator = benchmark.evaluators["augmented_novelty"]
            metric = evaluator.metrics["augmented_novelty"]
            assert metric.config.fingerprint_source == source

    def test_custom_tolerances(self):
        """Test benchmark with custom symmetry tolerances."""
        benchmark = AugmentedNoveltyBenchmark(
            symprec=0.001,
            angle_tolerance=1.0,
        )
        
        # Check metadata
        assert benchmark.config.metadata["symprec"] == 0.001
        assert benchmark.config.metadata["angle_tolerance"] == 1.0
        
        # Check metric configuration
        evaluator = benchmark.evaluators["augmented_novelty"]
        metric = evaluator.metrics["augmented_novelty"]
        assert metric.config.symprec == 0.001
        assert metric.config.angle_tolerance == 1.0

    def test_realistic_workflow(self):
        """Test realistic workflow with simulated reference data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create realistic reference fingerprints
            reference_fps = [
                "AUG_221_Cs:4a:1_1|Cl:4b:1_1_known",  # This matches structure 2
                "AUG_225_Na:4a:1_1|Cl:4b:1_2",
                "AUG_227_Ca:4a:2_1|F:8c:1_1",
            ]
            df = pd.DataFrame({"values": reference_fps})
            parquet_path = Path(temp_dir) / "unique_fingerprints.parquet"
            df.to_parquet(parquet_path)
            
            # Create benchmark
            benchmark = AugmentedNoveltyBenchmark(
                reference_fingerprints_path=temp_dir,
                fingerprint_source="property"
            )
            structures = create_test_structures_with_augmented_fingerprints()
            
            # Run evaluation
            result = benchmark.evaluate(structures)
            
            # Should complete without error
            assert isinstance(result.final_scores, dict)
            assert "novelty_score" in result.final_scores
            assert "fingerprinting_success_rate" in result.final_scores
            
            # With 3 structures, 1 known (structure 2), 2 novel (structures 1, 3)
            assert result.final_scores["total_structures_evaluated"] == 3
            assert result.final_scores["novel_structures_count"] == 2
            assert result.final_scores["novelty_score"] == 2.0 / 3.0

    def test_error_handling_in_evaluation(self):
        """Test error handling during evaluation."""
        benchmark = AugmentedNoveltyBenchmark()
        
        # Test with structures that might cause issues
        problematic_structures = []
        
        # Structure with no fingerprint and computation disabled
        lattice = [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]]
        structure = Structure(
            lattice=lattice,
            species=["Na", "Cl"],
            coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
            coords_are_cartesian=False,
        )
        # No fingerprint property added
        problematic_structures.append(structure)
        
        # Should handle gracefully without crashing
        try:
            result = benchmark.evaluate(problematic_structures)
            assert isinstance(result.final_scores, dict)
            # Should show that evaluation was attempted but possibly failed
            assert "total_structures_attempted" in result.final_scores
        except Exception as e:
            # If it fails, it should be a controlled failure
            assert "fingerprint" in str(e).lower() or "reference" in str(e).lower()

    def test_benchmark_comparison_scenarios(self):
        """Test benchmark in various comparison scenarios."""
        with tempfile.TemporaryDirectory() as temp_dir:
            structures = create_test_structures_with_augmented_fingerprints()
            
            # Scenario 1: All novel structures
            reference_fps = ["AUG_different_1", "AUG_different_2"]
            df = pd.DataFrame({"values": reference_fps})
            parquet_path = Path(temp_dir) / "unique_fingerprints.parquet"
            df.to_parquet(parquet_path)
            
            benchmark = AugmentedNoveltyBenchmark(
                reference_fingerprints_path=temp_dir,
                fingerprint_source="property"
            )
            result = benchmark.evaluate(structures)
            assert result.final_scores["novelty_score"] == 1.0  # All novel
            
        # Use a new temp directory for scenario 2 to avoid caching issues
        with tempfile.TemporaryDirectory() as temp_dir2:
            # Scenario 2: All known structures
            all_known_fps = [
                "AUG_225_Na:4a:1_1|Cl:4b:1_1_novel",
                "AUG_221_Cs:4a:1_1|Cl:4b:1_1_known",
                "AUG_227_Ca:4a:1_1|F:8c:1_2_novel",
            ]
            df_known = pd.DataFrame({"values": all_known_fps})
            parquet_path2 = Path(temp_dir2) / "unique_fingerprints.parquet"
            df_known.to_parquet(parquet_path2)
            
            # Create new benchmark instance to avoid caching issues
            benchmark_known = AugmentedNoveltyBenchmark(
                reference_fingerprints_path=temp_dir2,
                fingerprint_source="property"
            )
            result_known = benchmark_known.evaluate(structures)
            assert result_known.final_scores["novelty_score"] == 0.0  # All known


class TestIntegrationScenarios:
    """Integration tests for realistic usage scenarios."""
    
    def test_preprocessing_integration_simulation(self):
        """Test integration with preprocessing pipeline simulation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create reference data
            reference_fps = ["AUG_225_Na:4a:1_1|Cl:4b:1_1", "AUG_ref2"]
            df = pd.DataFrame({"values": reference_fps})
            parquet_path = Path(temp_dir) / "unique_fingerprints.parquet"
            df.to_parquet(parquet_path)
            
            benchmark = AugmentedNoveltyBenchmark(
                reference_fingerprints_path=temp_dir,
                fingerprint_source="property"
            )
            
            # Create structures as if they came from preprocessor
            structures = []
            lattice = [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]]
            
            for i, (elements, fp) in enumerate([
                (["Na", "Cl"], "AUG_225_Na:4a:1_1|Cl:4b:1_1"),  # Known
                (["K", "Br"], "AUG_225_K:4a:1_1|Br:4b:1_1"),    # Novel
                (["Cs", "I"], "AUG_225_Cs:4a:1_1|I:4b:1_1"),    # Novel
            ]):
                structure = Structure(
                    lattice=lattice,
                    species=elements,
                    coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
                    coords_are_cartesian=False,
                )
                
                # Add properties as the preprocessor would
                structure.properties.update({
                    "augmented_fingerprint": fp,
                    "augmented_fingerprint_properties": {
                        "augmented_fingerprint": fp,
                        "augmented_fingerprint_success": True,
                        "augmented_fingerprint_spacegroup": 225,
                        "augmented_fingerprint_elements": elements,
                    }
                })
                structures.append(structure)
            
            result = benchmark.evaluate(structures)
            
            # Should identify 2 out of 3 as novel
            assert result.final_scores["novelty_score"] == 2.0 / 3.0
            assert result.final_scores["novel_structures_count"] == 2
            assert result.final_scores["fingerprinting_success_rate"] == 1.0

    def test_batch_processing_simulation(self):
        """Test benchmark behavior with larger batches of structures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create reference with some known fingerprints
            known_fps = [f"AUG_225_batch_test_{i}" for i in range(0, 20, 2)]  # Every other one
            df = pd.DataFrame({"values": known_fps})
            parquet_path = Path(temp_dir) / "unique_fingerprints.parquet"
            df.to_parquet(parquet_path)
            
            benchmark = AugmentedNoveltyBenchmark(
                reference_fingerprints_path=temp_dir,
                fingerprint_source="property"
            )
            
            # Create a larger set of structures
            structures = []
            lattice = [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]]
            
            # Generate 20 structures with varying fingerprints
            for i in range(20):
                structure = Structure(
                    lattice=lattice,
                    species=["Na", "Cl"],
                    coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
                    coords_are_cartesian=False,
                )
                structure.properties["augmented_fingerprint"] = f"AUG_225_batch_test_{i}"
                structures.append(structure)
            
            result = benchmark.evaluate(structures)
            
            # Should process all structures
            assert result.final_scores["total_structures_evaluated"] == 20
            assert result.final_scores["fingerprinting_success_rate"] == 1.0
            
            # Novel count should be 20 - 10 = 10 (since 10 are known: 0,2,4,6,8,10,12,14,16,18)
            assert result.final_scores["novel_structures_count"] == 10
            assert result.final_scores["novelty_score"] == 0.5


# Manual test function for development
def manual_test_augmented_novelty_benchmark():
    """Manual test function for development and debugging."""
    print("🧪 Running manual augmented novelty benchmark test...")
    
    try:
        # Test 1: Basic functionality
        print("1. Testing basic benchmark functionality...")
        _ = AugmentedNoveltyBenchmark()
        print("✅ Benchmark initialized successfully")
        
        # Test 2: Structure evaluation with temporary data
        print("2. Testing structure evaluation...")
        structures = create_test_structures_with_augmented_fingerprints()
        print(f"   Created {len(structures)} test structures")
        
        for i, structure in enumerate(structures):
            fp = structure.properties.get("augmented_fingerprint", "No fingerprint")
            print(f"   Structure {i+1}: {fp}")
        
        # Test 3: Evaluation with temporary reference data
        print("3. Testing with temporary reference data...")
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_reference = [
                "AUG_221_Cs:4a:1_1|Cl:4b:1_1_known",  # Match structure 2
                "AUG_other_1",
                "AUG_other_2",
            ]
            df = pd.DataFrame({"values": mock_reference})
            parquet_path = Path(temp_dir) / "unique_fingerprints.parquet"
            df.to_parquet(parquet_path)
            
            benchmark_with_ref = AugmentedNoveltyBenchmark(
                reference_fingerprints_path=temp_dir,
                fingerprint_source="property"
            )
            result = benchmark_with_ref.evaluate(structures)
            
            print(f"   Novelty score: {result.final_scores['novelty_score']:.3f}")
            print(f"   Novel structures: {result.final_scores['novel_structures_count']}")
            print(f"   Total evaluated: {result.final_scores['total_structures_evaluated']}")
            print(f"   Success rate: {result.final_scores['fingerprinting_success_rate']:.3f}")
        
        # Test 4: Factory functions
        print("4. Testing factory functions...")
        benchmarks = [
            create_property_based_novelty_benchmark(),
            create_computation_based_novelty_benchmark(),
            create_robust_novelty_benchmark(),
            create_high_precision_novelty_benchmark(),
        ]
        
        for i, bm in enumerate(benchmarks):
            print(f"   Factory {i+1}: {bm.config.name}")
        
        print("\n✅ All manual tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run manual test for development
    manual_test_augmented_novelty_benchmark()