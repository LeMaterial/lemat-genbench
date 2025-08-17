"""Tests for new SUN (Stable, Unique, Novel) benchmark using augmented fingerprinting."""

import math
from unittest.mock import MagicMock, Mock, patch

from pymatgen.core.structure import Structure

from lemat_genbench.benchmarks.sun_new_benchmark import SUNNewBenchmark
from lemat_genbench.metrics.sun_new_metric import MetaSUNNewMetric, SUNNewMetric


def create_mock_uniqueness_result(structures, individual_values, failed_indices=None):
    """Helper function to create properly mocked uniqueness results with fingerprints."""
    if failed_indices is None:
        failed_indices = []
    
    mock_result = MagicMock()
    mock_result.individual_values = individual_values
    mock_result.failed_indices = failed_indices
    
    # Add fingerprints attribute that the implementation expects
    # For testing, we'll create unique fingerprints for unique structures
    # and identical fingerprints for duplicate structures
    fingerprints = []
    for i, val in enumerate(individual_values):
        if i not in failed_indices:
            if val == 1.0:
                # Unique structure gets unique fingerprint
                fingerprints.append(f"unique_fp_{i}")
            else:
                # Duplicate structure gets shared fingerprint
                # Use the reciprocal to identify groups (e.g., val=0.5 means 2 duplicates)
                group_size = int(round(1.0 / val)) if val > 0 else 1
                fingerprints.append(f"dup_fp_group{group_size}")
    
    mock_result.fingerprints = fingerprints
    return mock_result


def create_test_structures():
    """Create simple test structures for benchmarking."""
    structures = []

    # Structure 1: Simple cubic NaCl
    lattice = [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]]
    structure1 = Structure(
        lattice=lattice,
        species=["Na", "Cl"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structure1.properties = {"e_above_hull": 0.0}  # Stable
    structures.append(structure1)

    # Structure 2: Simple cubic KBr
    lattice = [[4.5, 0, 0], [0, 4.5, 0], [0, 0, 4.5]]
    structure2 = Structure(
        lattice=lattice,
        species=["K", "Br"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structure2.properties = {"e_above_hull": 0.08}  # Metastable
    structures.append(structure2)

    # Structure 3: Simple cubic LiF
    lattice = [[3.5, 0, 0], [0, 3.5, 0], [0, 0, 3.5]]
    structure3 = Structure(
        lattice=lattice,
        species=["Li", "F"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structure3.properties = {"e_above_hull": 0.2}  # Unstable
    structures.append(structure3)

    return structures


def create_test_structures_with_multi_mlip():
    """Create test structures with multi-MLIP properties."""
    structures = []

    # Structure 1: Stable with ensemble properties
    lattice = [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]]
    structure1 = Structure(
        lattice=lattice,
        species=["Na", "Cl"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structure1.properties = {
        "e_above_hull_mean": 0.0,  # Stable
        "e_above_hull_std": 0.02,
        "e_above_hull_orb": -0.01,
        "e_above_hull_mace": 0.01,
        "e_above_hull_uma": 0.0,
    }
    structures.append(structure1)

    # Structure 2: Metastable with ensemble properties
    lattice = [[4.5, 0, 0], [0, 4.5, 0], [0, 0, 4.5]]
    structure2 = Structure(
        lattice=lattice,
        species=["K", "Br"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structure2.properties = {
        "e_above_hull_mean": 0.08,  # Metastable
        "e_above_hull_std": 0.03,
        "e_above_hull_orb": 0.05,
        "e_above_hull_mace": 0.11,
        "e_above_hull_uma": 0.08,
    }
    structures.append(structure2)

    return structures


class TestSUNNewBenchmark:
    """Test suite for SUNNewBenchmark class."""

    def test_initialization(self):
        """Test initialization with default parameters."""
        benchmark = SUNNewBenchmark()

        # Check name and properties
        assert benchmark.config.name == "SUNNewBenchmark"
        assert "version" in benchmark.config.metadata
        assert benchmark.config.metadata["category"] == "sun"

        # Check correct evaluators (should have both SUN and MetaSUN by default)
        assert len(benchmark.evaluators) == 2
        assert "sun" in benchmark.evaluators
        assert "metasun" in benchmark.evaluators

        # Check metadata
        metadata = benchmark.config.metadata
        assert metadata["stability_threshold"] == 0.0
        assert metadata["metastability_threshold"] == 0.1
        assert metadata["reference_dataset_name"] == "LeMat-Bulk"
        assert metadata["fingerprinting_method"] == "augmented"
        assert metadata["fingerprint_source"] == "auto"
        assert metadata["include_metasun"] is True

    def test_initialization_without_metasun(self):
        """Test initialization without MetaSUN evaluator."""
        benchmark = SUNNewBenchmark(include_metasun=False)

        # Should only have SUN evaluator
        assert len(benchmark.evaluators) == 1
        assert "sun" in benchmark.evaluators
        assert "metasun" not in benchmark.evaluators

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        benchmark = SUNNewBenchmark(
            stability_threshold=0.05,
            metastability_threshold=0.15,
            reference_fingerprints_path="custom/path",
            reference_dataset_name="custom_dataset",
            fingerprint_source="property",
            symprec=0.1,
            angle_tolerance=10.0,
            fallback_to_computation=False,
            include_metasun=False,
            name="Custom SUN New Benchmark",
            description="Custom description",
            metadata={"test_key": "test_value"},
        )

        # Check custom values
        assert benchmark.config.name == "Custom SUN New Benchmark"
        assert benchmark.config.description == "Custom description"
        assert benchmark.config.metadata["test_key"] == "test_value"
        assert benchmark.config.metadata["stability_threshold"] == 0.05
        assert benchmark.config.metadata["metastability_threshold"] == 0.15
        assert benchmark.config.metadata["reference_fingerprints_path"] == "custom/path"
        assert benchmark.config.metadata["fingerprint_source"] == "property"
        assert benchmark.config.metadata["include_metasun"] is False

        # Should only have SUN evaluator
        assert len(benchmark.evaluators) == 1

    def test_evaluator_configuration(self):
        """Test that evaluators are properly configured."""
        benchmark = SUNNewBenchmark()

        # Check SUN evaluator
        sun_evaluator = benchmark.evaluators["sun"]
        assert sun_evaluator.config.name == "sun"  # Name is set to the key in the dict
        assert "sun" in sun_evaluator.metrics
        assert isinstance(sun_evaluator.metrics["sun"], SUNNewMetric)

        # Check MetaSUN evaluator
        metasun_evaluator = benchmark.evaluators["metasun"]
        assert (
            metasun_evaluator.config.name == "metasun"
        )  # Name is set to the key in the dict
        assert "metasun" in metasun_evaluator.metrics
        assert isinstance(metasun_evaluator.metrics["metasun"], MetaSUNNewMetric)

    @patch("lemat_genbench.metrics.novelty_new_metric.AugmentedNoveltyMetric")
    @patch("lemat_genbench.metrics.uniqueness_new_metric.UniquenessNewMetric")
    def test_aggregate_evaluator_results_basic(
        self, mock_uniqueness_class, mock_novelty_class
    ):
        """Test basic result aggregation."""
        benchmark = SUNNewBenchmark(include_metasun=False)

        # Mock the SUN metric results
        mock_sun_metric_result = Mock()
        mock_sun_metric_result.metrics = {
            "sun_rate": 0.25,
            "sun_count": 1,
            "msun_rate": 0.5,
            "msun_count": 2,
            "combined_sun_msun_rate": 0.75,
            "unique_count": 3,
            "unique_rate": 1.0,
            "total_structures_evaluated": 4,
            "failed_count": 0,
        }

        mock_evaluator_result = {
            "combined_value": 0.25,
            "metric_results": {"sun": mock_sun_metric_result},
        }

        evaluator_results = {"sun": mock_evaluator_result}

        final_scores = benchmark.aggregate_evaluator_results(evaluator_results)

        # Check that all metrics are properly extracted
        assert final_scores["sun_rate"] == 0.25
        assert final_scores["sun_count"] == 1
        assert final_scores["msun_rate"] == 0.5
        assert final_scores["msun_count"] == 2
        assert final_scores["combined_sun_msun_rate"] == 0.75
        assert final_scores["unique_count"] == 3
        assert final_scores["unique_rate"] == 1.0
        assert final_scores["total_structures_evaluated"] == 4
        assert final_scores["failed_count"] == 0

    @patch("lemat_genbench.metrics.novelty_new_metric.AugmentedNoveltyMetric")
    @patch("lemat_genbench.metrics.uniqueness_new_metric.UniquenessNewMetric")
    def test_aggregate_evaluator_results_with_metasun(
        self, mock_uniqueness_class, mock_novelty_class
    ):
        """Test result aggregation with both SUN and MetaSUN evaluators."""
        benchmark = SUNNewBenchmark(include_metasun=True)

        # Mock SUN metric results
        mock_sun_metric_result = Mock()
        mock_sun_metric_result.metrics = {
            "sun_rate": 0.25,
            "sun_count": 1,
            "msun_rate": 0.25,
            "msun_count": 1,
            "total_structures_evaluated": 4,
        }

        mock_sun_evaluator_result = {
            "combined_value": 0.25,
            "metric_results": {"sun": mock_sun_metric_result},
        }

        # Mock MetaSUN metric results
        mock_metasun_metric_result = Mock()
        mock_metasun_metric_result.metrics = {
            "sun_rate": 0.5,  # MetaSUN includes both stable and metastable
            "sun_count": 2,
            "total_structures_evaluated": 4,
        }

        mock_metasun_evaluator_result = {
            "combined_value": 0.5,
            "metric_results": {"metasun": mock_metasun_metric_result},
        }

        evaluator_results = {
            "sun": mock_sun_evaluator_result,
            "metasun": mock_metasun_evaluator_result,
        }

        final_scores = benchmark.aggregate_evaluator_results(evaluator_results)

        # Check SUN metrics
        assert final_scores["sun_rate"] == 0.25
        assert final_scores["sun_count"] == 1

        # Check MetaSUN metrics
        assert final_scores["metasun_rate"] == 0.5
        assert final_scores["metasun_count"] == 2

    def test_aggregate_evaluator_results_empty(self):
        """Test result aggregation with empty results."""
        benchmark = SUNNewBenchmark()

        final_scores = benchmark.aggregate_evaluator_results({})

        # Should return default NaN values
        assert math.isnan(final_scores["sun_rate"])
        assert math.isnan(final_scores["msun_rate"])
        assert final_scores["sun_count"] == 0
        assert final_scores["msun_count"] == 0
        assert final_scores["total_structures_evaluated"] == 0

    def test_aggregate_evaluator_results_with_dict_metrics(self):
        """Test result aggregation when metrics are returned as dictionaries."""
        benchmark = SUNNewBenchmark(include_metasun=False)

        # Mock metrics as dictionary instead of object
        mock_evaluator_result = {
            "combined_value": 0.3,
            "metric_results": {
                "sun": {
                    "metrics": {
                        "sun_rate": 0.3,
                        "sun_count": 3,
                        "total_structures_evaluated": 10,
                    }
                }
            },
        }

        evaluator_results = {"sun": mock_evaluator_result}

        final_scores = benchmark.aggregate_evaluator_results(evaluator_results)

        assert final_scores["sun_rate"] == 0.3
        assert final_scores["sun_count"] == 3
        assert final_scores["total_structures_evaluated"] == 10

    @patch("lemat_genbench.metrics.novelty_new_metric.AugmentedNoveltyMetric")
    @patch("lemat_genbench.metrics.uniqueness_new_metric.UniquenessNewMetric")
    def test_full_evaluation_mocked(self, mock_uniqueness_class, mock_novelty_class):
        """Test full benchmark evaluation with mocked sub-metrics."""
        structures = create_test_structures()

        # Mock uniqueness: all structures are unique using helper function
        mock_uniqueness = MagicMock()
        mock_uniqueness_result = create_mock_uniqueness_result(
            structures, [1.0, 1.0, 1.0], []
        )
        mock_uniqueness.compute.return_value = mock_uniqueness_result
        mock_uniqueness_class.return_value = mock_uniqueness

        # Mock novelty: all structures are novel
        mock_novelty = MagicMock()
        mock_novelty_result = MagicMock()
        mock_novelty_result.individual_values = [1.0, 1.0, 1.0]
        mock_novelty_result.failed_indices = []
        mock_novelty.compute.return_value = mock_novelty_result
        mock_novelty_class.return_value = mock_novelty

        benchmark = SUNNewBenchmark(include_metasun=False)

        result = benchmark.evaluate(structures)

        # Check that we got results
        assert result is not None
        assert result.final_scores is not None
        assert "sun_rate" in result.final_scores

        # Check metadata
        assert result.metadata["benchmark_name"] == "SUNNewBenchmark"
        assert result.metadata["n_structures"] == 3

    def test_benchmark_with_invalid_structures(self):
        """Test benchmark behavior with structures missing required properties."""
        benchmark = SUNNewBenchmark(include_metasun=False)

        # Create structures without e_above_hull properties
        structures = []
        lattice = [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]]
        structure = Structure(
            lattice=lattice,
            species=["Na", "Cl"],
            coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
            coords_are_cartesian=False,
        )
        # No properties set
        structures.append(structure)

        # This should not crash, but might return NaN values
        with (
            patch("lemat_genbench.metrics.novelty_new_metric.AugmentedNoveltyMetric"),
            patch("lemat_genbench.metrics.uniqueness_new_metric.UniquenessNewMetric"),
        ):
            result = benchmark.evaluate(structures)
            assert result is not None


class TestSUNNewBenchmarkIntegration:
    """Integration tests for SUNNewBenchmark with real metric components."""

    @patch("lemat_genbench.metrics.novelty_new_metric.AugmentedNoveltyMetric")
    @patch("lemat_genbench.metrics.uniqueness_new_metric.UniquenessNewMetric")
    def test_with_multi_mlip_structures(
        self, mock_uniqueness_class, mock_novelty_class
    ):
        """Test benchmark with multi-MLIP preprocessed structures."""
        structures = create_test_structures_with_multi_mlip()

        # Mock all structures as unique and novel using helper function
        mock_uniqueness_result = create_mock_uniqueness_result(
            structures, [1.0, 1.0], []
        )
        mock_uniqueness_class.return_value.compute.return_value = mock_uniqueness_result

        mock_novelty_result = MagicMock()
        mock_novelty_result.individual_values = [1.0, 1.0]
        mock_novelty_result.failed_indices = []
        mock_novelty_class.return_value.compute.return_value = mock_novelty_result

        benchmark = SUNNewBenchmark()

        result = benchmark.evaluate(structures)

        # Should handle multi-MLIP properties correctly
        assert result is not None
        assert "sun_rate" in result.final_scores
        assert result.metadata["n_structures"] == 2

    @patch("lemat_genbench.metrics.novelty_new_metric.AugmentedNoveltyMetric")
    @patch("lemat_genbench.metrics.uniqueness_new_metric.UniquenessNewMetric")
    def test_benchmark_consistency(self, mock_uniqueness_class, mock_novelty_class):
        """Test that benchmark results are consistent across multiple runs."""
        structures = create_test_structures()

        # Mock deterministic results using helper function
        mock_uniqueness_result = create_mock_uniqueness_result(
            structures, [1.0, 1.0, 1.0], []
        )
        mock_uniqueness_class.return_value.compute.return_value = mock_uniqueness_result

        mock_novelty_result = MagicMock()
        mock_novelty_result.individual_values = [1.0, 1.0, 1.0]
        mock_novelty_result.failed_indices = []
        mock_novelty_class.return_value.compute.return_value = mock_novelty_result

        benchmark = SUNNewBenchmark(include_metasun=False)

        # Run benchmark multiple times
        result1 = benchmark.evaluate(structures)
        result2 = benchmark.evaluate(structures)

        # Results should be identical (deterministic)
        assert result1.final_scores["sun_rate"] == result2.final_scores["sun_rate"]
        assert result1.final_scores["sun_count"] == result2.final_scores["sun_count"]

    def test_benchmark_with_empty_structures(self):
        """Test benchmark behavior with empty structure list."""
        benchmark = SUNNewBenchmark()

        result = benchmark.evaluate([])

        # Should handle empty input gracefully
        assert result is not None
        assert result.metadata["n_structures"] == 0

    @patch("lemat_genbench.metrics.novelty_new_metric.AugmentedNoveltyMetric")
    @patch("lemat_genbench.metrics.uniqueness_new_metric.UniquenessNewMetric")
    def test_benchmark_scaling(self, mock_uniqueness_class, mock_novelty_class):
        """Test benchmark performance with larger structure sets."""
        # Mock results for larger set
        n_structures = 50

        # Create larger set of structures
        structures = []
        for i in range(n_structures):
            lattice = [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]]
            structure = Structure(
                lattice=lattice,
                species=["Na", "Cl"],
                coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
                coords_are_cartesian=False,
            )
            structure.properties = {"e_above_hull": 0.0}  # All stable
            structures.append(structure)

        mock_uniqueness_result = create_mock_uniqueness_result(
            structures, [1.0] * n_structures, []
        )
        mock_uniqueness_class.return_value.compute.return_value = mock_uniqueness_result

        mock_novelty_result = MagicMock()
        mock_novelty_result.individual_values = [1.0] * n_structures
        mock_novelty_result.failed_indices = []
        mock_novelty_class.return_value.compute.return_value = mock_novelty_result

        benchmark = SUNNewBenchmark(include_metasun=False)

        result = benchmark.evaluate(structures)

        # Should handle larger sets efficiently
        assert result is not None
        assert result.metadata["n_structures"] == n_structures


class TestSUNNewBenchmarkFactoryFunctions:
    """Test suite for factory functions."""

    def test_create_property_based_sun_benchmark(self):
        """Test property-based SUN benchmark factory function."""
        from lemat_genbench.benchmarks.sun_new_benchmark import (
            create_property_based_sun_benchmark,
        )
        
        benchmark = create_property_based_sun_benchmark()
        
        assert isinstance(benchmark, SUNNewBenchmark)
        assert benchmark.config.name == "PropertyBasedSUNBenchmark"
        assert benchmark.config.metadata["fingerprint_source"] == "property"
        assert not benchmark.config.metadata["fallback_to_computation"]

    def test_create_computation_based_sun_benchmark(self):
        """Test computation-based SUN benchmark factory function."""
        from lemat_genbench.benchmarks.sun_new_benchmark import (
            create_computation_based_sun_benchmark,
        )
        
        benchmark = create_computation_based_sun_benchmark()
        
        assert isinstance(benchmark, SUNNewBenchmark)
        assert benchmark.config.name == "ComputationBasedSUNBenchmark"
        assert benchmark.config.metadata["fingerprint_source"] == "compute"

    def test_create_robust_sun_benchmark(self):
        """Test robust SUN benchmark factory function."""
        from lemat_genbench.benchmarks.sun_new_benchmark import (
            create_robust_sun_benchmark,
        )
        
        benchmark = create_robust_sun_benchmark()
        
        assert isinstance(benchmark, SUNNewBenchmark)
        assert benchmark.config.name == "RobustSUNBenchmark"
        assert benchmark.config.metadata["fingerprint_source"] == "auto"
        assert benchmark.config.metadata["symprec"] == 0.1
        assert benchmark.config.metadata["angle_tolerance"] == 10.0

    def test_create_high_precision_sun_benchmark(self):
        """Test high precision SUN benchmark factory function."""
        from lemat_genbench.benchmarks.sun_new_benchmark import (
            create_high_precision_sun_benchmark,
        )
        
        benchmark = create_high_precision_sun_benchmark()
        
        assert isinstance(benchmark, SUNNewBenchmark)
        assert benchmark.config.name == "HighPrecisionSUNBenchmark"
        assert benchmark.config.metadata["symprec"] == 0.001
        assert benchmark.config.metadata["angle_tolerance"] == 1.0


# Manual test function for development
def manual_test():
    """Manual test for development purposes."""
    print("Running manual SUN New benchmark test...")

    try:
        # Test 1: Basic initialization
        print("1. Testing basic initialization...")
        sun_benchmark = SUNNewBenchmark()

        print(f"SUN benchmark name: {sun_benchmark.config.name}")

        # Test 2: Structure creation
        print("2. Testing structure creation...")
        structures = create_test_structures()
        multi_mlip_structures = create_test_structures_with_multi_mlip()

        print(f"Created {len(structures)} basic test structures")
        print(f"Created {len(multi_mlip_structures)} multi-MLIP test structures")

        # Test 3: Evaluator configuration
        print("3. Testing evaluator configuration...")
        assert len(sun_benchmark.evaluators) == 2  # SUN + MetaSUN
        assert len(SUNNewBenchmark(include_metasun=False).evaluators) == 1  # Only SUN

        print("Evaluators configured correctly!")

        # Test 4: Result aggregation
        print("4. Testing result aggregation...")
        mock_results = {
            "sun": {
                "combined_value": 0.25,
                "metric_results": {
                    "sun": {
                        "metrics": {
                            "sun_rate": 0.25,
                            "sun_count": 1,
                            "total_structures_evaluated": 4,
                        }
                    }
                },
            }
        }

        final_scores = sun_benchmark.aggregate_evaluator_results(mock_results)
        assert final_scores["sun_rate"] == 0.25
        assert final_scores["sun_count"] == 1

        print("Result aggregation working correctly!")

        # Test 5: Mock helper function
        print("5. Testing mock helper function...")
        mock_result = create_mock_uniqueness_result(
            ["s1", "s2", "s3"], 
            [1.0, 0.5, 1.0], 
            []
        )
        print(f"Mock fingerprints: {mock_result.fingerprints}")
        print(f"Has fingerprints: {hasattr(mock_result, 'fingerprints')}")

        # Test 6: Factory functions
        print("6. Testing factory functions...")
        from lemat_genbench.benchmarks.sun_new_benchmark import (
            create_computation_based_sun_benchmark,
            create_high_precision_sun_benchmark,
            create_property_based_sun_benchmark,
            create_robust_sun_benchmark,
        )
        
        prop_benchmark = create_property_based_sun_benchmark()
        comp_benchmark = create_computation_based_sun_benchmark()
        robust_benchmark = create_robust_sun_benchmark()
        precision_benchmark = create_high_precision_sun_benchmark()
        
        print(f"Property-based benchmark: {prop_benchmark.config.name}")
        print(f"Computation-based benchmark: {comp_benchmark.config.name}")
        print(f"Robust benchmark: {robust_benchmark.config.name}")
        print(f"High precision benchmark: {precision_benchmark.config.name}")

        print("\nAll manual tests passed!")
        return True

    except Exception as e:
        print(f"Manual test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    manual_test()