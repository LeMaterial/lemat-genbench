"""Tests for SUN (Stable, Unique, Novel) benchmark."""

import math
from unittest.mock import MagicMock, Mock, patch

from pymatgen.core.structure import Structure

from lemat_genbench.benchmarks.sun_benchmark import SUNBenchmark
from lemat_genbench.metrics.sun_metric import MetaSUNMetric, SUNMetric


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


class TestSUNBenchmark:
    """Test suite for SUNBenchmark class."""

    def test_initialization(self):
        """Test initialization with default parameters."""
        benchmark = SUNBenchmark()

        # Check name and properties
        assert benchmark.config.name == "SUNBenchmark"
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
        assert metadata["reference_dataset"] == "LeMaterial/LeMat-Bulk"
        assert metadata["reference_config"] == "compatible_pbe"
        assert metadata["fingerprint_method"] == "bawl"
        assert metadata["include_metasun"] is True

    def test_initialization_without_metasun(self):
        """Test initialization without MetaSUN evaluator."""
        benchmark = SUNBenchmark(include_metasun=False)

        # Should only have SUN evaluator
        assert len(benchmark.evaluators) == 1
        assert "sun" in benchmark.evaluators
        assert "metasun" not in benchmark.evaluators

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        benchmark = SUNBenchmark(
            stability_threshold=0.05,
            metastability_threshold=0.15,
            reference_dataset="custom/dataset",
            reference_config="custom_config",
            fingerprint_method="bawl",
            max_reference_size=100,
            include_metasun=False,
            name="Custom SUN Benchmark",
            description="Custom description",
            metadata={"test_key": "test_value"},
        )

        # Check custom values
        assert benchmark.config.name == "Custom SUN Benchmark"
        assert benchmark.config.description == "Custom description"
        assert benchmark.config.metadata["test_key"] == "test_value"
        assert benchmark.config.metadata["stability_threshold"] == 0.05
        assert benchmark.config.metadata["metastability_threshold"] == 0.15
        assert benchmark.config.metadata["reference_dataset"] == "custom/dataset"
        assert benchmark.config.metadata["include_metasun"] is False

        # Should only have SUN evaluator
        assert len(benchmark.evaluators) == 1

    def test_evaluator_configuration(self):
        """Test that evaluators are properly configured."""
        benchmark = SUNBenchmark()

        # Check SUN evaluator
        sun_evaluator = benchmark.evaluators["sun"]
        assert sun_evaluator.config.name == "sun"  # Name is set to the key in the dict
        assert "sun" in sun_evaluator.metrics
        assert isinstance(sun_evaluator.metrics["sun"], SUNMetric)

        # Check MetaSUN evaluator
        metasun_evaluator = benchmark.evaluators["metasun"]
        assert (
            metasun_evaluator.config.name == "metasun"
        )  # Name is set to the key in the dict
        assert "metasun" in metasun_evaluator.metrics
        assert isinstance(metasun_evaluator.metrics["metasun"], MetaSUNMetric)

    @patch("lemat_genbench.metrics.sun_metric.NoveltyMetric")
    @patch("lemat_genbench.metrics.sun_metric.UniquenessMetric")
    def test_aggregate_evaluator_results_basic(
        self, mock_uniqueness_class, mock_novelty_class
    ):
        """Test basic result aggregation."""
        benchmark = SUNBenchmark(include_metasun=False)

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

    @patch("lemat_genbench.metrics.sun_metric.NoveltyMetric")
    @patch("lemat_genbench.metrics.sun_metric.UniquenessMetric")
    def test_aggregate_evaluator_results_with_metasun(
        self, mock_uniqueness_class, mock_novelty_class
    ):
        """Test result aggregation with both SUN and MetaSUN evaluators."""
        benchmark = SUNBenchmark(include_metasun=True)

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
        benchmark = SUNBenchmark()

        final_scores = benchmark.aggregate_evaluator_results({})

        # Should return default NaN values
        assert math.isnan(final_scores["sun_rate"])
        assert math.isnan(final_scores["msun_rate"])
        assert final_scores["sun_count"] == 0
        assert final_scores["msun_count"] == 0
        assert final_scores["total_structures_evaluated"] == 0

    def test_aggregate_evaluator_results_with_dict_metrics(self):
        """Test result aggregation when metrics are returned as dictionaries."""
        benchmark = SUNBenchmark(include_metasun=False)

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

    @patch("lemat_genbench.metrics.sun_metric.NoveltyMetric")
    @patch("lemat_genbench.metrics.sun_metric.UniquenessMetric")
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

        benchmark = SUNBenchmark(include_metasun=False)

        result = benchmark.evaluate(structures)

        # Check that we got results
        assert result is not None
        assert result.final_scores is not None
        assert "sun_rate" in result.final_scores

        # Check metadata
        assert result.metadata["benchmark_name"] == "SUNBenchmark"
        assert result.metadata["n_structures"] == 3

    def test_benchmark_with_invalid_structures(self):
        """Test benchmark behavior with structures missing required properties."""
        benchmark = SUNBenchmark(include_metasun=False)

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
            patch("lemat_genbench.metrics.sun_metric.NoveltyMetric"),
            patch("lemat_genbench.metrics.sun_metric.UniquenessMetric"),
        ):
            result = benchmark.evaluate(structures)
            assert result is not None


class TestSUNBenchmarkIntegration:
    """Integration tests for SUNBenchmark with real metric components."""

    @patch("lemat_genbench.metrics.sun_metric.NoveltyMetric")
    @patch("lemat_genbench.metrics.sun_metric.UniquenessMetric")
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

        benchmark = SUNBenchmark()

        result = benchmark.evaluate(structures)

        # Should handle multi-MLIP properties correctly
        assert result is not None
        assert "sun_rate" in result.final_scores
        assert result.metadata["n_structures"] == 2

    @patch("lemat_genbench.metrics.sun_metric.NoveltyMetric")
    @patch("lemat_genbench.metrics.sun_metric.UniquenessMetric")
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

        benchmark = SUNBenchmark(include_metasun=False)

        # Run benchmark multiple times
        result1 = benchmark.evaluate(structures)
        result2 = benchmark.evaluate(structures)

        # Results should be identical (deterministic)
        assert result1.final_scores["sun_rate"] == result2.final_scores["sun_rate"]
        assert result1.final_scores["sun_count"] == result2.final_scores["sun_count"]

    def test_benchmark_with_empty_structures(self):
        """Test benchmark behavior with empty structure list."""
        benchmark = SUNBenchmark()

        result = benchmark.evaluate([])

        # Should handle empty input gracefully
        assert result is not None
        assert result.metadata["n_structures"] == 0

    @patch("lemat_genbench.metrics.sun_metric.NoveltyMetric")
    @patch("lemat_genbench.metrics.sun_metric.UniquenessMetric")
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

        benchmark = SUNBenchmark(include_metasun=False)

        result = benchmark.evaluate(structures)

        # Should handle larger sets efficiently
        assert result is not None
        assert result.metadata["n_structures"] == n_structures


# Manual test function for development
def manual_test():
    """Manual test for development purposes."""
    print("Running manual SUN benchmark test...")

    try:
        # Test 1: Basic initialization
        print("1. Testing basic initialization...")
        sun_benchmark = SUNBenchmark()

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
        assert len(SUNBenchmark(include_metasun=False).evaluators) == 1  # Only SUN

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

        print("\nAll manual tests passed!")
        return True

    except Exception as e:
        print(f"Manual test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    manual_test()