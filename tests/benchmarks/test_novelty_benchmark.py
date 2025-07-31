"""Tests for novelty benchmark."""

import math
from unittest.mock import Mock, patch

import pytest
from pymatgen.core.structure import Structure
from pymatgen.util.testing import PymatgenTest

from lemat_genbench.benchmarks.novelty_benchmark import NoveltyBenchmark
from lemat_genbench.metrics.base import MetricResult
from lemat_genbench.metrics.novelty_metric import NoveltyMetric


def create_test_structures():
    """Create simple test structures for benchmarking."""
    structures = []

    # Simple cubic structure
    lattice = [[3.0, 0, 0], [0, 3.0, 0], [0, 0, 3.0]]
    structure1 = Structure(
        lattice=lattice,
        species=["Na", "Cl"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structures.append(structure1)

    # Simple face-centered cubic
    lattice = [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]]
    structure2 = Structure(
        lattice=lattice,
        species=["Cu"],
        coords=[[0, 0, 0]],
        coords_are_cartesian=False,
    )
    structures.append(structure2)

    return structures


class TestNoveltyBenchmark:
    """Test suite for NoveltyBenchmark class."""

    def test_initialization(self):
        """Test initialization with default parameters."""
        benchmark = NoveltyBenchmark()

        # Check name and properties
        assert benchmark.config.name == "NoveltyBenchmark"
        assert "version" in benchmark.config.metadata
        assert benchmark.config.metadata["category"] == "novelty"

        # Check correct evaluators
        assert len(benchmark.evaluators) == 1
        assert "novelty" in benchmark.evaluators

        # Check metadata
        metadata = benchmark.config.metadata
        assert metadata["reference_dataset"] == "LeMaterial/LeMat-Bulk"
        assert metadata["reference_config"] == "compatible_pbe"
        assert metadata["fingerprint_method"] == "bawl"

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        benchmark = NoveltyBenchmark(
            reference_dataset="custom/dataset",
            reference_config="custom_config",
            fingerprint_method="bawl",
            max_reference_size=100,
            name="Custom Novelty Benchmark",
            description="Custom description",
            metadata={"test_key": "test_value"},
        )

        # Check custom values
        assert benchmark.config.name == "Custom Novelty Benchmark"
        assert benchmark.config.description == "Custom description"
        assert benchmark.config.metadata["test_key"] == "test_value"
        assert benchmark.config.metadata["reference_dataset"] == "custom/dataset"
        assert benchmark.config.metadata["reference_config"] == "custom_config"
        assert benchmark.config.metadata["max_reference_size"] == 100

    @patch.object(NoveltyMetric, "compute")
    def test_evaluate(self, mock_compute):
        """Test benchmark evaluation on structures."""
        # Mock the NoveltyMetric.compute method
        mock_result = MetricResult(
            metrics={
                "novelty_score": 0.8,
                "novel_structures_count": 4,
                "total_structures_evaluated": 5,
            },
            primary_metric="novelty_score",
            uncertainties={"novelty_score": {"std": 0.1}},
            config=Mock(),
            computation_time=1.0,
            individual_values=[1.0, 1.0, 0.0, 1.0, 1.0],
            n_structures=5,
            failed_indices=[],
            warnings=[],
        )
        mock_compute.return_value = mock_result

        # Create benchmark with small reference for speed
        benchmark = NoveltyBenchmark(max_reference_size=10)

        # Create test structures
        structures = create_test_structures()

        # Run benchmark
        result = benchmark.evaluate(structures)

        # Check result format
        assert len(result.evaluator_results) == 1
        assert "novelty_score" in result.final_scores
        assert "novel_structures_count" in result.final_scores
        assert "total_structures_evaluated" in result.final_scores
        assert "novelty_ratio" in result.final_scores

        # Check score ranges and values
        assert 0 <= result.final_scores["novelty_score"] <= 1.0
        nr = result.final_scores["novelty_ratio"]
        ns = result.final_scores["novelty_score"]
        assert nr == ns
        assert isinstance(result.final_scores["novel_structures_count"], (int, float))
        assert isinstance(
            result.final_scores["total_structures_evaluated"], (int, float)
        )

    def test_empty_structures(self):
        """Test behavior with empty structure list."""
        benchmark = NoveltyBenchmark(max_reference_size=10)

        # Test behavior with no structures - should not raise error
        result = benchmark.evaluate([])

        # Should get default values - using math.isnan instead of np.isnan
        assert math.isnan(result.final_scores["novelty_score"])
        assert result.final_scores["novel_structures_count"] == 0
        assert result.final_scores["total_structures_evaluated"] == 0
        assert math.isnan(result.final_scores["novelty_ratio"])

    def test_aggregate_evaluator_results(self):
        """Test result aggregation logic."""
        benchmark = NoveltyBenchmark()

        # Mock evaluator results in the format passed by BaseBenchmark.evaluate
        mock_evaluator_results = {
            "novelty": {
                "combined_value": 0.75,
                "metric_results": {
                    "novelty": {
                        "metrics": {
                            "novelty_score": 0.75,
                            "novel_structures_count": 3,
                            "total_structures_evaluated": 4,
                        }
                    }
                },
            }
        }

        # Aggregate results
        scores = benchmark.aggregate_evaluator_results(mock_evaluator_results)

        # Check scores
        assert scores["novelty_score"] == 0.75
        assert scores["novelty_ratio"] == 0.75
        assert scores["novel_structures_count"] == 3
        assert scores["total_structures_evaluated"] == 4

    def test_aggregate_evaluator_results_with_mock_metric_result(self):
        """Test aggregation with MetricResult objects."""
        benchmark = NoveltyBenchmark()

        # Create a mock MetricResult
        mock_metric_result = Mock()
        mock_metric_result.metrics = {
            "novelty_score": 0.6,
            "novel_structures_count": 6,
            "total_structures_evaluated": 10,
        }

        mock_evaluator_results = {
            "novelty": {
                "combined_value": 0.6,
                "metric_results": {"novelty": mock_metric_result},
            }
        }

        scores = benchmark.aggregate_evaluator_results(mock_evaluator_results)

        assert scores["novelty_score"] == 0.6
        assert scores["novelty_ratio"] == 0.6
        assert scores["novel_structures_count"] == 6
        assert scores["total_structures_evaluated"] == 10

    def test_benchmark_metadata(self):
        """Test benchmark metadata structure."""
        benchmark = NoveltyBenchmark()

        metadata = benchmark.config.metadata

        # Check required metadata fields
        assert metadata["version"] == "0.1.0"
        assert metadata["category"] == "novelty"
        assert metadata["reference_dataset"] == "LeMaterial/LeMat-Bulk"
        assert metadata["reference_config"] == "compatible_pbe"
        assert metadata["fingerprint_method"] == "bawl"

    def test_evaluator_configuration(self):
        """Test that evaluator is properly configured."""
        benchmark = NoveltyBenchmark()

        # Check evaluator configuration
        novelty_evaluator = benchmark.evaluators["novelty"]
        assert novelty_evaluator.config.name == "novelty"  # Fixed expected name
        assert novelty_evaluator.config.weights == {"novelty": 1.0}
        assert novelty_evaluator.config.aggregation_method == "weighted_mean"

        # Check that the metric is properly configured
        assert "novelty" in novelty_evaluator.metrics
        novelty_metric = novelty_evaluator.metrics["novelty"]
        assert isinstance(novelty_metric, NoveltyMetric)

    @pytest.mark.slow
    @patch.object(NoveltyMetric, "compute")
    def test_realistic_workflow(self, mock_compute):
        """Test a complete workflow with realistic structures."""
        # Mock a realistic result
        mock_result = MetricResult(
            metrics={
                "novelty_score": 0.5,
                "novel_structures_count": 1,
                "total_structures_evaluated": 2,
            },
            primary_metric="novelty_score",
            uncertainties={"novelty_score": {"std": 0.3}},
            config=Mock(),
            computation_time=2.0,
            individual_values=[1.0, 0.0],
            n_structures=2,
            failed_indices=[],
            warnings=[],
        )
        mock_compute.return_value = mock_result

        # Use PymatgenTest structures
        test = PymatgenTest()
        structures = [test.get_structure("Si"), test.get_structure("LiFePO4")]

        # Create benchmark
        benchmark = NoveltyBenchmark(max_reference_size=50)

        # Run evaluation
        result = benchmark.evaluate(structures)

        # Should complete without error
        assert isinstance(result.final_scores, dict)
        assert "novelty_score" in result.final_scores
        assert result.final_scores["novelty_score"] == 0.5
        assert result.final_scores["novel_structures_count"] == 1
        assert result.final_scores["total_structures_evaluated"] == 2

    def test_different_reference_configs(self):
        """Test initialization with different reference configurations."""
        configs = ["compatible_pbe", "compatible_pbesol"]

        for config in configs:
            benchmark = NoveltyBenchmark(
                reference_config=config,
                max_reference_size=10,
            )

            # Should initialize without error
            assert benchmark.config.metadata["reference_config"] == config
            assert "novelty" in benchmark.evaluators

    def test_error_handling_in_aggregation(self):
        """Test error handling when aggregation receives malformed data."""
        benchmark = NoveltyBenchmark()

        # Test with missing evaluator results
        empty_results = {}
        scores = benchmark.aggregate_evaluator_results(empty_results)

        assert math.isnan(scores["novelty_score"])
        assert scores["novel_structures_count"] == 0
        assert scores["total_structures_evaluated"] == 0

        # Test with malformed evaluator results
        malformed_results = {
            "novelty": {
                "combined_value": None,
                "metric_results": {},
            }
        }
        scores = benchmark.aggregate_evaluator_results(malformed_results)

        assert math.isnan(scores["novelty_score"])
        assert scores["novel_structures_count"] == 0
        assert scores["total_structures_evaluated"] == 0
