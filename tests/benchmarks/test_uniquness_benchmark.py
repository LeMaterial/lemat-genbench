"""Tests for uniqueness benchmark."""

import math
from unittest.mock import Mock, patch

import pytest
from pymatgen.core.structure import Structure
from pymatgen.util.testing import PymatgenTest

from lemat_genbench.benchmarks.uniqueness_benchmark import (
    UniquenessBenchmark,
)
from lemat_genbench.metrics.base import MetricResult
from lemat_genbench.metrics.uniqueness_metric import UniquenessMetric


def create_test_structures():
    """Create test structures with known duplicates for benchmarking."""
    structures = []

    # Structure 1: Simple cubic NaCl
    lattice1 = [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]]
    structure1 = Structure(
        lattice=lattice1,
        species=["Na", "Cl"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structures.append(structure1)

    # Structure 2: Identical to Structure 1 (duplicate)
    structure2 = Structure(
        lattice=lattice1,
        species=["Na", "Cl"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structures.append(structure2)

    # Structure 3: Different structure - CsCl
    lattice3 = [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]]
    structure3 = Structure(
        lattice=lattice3,
        species=["Cs", "Cl"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structures.append(structure3)

    return structures


class TestUniquenessBenchmark:
    """Test suite for UniquenessBenchmark class."""

    def test_initialization(self):
        """Test initialization with default parameters."""
        benchmark = UniquenessBenchmark()

        # Check name and properties
        assert benchmark.config.name == "UniquenessBenchmark"
        assert "version" in benchmark.config.metadata
        assert benchmark.config.metadata["category"] == "uniqueness"

        # Check correct evaluators
        assert len(benchmark.evaluators) == 1
        assert "uniqueness" in benchmark.evaluators

        # Check metadata
        metadata = benchmark.config.metadata
        assert metadata["fingerprint_method"] == "bawl"

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        benchmark = UniquenessBenchmark(
            fingerprint_method="bawl",
            name="Custom Uniqueness Benchmark",
            description="Custom description",
            metadata={"test_key": "test_value"},
        )

        # Check custom values
        assert benchmark.config.name == "Custom Uniqueness Benchmark"
        assert benchmark.config.description == "Custom description"
        assert benchmark.config.metadata["test_key"] == "test_value"
        assert benchmark.config.metadata["fingerprint_method"] == "bawl"

    @patch.object(UniquenessMetric, "compute")
    def test_evaluate(self, mock_compute):
        """Test benchmark evaluation on structures."""
        # Mock the UniquenessMetric.compute method
        mock_result = MetricResult(
            metrics={
                "uniqueness_score": 0.6,
                "unique_structures_count": 3,
                "duplicate_structures_count": 2,
                "total_structures_evaluated": 5,
                "failed_fingerprinting_count": 0,
            },
            primary_metric="uniqueness_score",
            uncertainties={"uniqueness_score": {"std": 0.0}},
            config=Mock(),
            computation_time=1.0,
            individual_values=[1.0, 0.5, 1.0, 0.5, 1.0],
            n_structures=5,
            failed_indices=[],
            warnings=[],
        )
        mock_compute.return_value = mock_result

        # Create benchmark
        benchmark = UniquenessBenchmark()

        # Create test structures
        structures = create_test_structures()

        # Run benchmark
        result = benchmark.evaluate(structures)

        # Check result format
        assert len(result.evaluator_results) == 1
        assert "uniqueness_score" in result.final_scores
        assert "unique_structures_count" in result.final_scores
        assert "duplicate_structures_count" in result.final_scores
        assert "total_structures_evaluated" in result.final_scores
        assert "failed_fingerprinting_count" in result.final_scores
        assert "uniqueness_ratio" in result.final_scores

        # Check score ranges and values
        assert 0 <= result.final_scores["uniqueness_score"] <= 1.0
        assert (
            result.final_scores["uniqueness_ratio"]
            == result.final_scores["uniqueness_score"]
        )
        assert isinstance(result.final_scores["unique_structures_count"], (int, float))
        assert isinstance(
            result.final_scores["duplicate_structures_count"], (int, float)
        )
        assert isinstance(
            result.final_scores["total_structures_evaluated"], (int, float)
        )

    def test_empty_structures(self):
        """Test behavior with empty structure list."""
        benchmark = UniquenessBenchmark()

        # Test behavior with no structures - should not raise error
        result = benchmark.evaluate([])

        # Should get default values
        assert math.isnan(result.final_scores["uniqueness_score"])
        assert result.final_scores["unique_structures_count"] == 0
        assert result.final_scores["duplicate_structures_count"] == 0
        assert result.final_scores["total_structures_evaluated"] == 0
        assert result.final_scores["failed_fingerprinting_count"] == 0
        assert math.isnan(result.final_scores["uniqueness_ratio"])

    def test_aggregate_evaluator_results(self):
        """Test result aggregation logic."""
        benchmark = UniquenessBenchmark()

        # Mock evaluator results in the format passed by BaseBenchmark.evaluate
        mock_evaluator_results = {
            "uniqueness": {
                "combined_value": 0.75,
                "metric_results": {
                    "uniqueness": {
                        "metrics": {
                            "uniqueness_score": 0.75,
                            "unique_structures_count": 3,
                            "duplicate_structures_count": 1,
                            "total_structures_evaluated": 4,
                            "failed_fingerprinting_count": 0,
                        }
                    }
                },
            }
        }

        # Aggregate results
        scores = benchmark.aggregate_evaluator_results(mock_evaluator_results)

        # Check scores
        assert scores["uniqueness_score"] == 0.75
        assert scores["uniqueness_ratio"] == 0.75
        assert scores["unique_structures_count"] == 3
        assert scores["duplicate_structures_count"] == 1
        assert scores["total_structures_evaluated"] == 4
        assert scores["failed_fingerprinting_count"] == 0

    def test_aggregate_evaluator_results_with_mock_metric_result(self):
        """Test aggregation with MetricResult objects."""
        benchmark = UniquenessBenchmark()

        # Create a mock MetricResult
        mock_metric_result = Mock()
        mock_metric_result.metrics = {
            "uniqueness_score": 0.8,
            "unique_structures_count": 4,
            "duplicate_structures_count": 1,
            "total_structures_evaluated": 5,
            "failed_fingerprinting_count": 0,
        }

        mock_evaluator_results = {
            "uniqueness": {
                "combined_value": 0.8,
                "metric_results": {"uniqueness": mock_metric_result},
            }
        }

        scores = benchmark.aggregate_evaluator_results(mock_evaluator_results)

        assert scores["uniqueness_score"] == 0.8
        assert scores["uniqueness_ratio"] == 0.8
        assert scores["unique_structures_count"] == 4
        assert scores["duplicate_structures_count"] == 1
        assert scores["total_structures_evaluated"] == 5
        assert scores["failed_fingerprinting_count"] == 0

    def test_benchmark_metadata(self):
        """Test benchmark metadata structure."""
        benchmark = UniquenessBenchmark()

        metadata = benchmark.config.metadata

        # Check required metadata fields
        assert metadata["version"] == "0.1.0"
        assert metadata["category"] == "uniqueness"
        assert metadata["fingerprint_method"] == "bawl"

    def test_evaluator_configuration(self):
        """Test that evaluator is properly configured."""
        benchmark = UniquenessBenchmark()

        # Check evaluator configuration
        uniqueness_evaluator = benchmark.evaluators["uniqueness"]
        assert uniqueness_evaluator.config.name == "uniqueness"
        assert uniqueness_evaluator.config.weights == {"uniqueness": 1.0}
        assert uniqueness_evaluator.config.aggregation_method == "weighted_mean"

        # Check that the metric is properly configured
        assert "uniqueness" in uniqueness_evaluator.metrics
        uniqueness_metric = uniqueness_evaluator.metrics["uniqueness"]
        assert isinstance(uniqueness_metric, UniquenessMetric)

    @pytest.mark.slow
    @patch.object(UniquenessMetric, "compute")
    def test_realistic_workflow(self, mock_compute):
        """Test a complete workflow with realistic structures."""
        # Mock a realistic result with some duplicates
        mock_result = MetricResult(
            metrics={
                "uniqueness_score": 0.5,
                "unique_structures_count": 1,
                "duplicate_structures_count": 1,
                "total_structures_evaluated": 2,
                "failed_fingerprinting_count": 0,
            },
            primary_metric="uniqueness_score",
            uncertainties={"uniqueness_score": {"std": 0.0}},
            config=Mock(),
            computation_time=2.0,
            individual_values=[1.0, 0.5],
            n_structures=2,
            failed_indices=[],
            warnings=[],
        )
        mock_compute.return_value = mock_result

        # Use PymatgenTest structures
        test = PymatgenTest()
        structures = [test.get_structure("Si"), test.get_structure("Si")]
        # Duplicate

        # Create benchmark
        benchmark = UniquenessBenchmark()

        # Run evaluation
        result = benchmark.evaluate(structures)

        # Should complete without error
        assert isinstance(result.final_scores, dict)
        assert "uniqueness_score" in result.final_scores
        assert result.final_scores["uniqueness_score"] == 0.5
        assert result.final_scores["unique_structures_count"] == 1
        assert result.final_scores["duplicate_structures_count"] == 1

    def test_different_fingerprint_methods(self):
        """Test initialization with different fingerprint methods."""
        # Currently only BAWL is supported, but test the interface
        benchmark = UniquenessBenchmark(fingerprint_method="bawl")
        assert benchmark.config.metadata["fingerprint_method"] == "bawl"
        assert "uniqueness" in benchmark.evaluators

    def test_error_handling_in_aggregation(self):
        """Test error handling when aggregation receives malformed data."""
        benchmark = UniquenessBenchmark()

        # Test with missing evaluator results
        empty_results = {}
        scores = benchmark.aggregate_evaluator_results(empty_results)

        assert math.isnan(scores["uniqueness_score"])
        assert scores["unique_structures_count"] == 0
        assert scores["duplicate_structures_count"] == 0
        assert scores["total_structures_evaluated"] == 0
        assert scores["failed_fingerprinting_count"] == 0

        # Test with malformed evaluator results
        malformed_results = {
            "uniqueness": {
                "combined_value": None,
                "metric_results": {},
            }
        }
        scores = benchmark.aggregate_evaluator_results(malformed_results)

        assert math.isnan(scores["uniqueness_score"])
        assert scores["unique_structures_count"] == 0
        assert scores["duplicate_structures_count"] == 0
        assert scores["total_structures_evaluated"] == 0
        assert scores["failed_fingerprinting_count"] == 0

    @patch.object(UniquenessMetric, "compute")
    def test_all_unique_structures(self, mock_compute):
        """Test with all unique structures."""
        # Mock result with all unique structures
        mock_result = MetricResult(
            metrics={
                "uniqueness_score": 1.0,
                "unique_structures_count": 3,
                "duplicate_structures_count": 0,
                "total_structures_evaluated": 3,
                "failed_fingerprinting_count": 0,
            },
            primary_metric="uniqueness_score",
            uncertainties={"uniqueness_score": {"std": 0.0}},
            config=Mock(),
            computation_time=1.0,
            individual_values=[1.0, 1.0, 1.0],
            n_structures=3,
            failed_indices=[],
            warnings=[],
        )
        mock_compute.return_value = mock_result

        benchmark = UniquenessBenchmark()
        structures = create_test_structures()

        result = benchmark.evaluate(structures)

        assert result.final_scores["uniqueness_score"] == 1.0
        assert result.final_scores["unique_structures_count"] == 3
        assert result.final_scores["duplicate_structures_count"] == 0

    @patch.object(UniquenessMetric, "compute")
    def test_all_duplicate_structures(self, mock_compute):
        """Test with all duplicate structures."""
        # Mock result with all duplicate structures (only 1 unique)
        mock_result = MetricResult(
            metrics={
                "uniqueness_score": 0.25,  # 1 unique out of 4
                "unique_structures_count": 1,
                "duplicate_structures_count": 3,
                "total_structures_evaluated": 4,
                "failed_fingerprinting_count": 0,
            },
            primary_metric="uniqueness_score",
            uncertainties={"uniqueness_score": {"std": 0.0}},
            config=Mock(),
            computation_time=1.0,
            individual_values=[0.25, 0.25, 0.25, 0.25],  # All identical
            n_structures=4,
            failed_indices=[],
            warnings=[],
        )
        mock_compute.return_value = mock_result

        benchmark = UniquenessBenchmark()
        structures = create_test_structures()

        result = benchmark.evaluate(structures)

        assert result.final_scores["uniqueness_score"] == 0.25
        assert result.final_scores["unique_structures_count"] == 1
        assert result.final_scores["duplicate_structures_count"] == 3
