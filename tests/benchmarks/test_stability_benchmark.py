"""Tests for stability benchmark."""

from pymatgen.util.testing import PymatgenTest

from lematerial_forgebench.benchmarks.stability_benchmark import StabilityBenchmark
from lematerial_forgebench.preprocess.universal_stability_preprocess import (
    UniversalStabilityPreprocessor,
)

"""Tests for stability benchmark."""


class TestStabilityBenchmark:
    """Test suite for StabilityBenchmark class."""

    def test_initialization_default(self):
        """Test initialization with default parameters."""
        benchmark = StabilityBenchmark()

        # Check name and properties
        assert benchmark.config.name == "StabilityBenchmark"
        assert "version" in benchmark.config.metadata
        assert benchmark.config.metadata["category"] == "stability"

        # Check correct evaluator
        assert len(benchmark.evaluators) == 5
        assert "stability" in benchmark.evaluators

    def test_initialization_custom(self):
        """Test initialization with custom relaxer configuration."""

        benchmark = StabilityBenchmark(
            name="Custom Stability Benchmark",
            description="Custom description",
            metadata={"test_key": "test_value"},
        )

        # Check custom values
        assert benchmark.config.name == "Custom Stability Benchmark"
        assert benchmark.config.description == "Custom description"
        assert benchmark.config.metadata["test_key"] == "test_value"

    def test_evaluate_with_mp_entries(self):
        """Test benchmark evaluation on structures"""
        benchmark = StabilityBenchmark()

        # Create test structures
        test = PymatgenTest()
        structures = [test.get_structure("Si"), test.get_structure("LiFePO4")]

        # first, we need to preprocess the structures
        stability_preprocessor = UniversalStabilityPreprocessor()
        preprocessor_result = stability_preprocessor(structures)
        structures = preprocessor_result.processed_structures

        # Run benchmark
        result = benchmark.evaluate(structures)

        # Check result format
        assert len(result.evaluator_results) == 5
        assert "stability" in result.evaluator_results
        assert "stable_ratio" in result.final_scores

        # Check score types
        assert isinstance(result.final_scores["stable_ratio"], (int, float))
        assert isinstance(result.final_scores["metastable_ratio"], (int, float))
        assert isinstance(result.final_scores["mean_e_above_hull"], (int, float))
        assert isinstance(result.final_scores["mean_formation_energy"], (int, float))
        assert isinstance(result.final_scores["mean_relaxation_RMSE"], (int, float))

    def test_empty_structures(self):
        """Test behavior with empty structure list."""
        benchmark = StabilityBenchmark()

        # Test behavior with no structures - should not raise error
        result = benchmark.evaluate([])

        # Should get default values
        assert result.final_scores["stable_ratio"] is None
        assert result.final_scores["metastable_ratio"] is None
        assert result.final_scores["mean_e_above_hull"] is None
        assert result.final_scores["mean_formation_energy"] is None
        assert result.final_scores["mean_relaxation_RMSE"] is None

    def test_aggregate_evaluator_results(self):
        """Test result aggregation logic."""
        benchmark = StabilityBenchmark()

        # Mock evaluator_results as passed by BaseBenchmark.evaluate
        # It contains the evaluator's combined_value and the primary metric value
        # of each metric configured for that evaluator (e.g., "metric_name_value").
        mock_evaluator_results_from_base = {
            "stability": {  # Name of the evaluator
                "combined_value": 0.75,  # Evaluator's combined score
                "metric_results": {"stability": {"metrics": {"stable_ratio": 0.75}}},
            },
            "metastability": {"combined_value": 0.85},
            "formation_energy": {"combined_value": -6.7},
            "mean_e_above_hull": {"combined_value": 0.1},
            "relaxation_stability": {"combined_value": 0.01},
        }
        # Aggregate results
        scores = benchmark.aggregate_evaluator_results(mock_evaluator_results_from_base)
        # Check scores
        # aggregate_evaluator_results should pick up combined_value as stability_score
        # and stability_value as stable_ratio.
        # mean_e_above_hull and metastable_ratio will be defaults (nan, 0.0) because
        # they are not present in the input dict.
        assert scores["stable_ratio"] == 0.75
        assert scores["metastable_ratio"] == 0.85
        assert scores["mean_e_above_hull"] == 0.1
        assert scores["mean_formation_energy"] == -6.7
        assert scores["mean_relaxation_RMSE"] == 0.01

    def test_benchmark_metadata(self):
        """Test benchmark metadata structure."""
        benchmark = StabilityBenchmark()

        metadata = benchmark.config.metadata

        # Check required metadata fields
        assert metadata["version"] == "0.1.0"
        assert metadata["category"] == "stability"


def test_evaluator_configuration():
    """Test that evaluator is properly configured."""
    benchmark = StabilityBenchmark()

    # Check evaluator configuration
    stability_evaluator = benchmark.evaluators["stability"]
    assert stability_evaluator.config.name == "stability"
    assert stability_evaluator.config.weights == {"stability": 1.0}
    assert stability_evaluator.config.aggregation_method == "weighted_mean"
