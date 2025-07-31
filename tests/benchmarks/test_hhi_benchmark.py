"""Fixed tests for HHI benchmark - addressing the three failing test cases."""

from pymatgen.util.testing import PymatgenTest

from lemat_genbench.benchmarks.hhi_benchmark import HHIBenchmark


def create_test_structures():
    """Create test structures for HHI evaluation - fixed to use available structures."""
    test = PymatgenTest()
    structures = [
        test.get_structure("Si"),
        test.get_structure("LiFePO4"),
        test.get_structure("CsCl"),
    ]
    return structures


class TestHHIBenchmark:
    """Test suite for HHIBenchmark class."""

    def test_evaluate_with_real_structures(self):
        """Test benchmark evaluation on real structures."""
        benchmark = HHIBenchmark()
        structures = create_test_structures()

        # Run benchmark
        result = benchmark.evaluate(structures)

        # Check result format
        assert len(result.evaluator_results) == 2
        assert "hhi_production" in result.evaluator_results
        assert "hhi_reserve" in result.evaluator_results

        # Check final scores exist
        assert "hhi_production_mean" in result.final_scores
        assert "hhi_reserve_mean" in result.final_scores
        assert "hhi_combined_mean" in result.final_scores

        # Check score types (should be float or NaN)
        for score_name in [
            "hhi_production_mean",
            "hhi_reserve_mean",
            "hhi_combined_mean",
        ]:
            score_value = result.final_scores[score_name]
            assert isinstance(score_value, (int, float))

    def test_empty_structures(self):
        """Test behavior with empty structure list."""
        benchmark = HHIBenchmark()

        # Test behavior with no structures - should not raise error
        result = benchmark.evaluate([])

        assert result.final_scores["hhi_production_mean"] is None
        assert result.final_scores["hhi_reserve_mean"] is None
        assert result.final_scores["hhi_combined_mean"] is None

    def test_missing_evaluator_results(self):
        """Test handling of missing evaluator results."""
        benchmark = HHIBenchmark()

        # Test with missing production results
        partial_results = {
            "hhi_reserve": {"combined_value": 3.0},
        }
        scores = benchmark.aggregate_evaluator_results(partial_results)
        assert scores["hhi_production_mean"] is None
        assert scores["hhi_reserve_mean"] == 3.0
        assert scores["hhi_combined_mean"] == 3.0  # Should use available value

        # Test with missing reserve results
        partial_results = {
            "hhi_production": {"combined_value": 4.0},
        }
        scores = benchmark.aggregate_evaluator_results(partial_results)

        assert scores["hhi_production_mean"] == 4.0
        assert scores["hhi_reserve_mean"] is None
        assert scores["hhi_combined_mean"] == 4.0  # Should use available value

    # Rest of the tests remain the same as they were working correctly
    def test_initialization_default(self):
        """Test initialization with default parameters."""
        benchmark = HHIBenchmark()

        # Check name and properties
        assert benchmark.config.name == "HHIBenchmark"
        assert "version" in benchmark.config.metadata
        assert benchmark.config.metadata["category"] == "supply_risk"
        assert benchmark.config.metadata["scale_to_0_10"] is True

        # Check correct evaluators
        assert len(benchmark.evaluators) == 2
        assert "hhi_production" in benchmark.evaluators
        assert "hhi_reserve" in benchmark.evaluators

        # Check weights are normalized
        assert abs(benchmark.production_weight + benchmark.reserve_weight - 1.0) < 1e-6
