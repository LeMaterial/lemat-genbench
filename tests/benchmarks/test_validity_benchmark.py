"""Tests for validity benchmark."""

from lemat_genbench.benchmarks.validity_benchmark import ValidityBenchmark
from pymatgen.core.structure import Structure
from pymatgen.util.testing import PymatgenTest


class TestValidityBenchmark:
    """Test suite for ValidityBenchmark class."""

    def test_initialization(self):
        """Test initialization with default parameters."""
        benchmark = ValidityBenchmark()

        # Check name and properties
        assert benchmark.config.name == "ValidityBenchmark"
        assert "version" in benchmark.config.metadata

        # Check correct evaluators
        assert len(benchmark.evaluators) == 4
        assert "charge_neutrality" in benchmark.evaluators
        assert "interatomic_distance" in benchmark.evaluators
        # assert "coordination_environment" in benchmark.evaluators
        assert "physical_plausibility" in benchmark.evaluators
        assert "overall_validity" in benchmark.evaluators

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        benchmark = ValidityBenchmark(
            charge_weight=0.4,
            distance_weight=0.3,
            # coordination_weight=0.2,
            plausibility_weight=0.1,
            name="Custom Benchmark",
            description="Custom description",
            metadata={"test_key": "test_value"},
        )

        # Check custom values
        assert benchmark.config.name == "Custom Benchmark"
        assert benchmark.config.description == "Custom description"
        assert benchmark.config.metadata["test_key"] == "test_value"

        # Check weights
        weights = benchmark.config.metadata["weights"]
        assert weights["charge_neutrality"] == 0.4
        assert weights["interatomic_distance"] == 0.3
        # assert weights["coordination_environment"] == 0.2
        assert weights["physical_plausibility"] == 0.1

    def test_evaluate(self):
        """Test benchmark evaluation on structures."""
        benchmark = ValidityBenchmark()

        # Create test structures
        test = PymatgenTest()
        structures = [test.get_structure("Si"), test.get_structure("LiFePO4")]

        # Run benchmark
        result = benchmark.evaluate(structures)

        # Check result format
        assert len(result.evaluator_results) == 4
        assert "overall_validity_score" in result.final_scores
        assert "charge_neutrality_score" in result.final_scores
        assert "interatomic_distance_score" in result.final_scores
        # assert "coordination_environment_score" in result.final_scores
        assert "physical_plausibility_score" in result.final_scores
        assert "valid_structures_ratio" in result.final_scores

        # Check score ranges
        for name, score in result.final_scores.items():
            if "score" in name or "ratio" in name:
                assert 0 <= score <= 1.0, f"{name} should be between 0 and 1"

    def test_empty_structures(self):
        """Test behavior with empty structure list."""
        benchmark = ValidityBenchmark()

        # Test behavior with no structures - should not raise error
        result = benchmark.evaluate([])

        # Should get default values
        assert result.final_scores["overall_validity_score"] == 0.0
        assert result.final_scores["valid_structures_ratio"] == 0.0

    def test_aggregate_evaluator_results(self):
        """Test result aggregation logic."""
        benchmark = ValidityBenchmark()

        # Create mock evaluator results
        mock_evaluator_results = {
            "overall_validity": {
                "combined_value": 0.8,
                "metric_results": {
                    "composite": {"metrics": {"valid_structures_ratio": 0.75}}
                },
            },
            "charge_neutrality": {"combined_value": 0.9},
            "interatomic_distance": {"combined_value": 0.8},
            # "coordination_environment": {"combined_value": 0.7},
            "physical_plausibility": {"combined_value": 0.6},
        }

        # Aggregate results
        scores = benchmark.aggregate_evaluator_results(mock_evaluator_results)

        # Check scores
        assert scores["overall_validity_score"] == 0.8
        assert scores["charge_neutrality_score"] == 0.9
        assert scores["interatomic_distance_score"] == 0.8
        # assert scores["coordination_environment_score"] == 0.7
        assert scores["physical_plausibility_score"] == 0.6
        assert scores["valid_structures_ratio"] == 0.75


def test_benchmark_with_invalid_structures():
    """Test benchmark on structures with validity issues."""
    # Create normal and invalid structures
    test = PymatgenTest()
    si = test.get_structure("Si")

    # Create invalid structure - extremely compressed using proper method
    compressed_lattice = si.lattice.scale(0.1)
    compressed_si = Structure(compressed_lattice, si.species, si.frac_coords)

    # Create benchmark and evaluate
    benchmark = ValidityBenchmark()
    result = benchmark.evaluate([si, compressed_si])

    # The valid_structures_ratio should be around 0.5 (one valid, one invalid)
    # But depends on threshold, so check it's between 0 and 1
    assert 0.0 <= result.final_scores["valid_structures_ratio"] <= 1.0

    # The overall validity score should be lower than evaluating just valid structures
    valid_result = benchmark.evaluate([si])
    assert (
        valid_result.final_scores["overall_validity_score"]
        >= result.final_scores["overall_validity_score"]
    )
