"""Tests for validity benchmark."""

import pickle

from pymatgen.util.testing import PymatgenTest

from lematerial_forgebench.benchmarks.diversity_benchmark import (
    DiversityBenchmark,
)


class TestDiversityBenchmark:
    """Test suite for DiversityBenchmark class."""

    def test_initialization(self):
        """Test initialization with default parameters."""
        benchmark = DiversityBenchmark()

        # Check name and properties
        assert benchmark.config.name == "DiversityBenchmark"
        assert "version" in benchmark.config.metadata

        # Check correct evaluators
        assert len(benchmark.evaluators) == 4
        assert "element_diversity" in benchmark.evaluators
        assert "space_group_diversity" in benchmark.evaluators
        assert "site_number_diversity" in benchmark.evaluators
        assert "physical_size_diversity" in benchmark.evaluators

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""

        benchmark = DiversityBenchmark(
            name="Custom Benchmark",
            description="Custom description",
            metadata={"test_key": "test_value"},
        )

        # Check custom values
        assert benchmark.config.name == "Custom Benchmark"
        assert benchmark.config.description == "Custom description"
        assert benchmark.config.metadata["test_key"] == "test_value"

    def test_evaluate(self):
        """Test benchmark evaluation on structures."""
        test = PymatgenTest()
        structures = [test.get_structure("Si"), test.get_structure("LiFePO4")]

        benchmark = DiversityBenchmark()
        result = benchmark.evaluate(structures)

        # Check result format
        assert len(result.evaluator_results) == 4
        assert "element_diversity" in result.final_scores
        assert "space_group_diversity" in result.final_scores
        assert "site_number_diversity" in result.final_scores
        assert "physical_size_diversity" in result.final_scores

