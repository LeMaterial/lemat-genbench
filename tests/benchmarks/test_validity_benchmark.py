"""Tests for validity benchmark."""

from pymatgen.core.structure import Structure
from pymatgen.util.testing import PymatgenTest

from lemat_genbench.benchmarks.validity_benchmark import ValidityBenchmark


class TestValidityBenchmark:
    """Test suite for ValidityBenchmark class."""

    def test_initialization(self):
        """Test initialization with default parameters."""
        benchmark = ValidityBenchmark()

        # Check name and properties
        assert benchmark.config.name == "ValidityBenchmark"

        # Check correct evaluators - updated to 4 evaluators
        assert len(benchmark.evaluators) == 4
        assert "charge_neutrality" in benchmark.evaluators
        assert "interatomic_distance" in benchmark.evaluators
        assert "physical_plausibility" in benchmark.evaluators
        assert "overall_validity" in benchmark.evaluators

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        benchmark = ValidityBenchmark(
            charge_tolerance=0.05,
            distance_scaling=0.3,
            min_atomic_density=0.01,
            max_atomic_density=0.2,
            min_mass_density=2.0,
            max_mass_density=20.0,
            check_format=False,
            check_symmetry=False,
            name="Custom Benchmark",
            description="Custom description",
            metadata={"test_key": "test_value"},
        )

        # Check custom values
        assert benchmark.config.name == "Custom Benchmark"
        assert benchmark.config.description == "Custom description"
        if benchmark.config.metadata:
            assert benchmark.config.metadata.get("test_key") == "test_value"

    def test_evaluate(self):
        """Test benchmark evaluation on structures."""
        benchmark = ValidityBenchmark(
            check_format=False,  # Skip for speed
            check_symmetry=False
        )

        # Create test structures
        test = PymatgenTest()
        structures = [test.get_structure("Si"), test.get_structure("LiFePO4")]

        # Run benchmark
        result = benchmark.evaluate(structures)

        # Check result format - updated expected metrics
        assert len(result.evaluator_results) == 4
        expected_final_scores = [
            "charge_neutrality_ratio",
            "charge_neutrality_count",
            "avg_charge_deviation",
            "interatomic_distance_ratio",
            "interatomic_distance_count",
            "physical_plausibility_ratio",
            "physical_plausibility_count",
            "overall_validity_ratio",
            "overall_validity_count",
            "total_structures",
            "any_invalid_count",
            "any_invalid_ratio",
        ]
        
        for metric in expected_final_scores:
            assert metric in result.final_scores, f"Missing metric: {metric}"

        # Check score ranges
        ratio_metrics = [m for m in expected_final_scores if "ratio" in m]
        for metric in ratio_metrics:
            score = result.final_scores[metric]
            assert 0 <= score <= 1.0, f"{metric} should be between 0 and 1, got {score}"

        # Check count metrics
        count_metrics = [m for m in expected_final_scores if "count" in m]
        for metric in count_metrics:
            count = result.final_scores[metric]
            assert isinstance(count, (int, float)), f"{metric} should be numeric"
            assert count >= 0, f"{metric} should be non-negative"

    def test_empty_structures(self):
        """Test behavior with empty structure list."""
        benchmark = ValidityBenchmark()

        # Test behavior with no structures - should not raise error
        result = benchmark.evaluate([])

        # Should get default values
        assert result.final_scores["overall_validity_ratio"] == 0.0
        assert result.final_scores["total_structures"] == 0

    def test_count_consistency(self):
        """Test that counts and ratios are consistent."""
        test = PymatgenTest()
        structures = [test.get_structure("Si")] * 10  # 10 identical structures
        
        benchmark = ValidityBenchmark(check_format=False, check_symmetry=False)
        result = benchmark.evaluate(structures)
        
        total = result.final_scores["total_structures"]
        assert total == 10
        
        # Check consistency between counts and ratios
        charge_count = result.final_scores["charge_neutrality_count"]
        charge_ratio = result.final_scores["charge_neutrality_ratio"]
        assert abs(charge_ratio - (charge_count / total)) < 1e-10
        
        distance_count = result.final_scores["interatomic_distance_count"]
        distance_ratio = result.final_scores["interatomic_distance_ratio"]
        assert abs(distance_ratio - (distance_count / total)) < 1e-10
        
        overall_count = result.final_scores["overall_validity_count"]
        overall_ratio = result.final_scores["overall_validity_ratio"]
        assert abs(overall_ratio - (overall_count / total)) < 1e-10

    def test_intersection_logic(self):
        """Test that overall validity is intersection of individual checks."""
        # Create normal and invalid structures
        test = PymatgenTest()
        si = test.get_structure("Si")

        # Create invalid structure - extremely compressed using proper method
        compressed_lattice = si.lattice.scale(0.1)
        compressed_si = Structure(compressed_lattice, si.species, si.frac_coords)

        # Create benchmark and evaluate
        benchmark = ValidityBenchmark(check_format=False, check_symmetry=False)
        result = benchmark.evaluate([si, compressed_si])

        # Overall count should be <= min of individual counts
        overall_count = result.final_scores["overall_validity_count"]
        charge_count = result.final_scores["charge_neutrality_count"]
        distance_count = result.final_scores["interatomic_distance_count"]
        plausibility_count = result.final_scores["physical_plausibility_count"]
        
        assert overall_count <= min(charge_count, distance_count, plausibility_count)
        
        # The overall validity should be lower than evaluating just valid structures
        valid_result = benchmark.evaluate([si])
        assert (
            valid_result.final_scores["overall_validity_ratio"]
            >= result.final_scores["overall_validity_ratio"]
        )

    def test_invalid_count_calculation(self):
        """Test that invalid count calculation is correct."""
        test = PymatgenTest()
        si = test.get_structure("Si")
        
        # Create invalid structure
        compressed_si = Structure(si.lattice.scale(0.1), si.species, si.frac_coords)
        
        # Mix of structures
        structures = [si, compressed_si, si]  # 3 total structures
        
        benchmark = ValidityBenchmark(check_format=False, check_symmetry=False)
        result = benchmark.evaluate(structures)
        
        total = result.final_scores["total_structures"]
        overall_count = result.final_scores["overall_validity_count"]
        invalid_count = result.final_scores["any_invalid_count"]
        invalid_ratio = result.final_scores["any_invalid_ratio"]
        
        # Check consistency
        assert total == 3
        assert invalid_count == (total - overall_count)
        assert abs(invalid_ratio - (invalid_count / total)) < 1e-10

    def test_deviation_reporting(self):
        """Test that charge deviation is properly reported."""
        test = PymatgenTest()
        structures = [test.get_structure("Si"), test.get_structure("LiFePO4")]
        
        benchmark = ValidityBenchmark()
        result = benchmark.evaluate(structures)
        
        # Should have deviation information
        assert "avg_charge_deviation" in result.final_scores
        deviation = result.final_scores["avg_charge_deviation"]
        
        # Should be a valid number (not NaN) and non-negative
        assert isinstance(deviation, (int, float))
        if not float('nan') == deviation:  # Allow NaN but check if it's a number
            assert deviation >= 0.0

    def test_parameter_effects(self):
        """Test that different parameters affect results appropriately."""
        test = PymatgenTest()
        structure = test.get_structure("Si")
        
        # Strict benchmark
        strict_benchmark = ValidityBenchmark(
            charge_tolerance=0.001,
            distance_scaling=0.9,
            min_atomic_density=0.01,
            max_atomic_density=0.2,
            min_mass_density=2.0,
            max_mass_density=20.0,
            check_format=True,
            check_symmetry=True
        )
        
        # Lenient benchmark
        lenient_benchmark = ValidityBenchmark(
            charge_tolerance=1.0,
            distance_scaling=0.1,
            min_atomic_density=0.00001,
            max_atomic_density=0.5,
            min_mass_density=0.1,
            max_mass_density=50.0,
            check_format=False,
            check_symmetry=False
        )
        
        strict_result = strict_benchmark.evaluate([structure])
        lenient_result = lenient_benchmark.evaluate([structure])
        
        # Lenient should have higher or equal validity ratios
        assert (lenient_result.final_scores["charge_neutrality_ratio"] >= 
                strict_result.final_scores["charge_neutrality_ratio"])
        assert (lenient_result.final_scores["interatomic_distance_ratio"] >= 
                strict_result.final_scores["interatomic_distance_ratio"])
        assert (lenient_result.final_scores["physical_plausibility_ratio"] >= 
                strict_result.final_scores["physical_plausibility_ratio"])
        assert (lenient_result.final_scores["overall_validity_ratio"] >= 
                strict_result.final_scores["overall_validity_ratio"])

    def test_single_structure_evaluation(self):
        """Test evaluation with a single structure."""
        test = PymatgenTest()
        structure = test.get_structure("Si")
        
        benchmark = ValidityBenchmark(check_format=False, check_symmetry=False)
        result = benchmark.evaluate([structure])
        
        # Should work with single structure
        assert result.final_scores["total_structures"] == 1
        
        # All counts should be 0 or 1
        count_metrics = [
            "charge_neutrality_count",
            "interatomic_distance_count", 
            "physical_plausibility_count",
            "overall_validity_count",
            "any_invalid_count"
        ]
        
        for metric in count_metrics:
            count = result.final_scores[metric]
            assert count in [0, 1], f"{metric} should be 0 or 1 for single structure"

    def test_all_invalid_structures(self):
        """Test benchmark on all invalid structures."""
        test = PymatgenTest()
        si = test.get_structure("Si")
        
        # Create multiple invalid structures
        invalid_structures = []
        for scale in [0.05, 0.1, 0.15]:  # Very compressed structures
            compressed_lattice = si.lattice.scale(scale)
            invalid_structures.append(Structure(compressed_lattice, si.species, si.frac_coords))
        
        benchmark = ValidityBenchmark(check_format=False, check_symmetry=False)
        result = benchmark.evaluate(invalid_structures)
        
        # Should handle all invalid gracefully
        assert result.final_scores["total_structures"] == len(invalid_structures)
        assert result.final_scores["overall_validity_count"] <= len(invalid_structures)
        assert result.final_scores["any_invalid_count"] >= 0

    def test_mixed_structure_types(self):
        """Test benchmark on different types of structures."""
        test = PymatgenTest()
        structures = [
            test.get_structure("Si"),      # Simple cubic
            test.get_structure("LiFePO4"), # Complex ionic
            test.get_structure("CsCl"),    # Simple ionic
        ]
        
        benchmark = ValidityBenchmark(check_format=False, check_symmetry=False)
        result = benchmark.evaluate(structures)
        
        # Should handle different structure types
        assert result.final_scores["total_structures"] == len(structures)
        
        # All metrics should be present and reasonable
        assert 0.0 <= result.final_scores["overall_validity_ratio"] <= 1.0
        assert result.final_scores["overall_validity_count"] <= len(structures)


def test_benchmark_with_invalid_structures():
    """Test benchmark on structures with validity issues."""
    # Create normal and invalid structures
    test = PymatgenTest()
    si = test.get_structure("Si")

    # Create invalid structure - extremely compressed using proper method
    compressed_lattice = si.lattice.scale(0.1)
    compressed_si = Structure(compressed_lattice, si.species, si.frac_coords)

    # Create benchmark and evaluate
    benchmark = ValidityBenchmark(check_format=False, check_symmetry=False)
    result = benchmark.evaluate([si, compressed_si])

    # Check that results are reasonable
    assert 0.0 <= result.final_scores["overall_validity_ratio"] <= 1.0

    # The overall validity should be lower than evaluating just valid structures
    valid_result = benchmark.evaluate([si])
    assert (
        valid_result.final_scores["overall_validity_ratio"]
        >= result.final_scores["overall_validity_ratio"]
    )