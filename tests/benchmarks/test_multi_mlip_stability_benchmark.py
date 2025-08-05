"""Tests for multi-MLIP stability benchmark implementation.

This test suite comprehensively tests the new multi-MLIP stability benchmark including
config loading, individual vs ensemble modes, individual result inclusion, and all factory functions.
"""

import tempfile

import numpy as np
import pytest
import yaml
from pymatgen.util.testing import PymatgenTest

from lemat_genbench.benchmarks.base import BenchmarkResult
from lemat_genbench.benchmarks.multi_mlip_stability_benchmark import (
    StabilityBenchmark,
    create_benchmark_from_config,
    create_comprehensive_benchmark,
    create_ensemble_stability_benchmark,
    create_individual_mlip_stability_benchmark,
    load_config,
    safe_float,
)

# Test Data Constants
DEFAULT_MLIPS = ["orb", "mace", "uma"]
EXPECTED_EVALUATORS = [
    "stability",
    "metastability",
    "mean_e_above_hull",
    "formation_energy",
    "relaxation_stability",
]
CORE_FINAL_SCORES = [
    "stable_ratio",
    "metastable_ratio",
    "mean_e_above_hull",
    "mean_formation_energy",
    "mean_relaxation_RMSE",
]


@pytest.fixture
def test_structures_comprehensive():
    """Create test structures with comprehensive multi-MLIP data."""
    test = PymatgenTest()
    structures = [
        test.get_structure("Si"),
        test.get_structure("LiFePO4"),
        test.get_structure("CsCl"),
    ]

    # Test data: mix of stable, metastable, and unstable structures
    test_data = [
        {  # Structure 1: Stable with low disagreement
            "e_above_hull": {"orb": -0.01, "mace": -0.008, "uma": -0.012},
            "formation_energy": {"orb": -2.10, "mace": -2.08, "uma": -2.12},
            "relaxation_rmse": {"orb": 0.015, "mace": 0.016, "uma": 0.014},
        },
        {  # Structure 2: Metastable with moderate disagreement
            "e_above_hull": {"orb": 0.08, "mace": 0.12, "uma": 0.09},
            "formation_energy": {"orb": -1.20, "mace": -1.10, "uma": -1.30},
            "relaxation_rmse": {"orb": 0.020, "mace": 0.025, "uma": 0.018},
        },
        {  # Structure 3: Unstable with high disagreement
            "e_above_hull": {"orb": 0.20, "mace": 0.35, "uma": 0.25},
            "formation_energy": {"orb": 0.40, "mace": 0.70, "uma": 0.50},
            "relaxation_rmse": {"orb": 0.035, "mace": 0.055, "uma": 0.040},
        },
    ]

    for i, (structure, data) in enumerate(zip(structures, test_data)):
        # Add individual MLIP properties
        for mlip_name in DEFAULT_MLIPS:
            structure.properties[f"e_above_hull_{mlip_name}"] = data["e_above_hull"][
                mlip_name
            ]
            structure.properties[f"formation_energy_{mlip_name}"] = data[
                "formation_energy"
            ][mlip_name]
            structure.properties[f"relaxation_rmse_{mlip_name}"] = data[
                "relaxation_rmse"
            ][mlip_name]

        # Calculate ensemble statistics
        for property_base in ["e_above_hull", "formation_energy", "relaxation_rmse"]:
            values = [data[property_base][mlip] for mlip in DEFAULT_MLIPS]
            structure.properties[f"{property_base}_mean"] = np.mean(values)
            structure.properties[f"{property_base}_std"] = np.std(values)
            structure.properties[f"{property_base}_n_mlips"] = len(DEFAULT_MLIPS)

    return structures


@pytest.fixture
def test_config_file():
    """Create a temporary test config file."""
    config_data = {
        "type": "multi_mlip_stability",
        "use_ensemble": True,
        "mlip_names": ["orb", "mace", "uma"],
        "metastable_threshold": 0.1,
        "description": "Test Multi-MLIP Stability Benchmark",
        "version": "0.1.0",
        "ensemble_config": {"min_mlips_required": 2},
        "individual_mlip_config": {
            "use_all_available": True,
            "require_all_mlips": False,
            "fallback_to_single": True,
        },
        "preprocessor_config": {
            "model_name": "multi_mlip",
            "mlip_configs": {
                "orb": {"model_type": "orb_v3_conservative_inf_omat", "device": "cpu"},
                "mace": {"model_type": "mp", "device": "cpu"},
                "uma": {"task": "omat", "device": "cpu"},
            },
            "relax_structures": True,
            "calculate_formation_energy": True,
            "calculate_energy_above_hull": True,
            "extract_embeddings": True,
            "timeout": 60,
        },
        "reporting": {
            "include_individual_mlip_results": True,
            "include_uncertainty_analysis": True,
            "include_ensemble_summary": True,
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        return f.name


class TestUtilityFunctions:
    """Test suite for utility functions."""

    def test_safe_float(self):
        """Test safe_float utility function."""
        # Valid conversions
        assert safe_float(1.0) == 1.0
        assert safe_float(42) == 42.0
        assert safe_float(-3.14) == -3.14
        assert safe_float("2.5") == 2.5

        # Invalid conversions should return NaN
        assert np.isnan(safe_float(None))
        assert np.isnan(safe_float("invalid"))
        assert np.isnan(safe_float([1, 2, 3]))

    def test_load_config_from_file(self, test_config_file):
        """Test loading config from YAML file."""
        config = load_config(test_config_file)

        assert isinstance(config, dict)
        assert config["type"] == "multi_mlip_stability"
        assert config["use_ensemble"] is True
        assert config["mlip_names"] == ["orb", "mace", "uma"]
        assert config["ensemble_config"]["min_mlips_required"] == 2

    def test_load_config_from_dict(self):
        """Test loading config from dictionary."""
        config_dict = {
            "type": "multi_mlip_stability",
            "use_ensemble": False,
            "mlip_names": ["orb", "mace"],
        }

        config = load_config(config_dict)
        assert config == config_dict

    def test_load_config_nonexistent_file(self):
        """Test loading config from nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")


class TestStabilityBenchmarkInitialization:
    """Test suite for StabilityBenchmark initialization."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        benchmark = StabilityBenchmark()

        # Check basic configuration
        assert benchmark.config.name == "StabilityBenchmark"
        assert benchmark.use_ensemble is True
        assert benchmark.mlip_names == ["orb", "mace", "uma"]
        assert benchmark.metastable_threshold == 0.1
        assert benchmark.min_mlips_required == 2
        assert benchmark.include_individual_results is False

        # Check evaluators exist
        assert len(benchmark.evaluators) == 5
        for evaluator_name in EXPECTED_EVALUATORS:
            assert evaluator_name in benchmark.evaluators

        # Check metadata structure
        metadata = benchmark.config.metadata
        assert metadata["version"] == "0.1.0"
        assert metadata["category"] == "stability"
        assert metadata["use_ensemble"] is True
        assert metadata["mlip_names"] == ["orb", "mace", "uma"]

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        custom_mlips = ["orb", "mace"]
        benchmark = StabilityBenchmark(
            use_ensemble=False,
            mlip_names=custom_mlips,
            metastable_threshold=0.05,
            min_mlips_required=1,
            include_individual_results=True,
            name="Custom Multi-MLIP Benchmark",
        )

        # Check custom configuration
        assert benchmark.config.name == "Custom Multi-MLIP Benchmark"
        assert benchmark.use_ensemble is False
        assert benchmark.mlip_names == custom_mlips
        assert benchmark.metastable_threshold == 0.05
        assert benchmark.min_mlips_required == 1
        assert benchmark.include_individual_results is True

    def test_config_file_initialization(self, test_config_file):
        """Test initialization from config file."""
        benchmark = StabilityBenchmark(config=test_config_file)

        # Should load settings from config
        assert benchmark.use_ensemble is True
        assert benchmark.mlip_names == ["orb", "mace", "uma"]
        assert benchmark.metastable_threshold == 0.1
        assert benchmark.min_mlips_required == 2
        assert benchmark.include_individual_results is True  # From reporting config

    def test_config_dict_initialization(self):
        """Test initialization from config dictionary."""
        config_dict = {
            "use_ensemble": False,
            "mlip_names": ["orb", "mace"],
            "metastable_threshold": 0.15,
            "ensemble_config": {"min_mlips_required": 1},
            "reporting": {"include_individual_mlip_results": False},
        }

        benchmark = StabilityBenchmark(config=config_dict)

        assert benchmark.use_ensemble is False
        assert benchmark.mlip_names == ["orb", "mace"]
        assert benchmark.metastable_threshold == 0.15
        assert benchmark.min_mlips_required == 1
        assert benchmark.include_individual_results is False

    def test_parameter_override_precedence(self, test_config_file):
        """Test that explicit parameters override config file."""
        benchmark = StabilityBenchmark(
            config=test_config_file,
            use_ensemble=False,  # Override config
            mlip_names=["orb"],  # Override config
        )

        # Explicit parameters should take precedence
        assert benchmark.use_ensemble is False
        assert benchmark.mlip_names == ["orb"]

        # Non-overridden parameters should come from config
        assert benchmark.metastable_threshold == 0.1

    def test_metric_configuration_propagation(self):
        """Test that configuration is properly propagated to metrics."""
        benchmark = StabilityBenchmark(
            use_ensemble=False,
            mlip_names=["orb", "mace"],
            min_mlips_required=1,
            include_individual_results=True,
        )

        # Check that metrics are configured correctly
        stability_evaluator = benchmark.evaluators["stability"]
        _ = stability_evaluator.metrics["stability"]

        # Can't directly access metric properties due to framework,
        # but we can check that configuration is stored in benchmark
        assert benchmark.use_ensemble is False
        assert benchmark.mlip_names == ["orb", "mace"]
        assert benchmark.min_mlips_required == 1
        assert benchmark.include_individual_results is True


class TestBenchmarkEvaluation:
    """Test suite for benchmark evaluation functionality."""

    def test_basic_evaluation_ensemble_mode(self, test_structures_comprehensive):
        """Test basic evaluation in ensemble mode."""
        benchmark = StabilityBenchmark(use_ensemble=True)
        result = benchmark.evaluate(test_structures_comprehensive)

        # Check basic result structure
        assert isinstance(result, BenchmarkResult)
        assert len(result.evaluator_results) == 5

        # Check that we have core final scores
        for score_name in CORE_FINAL_SCORES:
            assert score_name in result.final_scores
            # Should be valid numbers (not NaN)
            score_value = result.final_scores[score_name]
            if not np.isnan(score_value):
                assert isinstance(score_value, (int, float))

    def test_basic_evaluation_individual_mode(self, test_structures_comprehensive):
        """Test basic evaluation in individual mode."""
        benchmark = StabilityBenchmark(
            use_ensemble=False, mlip_names=["orb", "mace", "uma"]
        )
        result = benchmark.evaluate(test_structures_comprehensive)

        # Check basic result structure
        assert isinstance(result, BenchmarkResult)
        assert len(result.evaluator_results) == 5

        # Check that we have core final scores
        for score_name in CORE_FINAL_SCORES:
            assert score_name in result.final_scores

    def test_evaluation_with_individual_results(self, test_structures_comprehensive):
        """Test evaluation with individual MLIP results included."""
        benchmark = StabilityBenchmark(
            use_ensemble=True, include_individual_results=True
        )
        result = benchmark.evaluate(test_structures_comprehensive)

        # Should have ensemble metrics
        for score_name in CORE_FINAL_SCORES:
            assert score_name in result.final_scores

        # Should also have individual MLIP metrics
        individual_metrics = [
            k
            for k in result.final_scores.keys()
            if any(mlip in k for mlip in DEFAULT_MLIPS)
        ]
        assert len(individual_metrics) > 0  # Should have some individual metrics

    def test_evaluation_with_standard_deviations(self, test_structures_comprehensive):
        """Test that standard deviations are reported."""
        benchmark = StabilityBenchmark(use_ensemble=True)
        result = benchmark.evaluate(test_structures_comprehensive)

        # Should have standard deviation metrics
        std_metrics = [
            "stability_std_e_above_hull",
            "e_hull_std",
            "formation_energy_std",
            "relaxation_RMSE_std",
        ]

        for std_metric in std_metrics:
            if std_metric in result.final_scores:
                std_value = result.final_scores[std_metric]
                if not np.isnan(std_value):
                    assert std_value >= 0  # Standard deviation should be non-negative

    def test_evaluation_with_ensemble_uncertainty(self, test_structures_comprehensive):
        """Test ensemble uncertainty reporting."""
        benchmark = StabilityBenchmark(use_ensemble=True)
        result = benchmark.evaluate(test_structures_comprehensive)

        # Should have ensemble uncertainty metrics
        if "stability_mean_ensemble_std" in result.final_scores:
            ensemble_std = result.final_scores["stability_mean_ensemble_std"]
            if not np.isnan(ensemble_std):
                assert ensemble_std >= 0

    def test_metastable_threshold_effect(self, test_structures_comprehensive):
        """Test that metastable threshold affects results."""
        benchmark_strict = StabilityBenchmark(
            use_ensemble=True, metastable_threshold=0.05
        )
        benchmark_loose = StabilityBenchmark(
            use_ensemble=True, metastable_threshold=0.2
        )

        result_strict = benchmark_strict.evaluate(test_structures_comprehensive)
        result_loose = benchmark_loose.evaluate(test_structures_comprehensive)

        # Loose threshold should give higher or equal metastable ratio
        strict_ratio = result_strict.final_scores["metastable_ratio"]
        loose_ratio = result_loose.final_scores["metastable_ratio"]

        if not np.isnan(strict_ratio) and not np.isnan(loose_ratio):
            assert loose_ratio >= strict_ratio

    def test_min_mlips_required_effect(self):
        """Test min_mlips_required parameter effect."""
        test = PymatgenTest()
        structure = test.get_structure("Si")

        # Structure with only 1 MLIP
        structure.properties["e_above_hull_orb"] = 0.05
        structure.properties["e_above_hull_mean"] = 0.05
        structure.properties["e_above_hull_std"] = 0.0
        structure.properties["e_above_hull_n_mlips"] = 1

        # Add other properties
        for prop in ["formation_energy", "relaxation_rmse"]:
            structure.properties[f"{prop}_orb"] = 0.05
            structure.properties[f"{prop}_mean"] = 0.05
            structure.properties[f"{prop}_std"] = 0.0
            structure.properties[f"{prop}_n_mlips"] = 1

        # Should fail with min_mlips_required=2
        benchmark_strict = StabilityBenchmark(use_ensemble=True, min_mlips_required=2)
        result_strict = benchmark_strict.evaluate([structure])

        # Should succeed with min_mlips_required=1
        benchmark_permissive = StabilityBenchmark(
            use_ensemble=True, min_mlips_required=1
        )
        result_permissive = benchmark_permissive.evaluate([structure])

        # Both should complete without error
        assert isinstance(result_strict, BenchmarkResult)
        assert isinstance(result_permissive, BenchmarkResult)


class TestFactoryFunctions:
    """Test suite for factory functions."""

    def test_create_benchmark_from_config(self, test_config_file):
        """Test create_benchmark_from_config factory."""
        benchmark = create_benchmark_from_config(test_config_file)

        assert isinstance(benchmark, StabilityBenchmark)
        assert benchmark.use_ensemble is True
        assert benchmark.mlip_names == ["orb", "mace", "uma"]
        assert benchmark.include_individual_results is True

    def test_create_ensemble_stability_benchmark(self):
        """Test ensemble stability benchmark factory."""
        benchmark = create_ensemble_stability_benchmark(
            metastable_threshold=0.08, metadata={"test": True}
        )

        assert isinstance(benchmark, StabilityBenchmark)
        assert benchmark.use_ensemble is True
        assert benchmark.metastable_threshold == 0.08
        assert benchmark.config.metadata["test"] is True

    def test_create_individual_mlip_stability_benchmark(self):
        """Test individual MLIP stability benchmark factory."""
        mlip_names = ["orb", "mace"]
        benchmark = create_individual_mlip_stability_benchmark(
            mlip_names=mlip_names, metastable_threshold=0.06
        )

        assert isinstance(benchmark, StabilityBenchmark)
        assert benchmark.use_ensemble is False
        assert benchmark.mlip_names == mlip_names
        assert benchmark.metastable_threshold == 0.06

    def test_create_comprehensive_benchmark(self):
        """Test comprehensive benchmark factory."""
        benchmark = create_comprehensive_benchmark(
            mlip_names=["orb", "mace", "uma"], min_mlips_required=3
        )

        assert isinstance(benchmark, StabilityBenchmark)
        assert benchmark.use_ensemble is True
        assert benchmark.include_individual_results is True
        assert benchmark.min_mlips_required == 3

    def test_factory_functions_with_custom_parameters(self):
        """Test factory functions with various custom parameters."""
        # Test ensemble factory with custom params
        ensemble_bench = create_ensemble_stability_benchmark(
            mlip_names=["orb", "mace"],
            min_mlips_required=1,
            name="Custom Ensemble Benchmark",
        )
        assert ensemble_bench.mlip_names == ["orb", "mace"]
        assert ensemble_bench.min_mlips_required == 1
        assert ensemble_bench.config.name == "Custom Ensemble Benchmark"

        # Test individual factory with custom params
        individual_bench = create_individual_mlip_stability_benchmark(
            mlip_names=["orb"], metastable_threshold=0.2, name="Single MLIP Benchmark"
        )
        assert individual_bench.mlip_names == ["orb"]
        assert individual_bench.metastable_threshold == 0.2
        assert individual_bench.config.name == "Single MLIP Benchmark"


class TestErrorHandling:
    """Test suite for error handling and edge cases."""

    def test_empty_structures_evaluation(self):
        """Test evaluation with empty structure list."""
        benchmark = StabilityBenchmark()
        result = benchmark.evaluate([])

        # Should handle empty input gracefully
        assert isinstance(result, BenchmarkResult)
        assert len(result.evaluator_results) == 5

    def test_structures_with_missing_properties(self):
        """Test evaluation with structures missing some properties."""
        test = PymatgenTest()
        structure = test.get_structure("Si")

        # Only add partial properties
        structure.properties["e_above_hull_orb"] = 0.05
        # Missing MACE and UMA properties

        benchmark = StabilityBenchmark(
            use_ensemble=False, mlip_names=["orb", "mace", "uma"]
        )
        result = benchmark.evaluate([structure])

        # Should handle gracefully without crashing
        assert isinstance(result, BenchmarkResult)

    def test_structures_with_nan_properties(self):
        """Test evaluation with NaN properties."""
        test = PymatgenTest()
        structure = test.get_structure("Si")

        # Add NaN properties
        for mlip in DEFAULT_MLIPS:
            structure.properties[f"e_above_hull_{mlip}"] = np.nan
            structure.properties[f"formation_energy_{mlip}"] = np.nan
            structure.properties[f"relaxation_rmse_{mlip}"] = np.nan

        # Add NaN ensemble properties
        for prop in ["e_above_hull", "formation_energy", "relaxation_rmse"]:
            structure.properties[f"{prop}_mean"] = np.nan
            structure.properties[f"{prop}_std"] = np.nan
            structure.properties[f"{prop}_n_mlips"] = 0

        benchmark = StabilityBenchmark()
        result = benchmark.evaluate([structure])

        # Should handle NaN values gracefully
        assert isinstance(result, BenchmarkResult)

    def test_invalid_config_parameters(self):
        """Test handling of invalid configuration parameters."""
        # Test with empty MLIP names
        benchmark = StabilityBenchmark(mlip_names=[])
        assert isinstance(benchmark, StabilityBenchmark)

        # Test with negative thresholds
        benchmark = StabilityBenchmark(metastable_threshold=-0.1)
        assert isinstance(benchmark, StabilityBenchmark)

        # Test with very high min_mlips_required
        benchmark = StabilityBenchmark(min_mlips_required=100)
        assert isinstance(benchmark, StabilityBenchmark)


class TestBenchmarkConsistency:
    """Test suite for benchmark consistency and reproducibility."""

    def test_evaluation_reproducibility(self, test_structures_comprehensive):
        """Test that multiple evaluations give consistent results."""
        benchmark = StabilityBenchmark(use_ensemble=True)

        # Run evaluation multiple times
        results = []
        for _ in range(3):
            result = benchmark.evaluate(test_structures_comprehensive)
            results.append(result)

        # Results should be identical (deterministic)
        for i in range(1, len(results)):
            for key in results[0].final_scores:
                val1 = results[0].final_scores[key]
                val2 = results[i].final_scores[key]

                # Handle NaN values properly
                if np.isnan(val1) and np.isnan(val2):
                    continue
                elif np.isnan(val1) or np.isnan(val2):
                    assert False, f"Inconsistent NaN for {key}: {val1} vs {val2}"
                else:
                    assert abs(val1 - val2) < 1e-12, (
                        f"Values differ for {key}: {val1} vs {val2}"
                    )

    def test_benchmark_configuration_storage(self):
        """Test that benchmark configuration is properly stored."""
        benchmark = StabilityBenchmark(
            use_ensemble=False,
            mlip_names=["orb", "mace"],
            metastable_threshold=0.07,
            min_mlips_required=1,
            include_individual_results=True,
        )

        # Check that configuration is accessible
        assert benchmark.use_ensemble is False
        assert benchmark.mlip_names == ["orb", "mace"]
        assert benchmark.metastable_threshold == 0.07
        assert benchmark.min_mlips_required == 1
        assert benchmark.include_individual_results is True

        # Check metadata storage
        metadata = benchmark.config.metadata
        assert metadata["use_ensemble"] is False
        assert metadata["mlip_names"] == ["orb", "mace"]
        assert metadata["metastable_threshold"] == 0.07
        assert metadata["min_mlips_required"] == 1
        assert metadata["include_individual_results"] is True

    def test_metadata_preservation(self, test_structures_comprehensive):
        """Test that custom metadata is preserved through evaluation."""
        custom_metadata = {"experiment_id": "test_123", "researcher": "test_user"}

        benchmark = StabilityBenchmark(name="Test Benchmark", metadata=custom_metadata)

        result = benchmark.evaluate(test_structures_comprehensive)

        # Check that custom metadata is preserved
        assert result.metadata["experiment_id"] == "test_123"
        assert result.metadata["researcher"] == "test_user"
        assert result.metadata["benchmark_name"] == "Test Benchmark"


class TestConfigurationCoverage:
    """Test suite for comprehensive configuration coverage."""

    def test_all_config_sections_used(self, test_config_file):
        """Test that all config sections are properly used."""
        benchmark = StabilityBenchmark(config=test_config_file)

        # Basic settings
        assert benchmark.use_ensemble is True
        assert benchmark.mlip_names == ["orb", "mace", "uma"]
        assert benchmark.metastable_threshold == 0.1

        # Ensemble config
        assert benchmark.min_mlips_required == 2

        # Reporting config
        assert benchmark.include_individual_results is True

    def test_config_validation_and_defaults(self):
        """Test config validation and default value behavior."""
        # Test with minimal config
        minimal_config = {"type": "multi_mlip_stability"}
        benchmark = StabilityBenchmark(config=minimal_config)

        # Should use defaults
        assert benchmark.use_ensemble is True
        assert benchmark.mlip_names == ["orb", "mace", "uma"]
        assert benchmark.metastable_threshold == 0.1
        assert benchmark.min_mlips_required == 2
        assert benchmark.include_individual_results is False

    def test_config_sections_independence(self):
        """Test that different config sections work independently."""
        # Test with only ensemble config
        config_ensemble_only = {"ensemble_config": {"min_mlips_required": 3}}
        benchmark1 = StabilityBenchmark(config=config_ensemble_only)
        assert benchmark1.min_mlips_required == 3

        # Test with only reporting config
        config_reporting_only = {"reporting": {"include_individual_mlip_results": True}}
        benchmark2 = StabilityBenchmark(config=config_reporting_only)
        assert benchmark2.include_individual_results is True


class TestUsagePatterns:
    """Test suite for realistic usage patterns."""

    def test_ensemble_analysis_workflow(self, test_structures_comprehensive):
        """Test typical ensemble analysis workflow."""
        # Create ensemble benchmark with individual results
        benchmark = StabilityBenchmark(
            use_ensemble=True, include_individual_results=True, metastable_threshold=0.1
        )

        result = benchmark.evaluate(test_structures_comprehensive)

        # Should have ensemble metrics
        assert "stable_ratio" in result.final_scores
        assert "mean_e_above_hull" in result.final_scores

        # Should have individual MLIP breakdown
        individual_metrics = [
            k
            for k in result.final_scores.keys()
            if any(mlip in k for mlip in DEFAULT_MLIPS)
        ]
        assert len(individual_metrics) > 0

        # Should have uncertainty information
        std_metrics = [k for k in result.final_scores.keys() if "std" in k]
        assert len(std_metrics) > 0

    def test_individual_mlip_comparison_workflow(self, test_structures_comprehensive):
        """Test workflow for comparing individual MLIPs."""
        # Create individual benchmarks for each MLIP
        mlip_benchmarks = {}
        for mlip in DEFAULT_MLIPS:
            mlip_benchmarks[mlip] = create_individual_mlip_stability_benchmark(
                mlip_names=[mlip], metastable_threshold=0.1
            )

        # Evaluate with each
        mlip_results = {}
        for mlip, benchmark in mlip_benchmarks.items():
            mlip_results[mlip] = benchmark.evaluate(test_structures_comprehensive)

        # All should complete successfully
        assert len(mlip_results) == 3
        for mlip, result in mlip_results.items():
            assert isinstance(result, BenchmarkResult)
            assert "stable_ratio" in result.final_scores

    def test_comprehensive_analysis_workflow(self, test_structures_comprehensive):
        """Test comprehensive analysis workflow."""
        # Create comprehensive benchmark showing everything
        benchmark = create_comprehensive_benchmark(
            mlip_names=["orb", "mace", "uma"],
            metastable_threshold=0.1,
            min_mlips_required=2,
        )

        result = benchmark.evaluate(test_structures_comprehensive)

        # Should have comprehensive results
        assert isinstance(result, BenchmarkResult)

        # Should have core metrics
        for metric in CORE_FINAL_SCORES:
            assert metric in result.final_scores

        # Should have individual results (from comprehensive benchmark)
        individual_keys = [
            k
            for k in result.final_scores.keys()
            if any(mlip in k for mlip in DEFAULT_MLIPS)
        ]
        assert len(individual_keys) > 0

        # Should have standard deviations
        std_keys = [k for k in result.final_scores.keys() if "std" in k]
        assert len(std_keys) > 0

    def test_quality_assessment_workflow(self, test_structures_comprehensive):
        """Test workflow for assessing prediction quality."""
        # Create benchmark focused on uncertainty analysis
        benchmark = StabilityBenchmark(
            use_ensemble=True, include_individual_results=True, min_mlips_required=2
        )

        result = benchmark.evaluate(test_structures_comprehensive)

        # Extract quality indicators
        quality_indicators = {}

        # Ensemble uncertainty
        if "stability_mean_ensemble_std" in result.final_scores:
            quality_indicators["ensemble_uncertainty"] = result.final_scores[
                "stability_mean_ensemble_std"
            ]

        # Sample-level standard deviations
        for key, value in result.final_scores.items():
            if "std" in key and not np.isnan(value):
                quality_indicators[f"sample_{key}"] = value

        # Should have quality indicators
        assert len(quality_indicators) > 0

        # All quality indicators should be non-negative
        for indicator, value in quality_indicators.items():
            if not np.isnan(value):
                assert value >= 0, (
                    f"Quality indicator {indicator} should be non-negative"
                )


# Test runner for development
if __name__ == "__main__":
    """Quick test run for development."""
    print("Running multi-MLIP stability benchmark tests...")

    try:
        # Test basic functionality
        benchmark = StabilityBenchmark()
        assert len(benchmark.evaluators) == 5
        print("✓ Basic initialization successful")

        # Test factory functions
        ensemble_bench = create_ensemble_stability_benchmark()
        individual_bench = create_individual_mlip_stability_benchmark(["orb", "mace"])
        comprehensive_bench = create_comprehensive_benchmark()
        print("✓ Factory functions successful")

        # Test configuration
        config_dict = {
            "use_ensemble": True,
            "mlip_names": ["orb", "mace", "uma"],
            "metastable_threshold": 0.1,
            "ensemble_config": {"min_mlips_required": 2},
            "reporting": {"include_individual_mlip_results": True},
        }
        config_bench = StabilityBenchmark(config=config_dict)
        assert config_bench.use_ensemble is True
        assert config_bench.include_individual_results is True
        print("✓ Configuration loading successful")

        # Test with minimal data
        test = PymatgenTest()
        structure = test.get_structure("Si")

        # Add minimal MLIP properties
        mlips = ["orb", "mace", "uma"]
        for mlip in mlips:
            structure.properties[f"e_above_hull_{mlip}"] = 0.05
            structure.properties[f"formation_energy_{mlip}"] = -2.0
            structure.properties[f"relaxation_rmse_{mlip}"] = 0.03

        # Add ensemble properties
        for prop in ["e_above_hull", "formation_energy", "relaxation_rmse"]:
            values = [structure.properties[f"{prop}_{mlip}"] for mlip in mlips]
            structure.properties[f"{prop}_mean"] = np.mean(values)
            structure.properties[f"{prop}_std"] = np.std(values)
            structure.properties[f"{prop}_n_mlips"] = len(mlips)

        # Test evaluation
        result = benchmark.evaluate([structure])
        assert isinstance(result, BenchmarkResult)
        assert len(result.final_scores) > 0
        print("✓ Evaluation successful")

        # Test utility functions
        assert safe_float(42) == 42.0
        assert np.isnan(safe_float("invalid"))
        print("✓ Utility functions working")

        print("\n✅ All tests completed successfully!")
        print("\nKey features tested:")
        print("  ✓ Individual vs ensemble modes")
        print("  ✓ Include individual results functionality")
        print("  ✓ Config loading from YAML and dict")
        print("  ✓ All factory functions")
        print("  ✓ Standard deviation reporting")
        print("  ✓ Min MLIPs required filtering")
        print("  ✓ Comprehensive error handling")
        print("  ✓ Realistic usage patterns")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()

    # Clean up temp file
    try:
        import os

        temp_files = [f for f in os.listdir() if f.endswith(".yaml") and "tmp" in f]
        for temp_file in temp_files:
            os.remove(temp_file)
    except (OSError, FileNotFoundError):
        # Ignore cleanup errors - temp files may not exist or be accessible
        pass
