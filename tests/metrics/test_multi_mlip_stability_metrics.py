"""Tests for multi-MLIP stability metrics implementation.

This test suite comprehensively tests the new multi-MLIP stability metrics including
individual vs ensemble modes, individual result inclusion, and standard deviation reporting.
"""

import numpy as np
import pytest
from pymatgen.util.testing import PymatgenTest

from lemat_genbench.metrics.base import MetricResult
from lemat_genbench.metrics.multi_mlip_stability_metrics import (
    E_HullMetric,
    FormationEnergyMetric,
    MetastabilityMetric,
    RelaxationStabilityMetric,
    StabilityMetric,
    extract_ensemble_value,
    extract_individual_values,
    safe_float_convert,
)

# Test Data Constants
DEFAULT_MLIPS = ["orb", "mace", "uma"]


@pytest.fixture
def test_structures_with_multi_mlip_properties():
    """Create test structures with comprehensive multi-MLIP properties."""
    test = PymatgenTest()
    structures = [
        test.get_structure("Si"),
        test.get_structure("LiFePO4"),
        test.get_structure("CsCl"),
    ]

    # Test data: mix of stable, metastable, and unstable structures
    test_data = [
        {  # Structure 1: Stable according to all MLIPs
            "e_above_hull": {"orb": -0.01, "mace": -0.005, "uma": 0.001},
            "formation_energy": {"orb": -2.1, "mace": -2.05, "uma": -2.08},
            "relaxation_rmse": {"orb": 0.015, "mace": 0.018, "uma": 0.020},
        },
        {  # Structure 2: Mixed stability predictions (metastable/unstable)
            "e_above_hull": {"orb": 0.08, "mace": 0.12, "uma": 0.09},
            "formation_energy": {"orb": -1.2, "mace": -1.1, "uma": -1.3},
            "relaxation_rmse": {"orb": 0.025, "mace": 0.030, "uma": 0.022},
        },
        {  # Structure 3: Unstable according to all MLIPs
            "e_above_hull": {"orb": 0.25, "mace": 0.30, "uma": 0.28},
            "formation_energy": {"orb": 0.5, "mace": 0.6, "uma": 0.4},
            "relaxation_rmse": {"orb": 0.040, "mace": 0.045, "uma": 0.038},
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

        # Calculate and add ensemble statistics (as done by MultiMLIPStabilityPreprocessor)
        for property_base in ["e_above_hull", "formation_energy", "relaxation_rmse"]:
            values = [data[property_base][mlip] for mlip in DEFAULT_MLIPS]
            structure.properties[f"{property_base}_mean"] = np.mean(values)
            structure.properties[f"{property_base}_std"] = np.std(values)
            structure.properties[f"{property_base}_n_mlips"] = len(DEFAULT_MLIPS)

    return structures


@pytest.fixture
def test_structures_insufficient_mlips():
    """Create test structures with insufficient MLIPs for ensemble."""
    test = PymatgenTest()
    structure = test.get_structure("Si")

    # Only one MLIP available
    structure.properties["e_above_hull_orb"] = 0.05
    structure.properties["formation_energy_orb"] = -1.5
    structure.properties["relaxation_rmse_orb"] = 0.025

    # Ensemble properties with n_mlips = 1 (below min_mlips_required)
    structure.properties["e_above_hull_mean"] = 0.05
    structure.properties["e_above_hull_std"] = 0.0
    structure.properties["e_above_hull_n_mlips"] = 1

    structure.properties["formation_energy_mean"] = -1.5
    structure.properties["formation_energy_std"] = 0.0
    structure.properties["formation_energy_n_mlips"] = 1

    structure.properties["relaxation_rmse_mean"] = 0.025
    structure.properties["relaxation_rmse_std"] = 0.0
    structure.properties["relaxation_rmse_n_mlips"] = 1

    return [structure]


@pytest.fixture
def test_structures_high_disagreement():
    """Create test structures with high MLIP disagreement."""
    test = PymatgenTest()
    structure = test.get_structure("Si")

    # High disagreement in predictions
    e_hull_values = [-0.1, 0.2, 0.05]  # High disagreement
    fe_values = [-3.0, -1.0, -2.0]  # High disagreement
    rmse_values = [0.01, 0.08, 0.03]  # High disagreement

    for i, mlip in enumerate(DEFAULT_MLIPS):
        structure.properties[f"e_above_hull_{mlip}"] = e_hull_values[i]
        structure.properties[f"formation_energy_{mlip}"] = fe_values[i]
        structure.properties[f"relaxation_rmse_{mlip}"] = rmse_values[i]

    # Calculate ensemble properties with high std
    structure.properties["e_above_hull_mean"] = np.mean(e_hull_values)
    structure.properties["e_above_hull_std"] = np.std(e_hull_values)  # Will be high
    structure.properties["e_above_hull_n_mlips"] = 3

    structure.properties["formation_energy_mean"] = np.mean(fe_values)
    structure.properties["formation_energy_std"] = np.std(fe_values)
    structure.properties["formation_energy_n_mlips"] = 3

    structure.properties["relaxation_rmse_mean"] = np.mean(rmse_values)
    structure.properties["relaxation_rmse_std"] = np.std(rmse_values)
    structure.properties["relaxation_rmse_n_mlips"] = 3

    return [structure]


class TestUtilityFunctions:
    """Test suite for utility functions."""

    def test_safe_float_convert(self):
        """Test safe_float_convert function."""
        # Valid conversions
        assert safe_float_convert(1.0) == 1.0
        assert safe_float_convert(42) == 42.0
        assert safe_float_convert("3.14") == 3.14

        # Invalid conversions should return NaN
        assert np.isnan(safe_float_convert(None))
        assert np.isnan(safe_float_convert("invalid"))
        assert np.isnan(safe_float_convert([1, 2, 3]))
        assert np.isnan(safe_float_convert(np.inf))

    def test_extract_individual_values(
        self, test_structures_with_multi_mlip_properties
    ):
        """Test extract_individual_values function."""
        structure = test_structures_with_multi_mlip_properties[0]

        # Test e_above_hull extraction
        values = extract_individual_values(structure, DEFAULT_MLIPS, "e_above_hull")
        assert len(values) == 3
        assert values["orb"] == -0.01
        assert values["mace"] == -0.005
        assert values["uma"] == 0.001

        # Test formation_energy extraction
        fe_values = extract_individual_values(
            structure, DEFAULT_MLIPS, "formation_energy"
        )
        assert len(fe_values) == 3
        assert fe_values["orb"] == -2.1

    def test_extract_individual_values_missing_data(self):
        """Test extract_individual_values with missing data."""
        test = PymatgenTest()
        structure = test.get_structure("Si")

        # Only add data for ORB
        structure.properties["e_above_hull_orb"] = 0.05

        values = extract_individual_values(structure, DEFAULT_MLIPS, "e_above_hull")
        assert len(values) == 3
        assert values["orb"] == 0.05
        assert np.isnan(values["mace"])
        assert np.isnan(values["uma"])

    def test_extract_ensemble_value(self, test_structures_with_multi_mlip_properties):
        """Test extract_ensemble_value function."""
        structure = test_structures_with_multi_mlip_properties[0]

        # Test successful extraction
        mean_val, std_val = extract_ensemble_value(
            structure, "e_above_hull", min_mlips_required=2
        )
        assert not np.isnan(mean_val)
        assert not np.isnan(std_val)

        # Calculate expected values
        expected_mean = np.mean([-0.01, -0.005, 0.001])
        expected_std = np.std([-0.01, -0.005, 0.001])
        assert abs(mean_val - expected_mean) < 1e-10
        assert abs(std_val - expected_std) < 1e-10

    def test_extract_ensemble_value_insufficient_mlips(
        self, test_structures_insufficient_mlips
    ):
        """Test extract_ensemble_value with insufficient MLIPs."""
        structure = test_structures_insufficient_mlips[0]

        # Should return NaN when insufficient MLIPs
        mean_val, std_val = extract_ensemble_value(
            structure, "e_above_hull", min_mlips_required=2
        )
        assert np.isnan(mean_val)
        assert np.isnan(std_val)

        # Should work when minimum is lowered
        mean_val, std_val = extract_ensemble_value(
            structure, "e_above_hull", min_mlips_required=1
        )
        assert not np.isnan(mean_val)
        assert mean_val == 0.05


class TestStabilityMetric:
    """Test suite for StabilityMetric."""

    def test_initialization_default(self):
        """Test default initialization."""
        metric = StabilityMetric()
        assert metric.use_ensemble is True
        assert metric.mlip_names == ["orb", "mace", "uma"]
        assert metric.min_mlips_required == 2
        assert metric.include_individual_results is False

    def test_initialization_custom(self):
        """Test custom initialization."""
        metric = StabilityMetric(
            use_ensemble=False,
            mlip_names=["orb", "mace"],
            min_mlips_required=1,
            include_individual_results=True,
            name="Custom Stability",
        )
        assert metric.use_ensemble is False
        assert metric.mlip_names == ["orb", "mace"]
        assert metric.min_mlips_required == 1
        assert metric.include_individual_results is True
        assert metric.name == "Custom Stability"

    def test_ensemble_mode_basic(self, test_structures_with_multi_mlip_properties):
        """Test ensemble mode functionality."""
        metric = StabilityMetric(use_ensemble=True)
        result = metric.compute(test_structures_with_multi_mlip_properties)

        assert isinstance(result, MetricResult)
        assert result.primary_metric == "stable_ratio"
        assert "stable_ratio" in result.metrics
        assert 0.0 <= result.metrics["stable_ratio"] <= 1.0
        assert "mean_e_above_hull" in result.metrics
        assert "std_e_above_hull" in result.metrics

    def test_individual_mode_basic(self, test_structures_with_multi_mlip_properties):
        """Test individual mode functionality."""
        metric = StabilityMetric(use_ensemble=False, mlip_names=["orb", "mace", "uma"])
        result = metric.compute(test_structures_with_multi_mlip_properties)

        assert isinstance(result, MetricResult)
        assert result.primary_metric == "stable_ratio"
        assert "stable_ratio" in result.metrics
        assert 0.0 <= result.metrics["stable_ratio"] <= 1.0

    def test_include_individual_results_ensemble_mode(
        self, test_structures_with_multi_mlip_properties
    ):
        """Test ensemble mode with individual results included."""
        metric = StabilityMetric(use_ensemble=True, include_individual_results=True)
        result = metric.compute(test_structures_with_multi_mlip_properties)

        # Should have ensemble metrics
        assert "stable_ratio" in result.metrics
        assert "mean_e_above_hull" in result.metrics
        assert "std_e_above_hull" in result.metrics

        # Should also have individual MLIP metrics
        for mlip_name in DEFAULT_MLIPS:
            assert f"stable_ratio_{mlip_name}" in result.metrics
            assert f"mean_e_above_hull_{mlip_name}" in result.metrics
            assert f"std_e_above_hull_{mlip_name}" in result.metrics

    def test_include_individual_results_individual_mode(
        self, test_structures_with_multi_mlip_properties
    ):
        """Test individual mode automatically includes individual results."""
        metric = StabilityMetric(use_ensemble=False, mlip_names=["orb", "mace"])
        result = metric.compute(test_structures_with_multi_mlip_properties)

        # Should have aggregated metrics
        assert "stable_ratio" in result.metrics

        # Should have individual MLIP metrics
        assert "stable_ratio_orb" in result.metrics
        assert "stable_ratio_mace" in result.metrics

    def test_ensemble_uncertainty_reporting(self, test_structures_high_disagreement):
        """Test ensemble uncertainty reporting."""
        metric = StabilityMetric(use_ensemble=True)
        result = metric.compute(test_structures_high_disagreement)

        # Should have ensemble uncertainty metrics
        assert "mean_ensemble_std" in result.metrics
        assert "std_ensemble_std" in result.metrics

        # High disagreement should result in high std
        mean_std = result.metrics["mean_ensemble_std"]
        assert not np.isnan(mean_std)
        assert mean_std > 0.05  # Should be reasonably high due to disagreement

    def test_compute_structure_ensemble_mode(
        self, test_structures_with_multi_mlip_properties
    ):
        """Test compute_structure method in ensemble mode."""
        metric = StabilityMetric(use_ensemble=True, include_individual_results=True)
        structure = test_structures_with_multi_mlip_properties[0]

        result = metric.compute_structure(structure, **metric._get_compute_attributes())

        assert isinstance(result, dict)
        assert "value" in result
        assert "std" in result
        assert "value_orb" in result
        assert "value_mace" in result
        assert "value_uma" in result

    def test_compute_structure_individual_mode(
        self, test_structures_with_multi_mlip_properties
    ):
        """Test compute_structure method in individual mode."""
        metric = StabilityMetric(use_ensemble=False, mlip_names=["orb", "mace"])
        structure = test_structures_with_multi_mlip_properties[0]

        result = metric.compute_structure(structure, **metric._get_compute_attributes())

        assert isinstance(result, dict)
        assert "value" in result
        assert "std" in result
        assert "value_orb" in result
        assert "value_mace" in result

    def test_stable_ratio_calculation(self, test_structures_with_multi_mlip_properties):
        """Test stable ratio calculation accuracy."""
        metric = StabilityMetric(use_ensemble=True)
        result = metric.compute(test_structures_with_multi_mlip_properties)

        # Structure 1: mean ≈ -0.0047 (stable, ≤ 0)
        # Structure 2: mean = 0.097 (unstable, > 0)
        # Structure 3: mean = 0.277 (unstable, > 0)
        # Expected stable_ratio = 1/3 ≈ 0.333
        expected_stable_ratio = 1 / 3
        actual_stable_ratio = result.metrics["stable_ratio"]

        assert abs(actual_stable_ratio - expected_stable_ratio) < 0.05

    def test_min_mlips_required_filtering(self, test_structures_insufficient_mlips):
        """Test min_mlips_required parameter filtering."""
        # Should fail with default min_mlips_required=2
        metric_strict = StabilityMetric(use_ensemble=True, min_mlips_required=2)
        result_strict = metric_strict.compute(test_structures_insufficient_mlips)

        # Should return NaN for insufficient MLIPs
        assert np.isnan(result_strict.individual_values[0]["value"])

        # Should work with min_mlips_required=1
        metric_permissive = StabilityMetric(use_ensemble=True, min_mlips_required=1)
        result_permissive = metric_permissive.compute(
            test_structures_insufficient_mlips
        )

        # Should return valid value
        assert not np.isnan(result_permissive.individual_values[0]["value"])

    def test_empty_structures(self):
        """Test handling of empty structure list."""
        metric = StabilityMetric()
        result = metric.compute([])

        assert isinstance(result, MetricResult)
        assert len(result.individual_values) == 0
        # When no structures are provided, BaseMetric creates a default result
        # with just the metric name, so n_valid_structures won't be present
        assert result.primary_metric == "StabilityMetric"


class TestMetastabilityMetric:
    """Test suite for MetastabilityMetric."""

    def test_basic_functionality(self, test_structures_with_multi_mlip_properties):
        """Test basic MetastabilityMetric functionality."""
        metric = MetastabilityMetric(use_ensemble=True, metastable_threshold=0.1)
        result = metric.compute(test_structures_with_multi_mlip_properties)

        assert isinstance(result, MetricResult)
        assert result.primary_metric == "metastable_ratio"
        assert "metastable_ratio" in result.metrics
        assert 0.0 <= result.metrics["metastable_ratio"] <= 1.0

    def test_metastable_ratio_calculation(
        self, test_structures_with_multi_mlip_properties
    ):
        """Test metastable ratio calculation accuracy."""
        metric = MetastabilityMetric(use_ensemble=True, metastable_threshold=0.1)
        result = metric.compute(test_structures_with_multi_mlip_properties)

        # Structure 1: mean ≈ -0.0047 (≤ 0.1, metastable)
        # Structure 2: mean = 0.097 (≤ 0.1, metastable)
        # Structure 3: mean = 0.277 (> 0.1, not metastable)
        # Expected metastable_ratio = 2/3 ≈ 0.667
        expected_ratio = 2 / 3
        actual_ratio = result.metrics["metastable_ratio"]

        assert abs(actual_ratio - expected_ratio) < 0.05

    def test_threshold_variation(self, test_structures_with_multi_mlip_properties):
        """Test metastability with different thresholds."""
        metric_strict = MetastabilityMetric(
            use_ensemble=True, metastable_threshold=0.05
        )
        metric_loose = MetastabilityMetric(use_ensemble=True, metastable_threshold=0.3)

        result_strict = metric_strict.compute(
            test_structures_with_multi_mlip_properties
        )
        result_loose = metric_loose.compute(test_structures_with_multi_mlip_properties)

        # Loose threshold should give higher metastable ratio
        assert (
            result_loose.metrics["metastable_ratio"]
            >= result_strict.metrics["metastable_ratio"]
        )

    def test_include_individual_results(
        self, test_structures_with_multi_mlip_properties
    ):
        """Test individual results inclusion."""
        metric = MetastabilityMetric(
            use_ensemble=True, metastable_threshold=0.1, include_individual_results=True
        )
        result = metric.compute(test_structures_with_multi_mlip_properties)

        # Should have individual MLIP results
        for mlip_name in DEFAULT_MLIPS:
            assert f"metastable_ratio_{mlip_name}" in result.metrics


class TestE_HullMetric:
    """Test suite for E_HullMetric."""

    def test_basic_functionality(self, test_structures_with_multi_mlip_properties):
        """Test basic E_HullMetric functionality."""
        metric = E_HullMetric(use_ensemble=True)
        result = metric.compute(test_structures_with_multi_mlip_properties)

        assert isinstance(result, MetricResult)
        assert result.primary_metric == "mean_e_above_hull"
        assert "mean_e_above_hull" in result.metrics
        assert "std_e_above_hull" in result.metrics

    def test_mean_calculation_accuracy(
        self, test_structures_with_multi_mlip_properties
    ):
        """Test mean e_above_hull calculation."""
        metric = E_HullMetric(use_ensemble=True)
        result = metric.compute(test_structures_with_multi_mlip_properties)

        # Calculate expected mean manually
        expected_values = [
            np.mean([-0.01, -0.005, 0.001]),  # Structure 1
            np.mean([0.08, 0.12, 0.09]),  # Structure 2
            np.mean([0.25, 0.30, 0.28]),  # Structure 3
        ]
        expected_mean = np.mean(expected_values)
        actual_mean = result.metrics["mean_e_above_hull"]

        assert abs(actual_mean - expected_mean) < 0.01

    def test_lower_is_better_setting(self):
        """Test that lower_is_better is correctly set."""
        metric = E_HullMetric()
        assert metric.config.lower_is_better is True

    def test_include_individual_results(
        self, test_structures_with_multi_mlip_properties
    ):
        """Test individual results inclusion."""
        metric = E_HullMetric(use_ensemble=True, include_individual_results=True)
        result = metric.compute(test_structures_with_multi_mlip_properties)

        # Should have individual MLIP results
        for mlip_name in DEFAULT_MLIPS:
            assert f"mean_e_above_hull_{mlip_name}" in result.metrics
            assert f"std_e_above_hull_{mlip_name}" in result.metrics


class TestFormationEnergyMetric:
    """Test suite for FormationEnergyMetric."""

    def test_basic_functionality(self, test_structures_with_multi_mlip_properties):
        """Test basic FormationEnergyMetric functionality."""
        metric = FormationEnergyMetric(use_ensemble=True)
        result = metric.compute(test_structures_with_multi_mlip_properties)

        assert isinstance(result, MetricResult)
        assert result.primary_metric == "mean_formation_energy"
        assert "mean_formation_energy" in result.metrics
        assert "std_formation_energy" in result.metrics

    def test_ensemble_vs_individual_modes(
        self, test_structures_with_multi_mlip_properties
    ):
        """Test individual vs ensemble mode differences."""
        metric_ensemble = FormationEnergyMetric(use_ensemble=True)
        metric_individual = FormationEnergyMetric(
            use_ensemble=False, mlip_names=["orb", "mace"]
        )

        result_ensemble = metric_ensemble.compute(
            test_structures_with_multi_mlip_properties
        )
        result_individual = metric_individual.compute(
            test_structures_with_multi_mlip_properties
        )

        # Both should produce valid results
        assert not np.isnan(result_ensemble.metrics["mean_formation_energy"])
        assert not np.isnan(result_individual.metrics["mean_formation_energy"])

    def test_include_individual_results(
        self, test_structures_with_multi_mlip_properties
    ):
        """Test individual results inclusion."""
        metric = FormationEnergyMetric(
            use_ensemble=True, include_individual_results=True
        )
        result = metric.compute(test_structures_with_multi_mlip_properties)

        # Should have individual MLIP results
        for mlip_name in DEFAULT_MLIPS:
            assert f"mean_formation_energy_{mlip_name}" in result.metrics
            assert f"std_formation_energy_{mlip_name}" in result.metrics


class TestRelaxationStabilityMetric:
    """Test suite for RelaxationStabilityMetric."""

    def test_basic_functionality(self, test_structures_with_multi_mlip_properties):
        """Test basic RelaxationStabilityMetric functionality."""
        metric = RelaxationStabilityMetric(use_ensemble=True)
        result = metric.compute(test_structures_with_multi_mlip_properties)

        assert isinstance(result, MetricResult)
        assert result.primary_metric == "mean_relaxation_RMSE"
        assert "mean_relaxation_RMSE" in result.metrics
        assert "std_relaxation_RMSE" in result.metrics

    def test_rmse_value_ranges(self, test_structures_with_multi_mlip_properties):
        """Test that RMSE values are positive and reasonable."""
        metric = RelaxationStabilityMetric(use_ensemble=True)
        result = metric.compute(test_structures_with_multi_mlip_properties)

        # RMSE should be positive
        mean_rmse = result.metrics["mean_relaxation_RMSE"]
        assert mean_rmse > 0
        assert 0.0 < mean_rmse < 1.0

    def test_lower_is_better_setting(self):
        """Test that lower_is_better is correctly set for RMSE."""
        metric = RelaxationStabilityMetric()
        assert metric.config.lower_is_better is True

    def test_include_individual_results(
        self, test_structures_with_multi_mlip_properties
    ):
        """Test individual results inclusion."""
        metric = RelaxationStabilityMetric(
            use_ensemble=True, include_individual_results=True
        )
        result = metric.compute(test_structures_with_multi_mlip_properties)

        # Should have individual MLIP results
        for mlip_name in DEFAULT_MLIPS:
            assert f"mean_relaxation_RMSE_{mlip_name}" in result.metrics
            assert f"std_relaxation_RMSE_{mlip_name}" in result.metrics


class TestMetricCompatibility:
    """Test compatibility with BaseMetric interface."""

    def test_base_metric_interface_compliance(
        self, test_structures_with_multi_mlip_properties
    ):
        """Test that all metrics properly implement BaseMetric interface."""
        metrics = [
            StabilityMetric(),
            MetastabilityMetric(),
            E_HullMetric(),
            FormationEnergyMetric(),
            RelaxationStabilityMetric(),
        ]

        for metric in metrics:
            # Test required methods exist
            assert hasattr(metric, "compute")
            assert hasattr(metric, "_get_compute_attributes")
            assert hasattr(metric, "aggregate_results")
            assert hasattr(metric, "compute_structure")

            # Test compute method returns proper result
            result = metric.compute(test_structures_with_multi_mlip_properties)
            assert isinstance(result, MetricResult)

    def test_primary_metric_definitions(
        self, test_structures_with_multi_mlip_properties
    ):
        """Test that all metrics define primary metrics correctly."""
        expected_primary_metrics = [
            (StabilityMetric(), "stable_ratio"),
            (MetastabilityMetric(), "metastable_ratio"),
            (E_HullMetric(), "mean_e_above_hull"),
            (FormationEnergyMetric(), "mean_formation_energy"),
            (RelaxationStabilityMetric(), "mean_relaxation_RMSE"),
        ]

        for metric, expected_primary in expected_primary_metrics:
            result = metric.compute(test_structures_with_multi_mlip_properties)
            assert result.primary_metric == expected_primary
            assert expected_primary in result.metrics

    def test_reproducibility(self, test_structures_with_multi_mlip_properties):
        """Test that computation results are reproducible."""
        metric = StabilityMetric(use_ensemble=True)

        # Run multiple times
        results = []
        for _ in range(3):
            result = metric.compute(test_structures_with_multi_mlip_properties)
            results.append(result)

        # Results should be identical
        for i in range(1, len(results)):
            assert (
                results[0].metrics["stable_ratio"] == results[i].metrics["stable_ratio"]
            )


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_malformed_property_handling(self):
        """Test handling of malformed property data."""
        test = PymatgenTest()
        structure = test.get_structure("Si")

        # Add malformed properties
        structure.properties["e_above_hull_orb"] = "invalid_string"
        structure.properties["e_above_hull_mace"] = [1, 2, 3]
        structure.properties["e_above_hull_uma"] = None

        metric = StabilityMetric(use_ensemble=False, mlip_names=["orb", "mace", "uma"])
        result = metric.compute([structure])

        # Should handle gracefully and return NaN
        assert isinstance(result, MetricResult)
        assert np.isnan(result.individual_values[0]["value"])

    def test_extreme_threshold_values(self, test_structures_with_multi_mlip_properties):
        """Test behavior with extreme min_mlips_required values."""
        # Test with very high threshold
        metric = StabilityMetric(use_ensemble=True, min_mlips_required=10)
        result = metric.compute(test_structures_with_multi_mlip_properties)

        # Should return NaN for all structures (not enough MLIPs)
        for val in result.individual_values:
            assert np.isnan(val["value"])

    def test_duplicate_mlip_names(self, test_structures_with_multi_mlip_properties):
        """Test handling of duplicate MLIP names."""
        metric = StabilityMetric(use_ensemble=False, mlip_names=["orb", "orb", "mace"])
        result = metric.compute(test_structures_with_multi_mlip_properties)

        # Should handle duplicates gracefully
        assert isinstance(result, MetricResult)

    def test_empty_mlip_names_list(self, test_structures_with_multi_mlip_properties):
        """Test handling of empty MLIP names list."""
        metric = StabilityMetric(use_ensemble=False, mlip_names=[])
        result = metric.compute(test_structures_with_multi_mlip_properties)

        # Should handle gracefully
        assert isinstance(result, MetricResult)


class TestNumericalStability:
    """Test numerical stability and precision."""

    def test_floating_point_precision(self):
        """Test handling of floating-point precision issues."""
        test = PymatgenTest()
        structure = test.get_structure("Si")

        # Add very small values
        tiny_values = [1e-15, -1e-15, 1e-14]
        for i, mlip in enumerate(["orb", "mace", "uma"]):
            structure.properties[f"e_above_hull_{mlip}"] = tiny_values[i]

        # Add ensemble properties
        structure.properties["e_above_hull_mean"] = np.mean(tiny_values)
        structure.properties["e_above_hull_std"] = np.std(tiny_values)
        structure.properties["e_above_hull_n_mlips"] = 3

        metric = StabilityMetric(use_ensemble=True)
        result = metric.compute([structure])

        # Should handle tiny values
        assert isinstance(result, MetricResult)
        assert not np.isnan(result.individual_values[0]["value"])

    def test_large_value_handling(self):
        """Test handling of very large values."""
        test = PymatgenTest()
        structure = test.get_structure("Si")

        # Add large values
        large_values = [1e6, 1e7, 1e8]
        for i, mlip in enumerate(["orb", "mace", "uma"]):
            structure.properties[f"e_above_hull_{mlip}"] = large_values[i]

        structure.properties["e_above_hull_mean"] = np.mean(large_values)
        structure.properties["e_above_hull_std"] = np.std(large_values)
        structure.properties["e_above_hull_n_mlips"] = 3

        metric = StabilityMetric(use_ensemble=True)
        result = metric.compute([structure])

        # Should handle large values
        assert isinstance(result, MetricResult)
        assert not np.isnan(result.individual_values[0]["value"])


class TestIntegrationPatterns:
    """Test realistic integration patterns."""

    def test_typical_usage_pattern(self, test_structures_with_multi_mlip_properties):
        """Test typical usage pattern in benchmark context."""
        # Simulate how metrics would be used in a benchmark
        metrics = {
            "stability": StabilityMetric(
                use_ensemble=True, include_individual_results=True
            ),
            "metastability": MetastabilityMetric(
                use_ensemble=True, include_individual_results=True
            ),
            "e_hull": E_HullMetric(use_ensemble=True, include_individual_results=True),
            "formation_energy": FormationEnergyMetric(
                use_ensemble=True, include_individual_results=True
            ),
            "relaxation": RelaxationStabilityMetric(
                use_ensemble=True, include_individual_results=True
            ),
        }

        results = {}
        for name, metric in metrics.items():
            results[name] = metric.compute(test_structures_with_multi_mlip_properties)

        # All should complete successfully
        assert len(results) == 5
        for result in results.values():
            assert isinstance(result, MetricResult)
            assert not np.isnan(result.metrics[result.primary_metric])

            # Should have individual MLIP results
            for mlip_name in DEFAULT_MLIPS:
                individual_metric_key = f"{result.primary_metric}_{mlip_name}"
                if individual_metric_key in result.metrics:
                    assert isinstance(
                        result.metrics[individual_metric_key], (int, float)
                    )

    def test_ensemble_vs_individual_comparison(
        self, test_structures_with_multi_mlip_properties
    ):
        """Test realistic ensemble vs individual comparison."""
        # Ensemble approach
        ensemble_stability = StabilityMetric(use_ensemble=True)
        ensemble_result = ensemble_stability.compute(
            test_structures_with_multi_mlip_properties
        )

        # Individual MLIP approach
        individual_stability = StabilityMetric(
            use_ensemble=False, mlip_names=["orb", "mace"]
        )
        individual_result = individual_stability.compute(
            test_structures_with_multi_mlip_properties
        )

        # Both should work and produce reasonable results
        assert not np.isnan(ensemble_result.metrics["stable_ratio"])
        assert not np.isnan(individual_result.metrics["stable_ratio"])

        # Results should be in reasonable relationship
        assert 0.0 <= ensemble_result.metrics["stable_ratio"] <= 1.0
        assert 0.0 <= individual_result.metrics["stable_ratio"] <= 1.0


# Test runner for development
if __name__ == "__main__":
    """Quick test run for development and debugging."""
    print("Running multi-MLIP stability metrics tests...")

    try:
        # Create minimal test data
        test = PymatgenTest()
        structure = test.get_structure("Si")

        # Add properties that match actual implementation expectations
        mlips = ["orb", "mace", "uma"]
        test_data = {
            "e_above_hull": 0.05,
            "formation_energy": -2.0,
            "relaxation_rmse": 0.03,
        }

        for mlip in mlips:
            for prop, value in test_data.items():
                structure.properties[f"{prop}_{mlip}"] = value + np.random.normal(
                    0, 0.01
                )

        # Add ensemble properties
        for prop, base_value in test_data.items():
            values = [structure.properties[f"{prop}_{mlip}"] for mlip in mlips]
            structure.properties[f"{prop}_mean"] = np.mean(values)
            structure.properties[f"{prop}_std"] = np.std(values)
            structure.properties[f"{prop}_n_mlips"] = len(mlips)

        structures = [structure]

        # Test all metrics
        metrics_to_test = [
            (
                "StabilityMetric",
                StabilityMetric(use_ensemble=True, include_individual_results=True),
            ),
            (
                "MetastabilityMetric",
                MetastabilityMetric(use_ensemble=True, include_individual_results=True),
            ),
            (
                "E_HullMetric",
                E_HullMetric(use_ensemble=True, include_individual_results=True),
            ),
            (
                "FormationEnergyMetric",
                FormationEnergyMetric(
                    use_ensemble=True, include_individual_results=True
                ),
            ),
            (
                "RelaxationStabilityMetric",
                RelaxationStabilityMetric(
                    use_ensemble=True, include_individual_results=True
                ),
            ),
        ]

        print("Testing individual metrics...")
        for metric_name, metric in metrics_to_test:
            try:
                result = metric.compute(structures)
                primary_value = result.metrics[result.primary_metric]
                print(f"✓ {metric_name}: {result.primary_metric} = {primary_value:.4f}")

                # Check for individual MLIP results
                individual_metrics = [
                    k for k in result.metrics.keys() if any(mlip in k for mlip in mlips)
                ]
                if individual_metrics:
                    print(f"  Individual MLIP metrics: {len(individual_metrics)} found")

            except Exception as e:
                print(f"✗ {metric_name} failed: {e}")

        print("\nTesting utility functions...")

        # Test utility functions
        assert safe_float_convert(42) == 42.0
        assert np.isnan(safe_float_convert("invalid"))
        print("✓ safe_float_convert working")

        values = extract_individual_values(structure, mlips, "e_above_hull")
        assert len(values) == 3
        print("✓ extract_individual_values working")

        mean_val, std_val = extract_ensemble_value(
            structure, "e_above_hull", min_mlips_required=2
        )
        assert not np.isnan(mean_val)
        print("✓ extract_ensemble_value working")

        print("\n✓ All tests completed successfully!")
        print("Multi-MLIP stability metrics implementation validated!")

    except Exception as e:
        print(f"✗ Test validation failed: {e}")
        import traceback

        traceback.print_exc()
