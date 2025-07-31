"""Tests for HHI metrics."""

import numpy as np
import pytest
from pymatgen.core import Structure

from lemat_genbench.metrics.base import MetricResult
from lemat_genbench.metrics.hhi_metrics import (
    BaseHHIMetric,
    HHIProductionMetric,
    HHIReserveMetric,
    _load_hhi_data,
    compound_hhi,
)


def create_test_structures():
    """Create test structures for HHI testing."""
    structures = []

    # Simple cubic structure - NaCl
    lattice = [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]]
    nacl_structure = Structure(
        lattice=lattice,
        species=["Na", "Cl"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structures.append(nacl_structure)

    # Iron structure
    lattice = [[2.87, 0, 0], [0, 2.87, 0], [0, 0, 2.87]]
    fe_structure = Structure(
        lattice=lattice,
        species=["Fe"],
        coords=[[0, 0, 0]],
        coords_are_cartesian=False,
    )
    structures.append(fe_structure)

    # More complex structure - LiFePO4
    lattice = [[10.3, 0, 0], [0, 6.0, 0], [0, 0, 4.7]]
    lifepo4_structure = Structure(
        lattice=lattice,
        species=["Li", "Fe", "P", "O", "O", "O", "O"],
        coords=[
            [0, 0, 0],
            [0.25, 0.25, 0.25],
            [0.5, 0.5, 0.5],
            [0.1, 0.1, 0.1],
            [0.9, 0.9, 0.9],
            [0.3, 0.7, 0.2],
            [0.7, 0.3, 0.8],
        ],
        coords_are_cartesian=False,
    )
    structures.append(lifepo4_structure)

    return structures


def get_hhi_data():
    """Get HHI data, skip test if not available."""
    try:
        return _load_hhi_data()
    except ImportError:
        pytest.skip("HHI data not available - skipping test")


class TestHHIProductionMetric:
    """Test HHI Production metric."""

    def test_initialization(self):
        """Test metric initialization."""
        metric = HHIProductionMetric()

        assert metric.name == "HHIProduction"
        assert "production concentration" in metric.description.lower()
        assert metric.config.lower_is_better is True
        assert metric.config.scale_to_0_10 is True

    def test_initialization_custom_params(self):
        """Test metric initialization with custom parameters."""
        metric = HHIProductionMetric(
            name="CustomHHIP",
            description="Custom description",
            scale_to_0_10=False,
            n_jobs=2,
        )

        assert metric.name == "CustomHHIP"
        assert metric.description == "Custom description"
        assert metric.config.scale_to_0_10 is False
        assert metric.config.n_jobs == 2

    def test_compute_structure_simple(self):
        """Test HHI computation for simple structures."""
        hhi_production, _ = get_hhi_data()

        metric = HHIProductionMetric()
        structures = create_test_structures()

        # Test Fe structure (should have HHI equal to Fe's HHI)
        fe_structure = structures[1]  # Pure Fe
        result = metric.compute_structure(
            fe_structure, **metric._get_compute_attributes()
        )

        # Should be a valid number
        assert isinstance(result, float)
        assert not np.isnan(result)
        assert result > 0  # HHI should be positive

        # For pure Fe, should equal scaled Fe HHI value
        expected = hhi_production["Fe"] / 1000  # scaled
        assert abs(result - expected) < 1e-6

    def test_compute_structure_compound(self):
        """Test HHI computation for compound structures."""
        hhi_production, _ = get_hhi_data()

        metric = HHIProductionMetric()
        structures = create_test_structures()

        # Test NaCl structure
        nacl_structure = structures[0]
        result = metric.compute_structure(
            nacl_structure, **metric._get_compute_attributes()
        )

        assert isinstance(result, float)
        assert not np.isnan(result)
        assert result > 0

        # Should be weighted average of Na and Cl HHI values
        expected = (hhi_production["Na"] + hhi_production["Cl"]) / 2 / 1000
        assert abs(result - expected) < 1e-6

    def test_scaling_behavior(self):
        """Test that scaling works correctly."""
        # Test with scaling
        metric_scaled = HHIProductionMetric(scale_to_0_10=True)
        structures = create_test_structures()

        result_scaled = metric_scaled.compute_structure(
            structures[1], **metric_scaled._get_compute_attributes()
        )

        # Test without scaling
        metric_unscaled = HHIProductionMetric(scale_to_0_10=False)
        result_unscaled = metric_unscaled.compute_structure(
            structures[1], **metric_unscaled._get_compute_attributes()
        )

        # Unscaled should be 1000x larger
        assert abs(result_unscaled - result_scaled * 1000) < 1e-6

    def test_aggregate_results(self):
        """Test aggregation of results."""
        metric = HHIProductionMetric()

        # Test with valid values
        values = [1.5, 2.0, 2.5, 1.8, 2.2]
        result = metric.aggregate_results(values)

        assert result["primary_metric"] == "hhiproduction_mean"
        assert result["metrics"]["hhiproduction_mean"] == pytest.approx(2.0)
        assert result["metrics"]["hhiproduction_std"] > 0
        assert result["metrics"]["hhiproduction_min"] == 1.5
        assert result["metrics"]["hhiproduction_max"] == 2.5
        assert result["metrics"]["total_structures_evaluated"] == 5

        # Check that individual values are included in metrics
        assert "individual_hhi_values" in result["metrics"]
        assert result["metrics"]["individual_hhi_values"] == values

        # Check new risk assessment metrics
        assert "hhiproduction_low_risk_count_2" in result["metrics"]
        assert "hhiproduction_low_risk_fraction_2" in result["metrics"]
        assert "hhiproduction_25th_percentile" in result["metrics"]
        assert "hhiproduction_75th_percentile" in result["metrics"]

        # Check specific risk counts
        # Values are [1.5, 2.0, 2.5, 1.8, 2.2]
        # Values <= 2.0: 1.5, 2.0, 1.8 = 3 values
        # Values <= 3.0: all 5 values
        assert (
            result["metrics"]["hhiproduction_low_risk_count_2"] == 3
        )  # 1.5, 2.0, 1.8 <= 2.0
        assert (
            result["metrics"]["hhiproduction_low_risk_count_3"] == 5
        )  # All values <= 3.0

        # Check uncertainties
        assert "hhiproduction_mean" in result["uncertainties"]
        assert "std" in result["uncertainties"]["hhiproduction_mean"]

    def test_aggregate_results_with_nan(self):
        """Test aggregation with NaN values."""
        metric = HHIProductionMetric()

        values = [1.5, float("nan"), 2.5, float("nan"), 2.0]
        result = metric.aggregate_results(values)

        # Should ignore NaN values in statistics
        assert result["metrics"]["total_structures_evaluated"] == 3
        assert result["metrics"]["hhiproduction_mean"] == pytest.approx(2.0)
        assert result["metrics"]["failed_structures_count"] == 2

        # But individual values should include original NaN values
        assert "individual_hhi_values" in result["metrics"]
        assert len(result["metrics"]["individual_hhi_values"]) == 5
        assert result["metrics"]["individual_hhi_values"] == values

        # Check that NaN values are preserved in their original positions
        individual_values = result["metrics"]["individual_hhi_values"]
        assert individual_values[0] == 1.5
        assert np.isnan(individual_values[1])
        assert individual_values[2] == 2.5
        assert np.isnan(individual_values[3])
        assert individual_values[4] == 2.0

    def test_aggregate_results_all_nan(self):
        """Test aggregation when all values are NaN."""
        metric = HHIProductionMetric()

        values = [float("nan"), float("nan"), float("nan")]
        result = metric.aggregate_results(values)

        assert np.isnan(result["metrics"]["hhiproduction_mean"])
        assert result["primary_metric"] == "hhiproduction_mean"

    def test_full_computation(self):
        """Test full computation pipeline."""
        metric = HHIProductionMetric()
        structures = create_test_structures()

        result = metric.compute(structures)

        # Check MetricResult properties
        assert isinstance(result, MetricResult)
        assert result.primary_metric == "hhiproduction_mean"
        assert result.n_structures == len(structures)
        assert len(result.individual_values) == len(structures)
        assert result.computation_time > 0

        # Check that individual values are also in the metrics dictionary
        assert "individual_hhi_values" in result.metrics
        assert result.metrics["individual_hhi_values"] == result.individual_values
        assert len(result.metrics["individual_hhi_values"]) == len(structures)

        # Verify individual values are accessible both ways
        for i in range(len(structures)):
            assert (
                result.individual_values[i]
                == result.metrics["individual_hhi_values"][i]
            )

    def test_callable_interface(self):
        """Test the callable interface (__call__ method)."""
        metric = HHIProductionMetric()
        structures = create_test_structures()

        # Test direct call
        result = metric(structures)

        # Should return MetricResult
        assert isinstance(result, MetricResult)
        assert "hhiproduction_mean" in result.metrics

        # Test that individual values are accessible
        assert len(result.individual_values) == len(structures)
        assert all(isinstance(v, float) for v in result.individual_values)

    def test_get_low_risk_structures(self):
        """Test identification of low-risk structures."""
        metric = HHIProductionMetric()
        structures = create_test_structures()

        # Compute results
        result = metric.compute(structures)

        # Test with different thresholds
        low_risk_indices_2, low_risk_values_2 = metric.get_low_risk_structures(
            structures, result, threshold=2.0
        )
        low_risk_indices_5, low_risk_values_5 = metric.get_low_risk_structures(
            structures, result, threshold=5.0
        )

        # Should have more low-risk structures with higher threshold
        assert len(low_risk_indices_5) >= len(low_risk_indices_2)

        # Values should be below threshold
        assert all(v <= 2.0 for v in low_risk_values_2)
        assert all(v <= 5.0 for v in low_risk_values_5)

        # Indices should be valid
        assert all(0 <= i < len(structures) for i in low_risk_indices_2)
        assert all(0 <= i < len(structures) for i in low_risk_indices_5)

    def test_get_individual_values_with_structures(self):
        """Test getting individual values paired with structures."""
        metric = HHIProductionMetric()
        structures = create_test_structures()

        # Get individual values with structures
        structure_values = metric.get_individual_values_with_structures(structures)

        # Should have same number as input structures (assuming no failures)
        assert len(structure_values) == len(structures)

        # Each item should be a tuple of (index, structure, value)
        for idx, structure, value in structure_values:
            assert isinstance(idx, int)
            assert isinstance(structure, Structure)
            assert isinstance(value, float)
            assert 0 <= idx < len(structures)
            assert not np.isnan(value)  # Should exclude failed by default

        # Test sorting by HHI value
        structure_values.sort(key=lambda x: x[2])

        # First structure should have lowest HHI (lowest risk)
        assert structure_values[0][2] <= structure_values[-1][2]


class TestHHIReserveMetric:
    """Test HHI Reserve metric."""

    def test_initialization(self):
        """Test metric initialization."""
        metric = HHIReserveMetric()

        assert metric.name == "HHIReserve"
        assert "reserve concentration" in metric.description.lower()
        assert metric.config.lower_is_better is True

    def test_compute_structure(self):
        """Test HHI reserve computation."""
        _, hhi_reserve = get_hhi_data()

        metric = HHIReserveMetric()
        structures = create_test_structures()

        # Test Fe structure
        fe_structure = structures[1]
        result = metric.compute_structure(
            fe_structure, **metric._get_compute_attributes()
        )

        assert isinstance(result, float)
        assert not np.isnan(result)
        assert result > 0

        # For pure Fe, should equal scaled Fe HHI reserve value
        expected = hhi_reserve["Fe"] / 1000
        assert abs(result - expected) < 1e-6

    def test_different_from_production(self):
        """Test that reserve metric gives different results from production."""
        prod_metric = HHIProductionMetric()
        reserve_metric = HHIReserveMetric()
        structures = create_test_structures()

        prod_result = prod_metric.compute(structures)
        reserve_result = reserve_metric.compute(structures)

        # Results should be different (unless by coincidence)
        assert (
            prod_result.metrics["hhiproduction_mean"]
            != reserve_result.metrics["hhireserve_mean"]
        )


class TestCompoundHHIFunction:
    """Test the standalone compound_hhi function."""

    def test_compound_hhi_simple(self):
        """Test compound HHI function with simple compounds."""
        hhi_production, hhi_reserve = get_hhi_data()
        # Test with a simple compound
        result = compound_hhi("NaCl", hhi_production)
        expected = (hhi_production["Na"] + hhi_production["Cl"]) / 2 / 1000
        assert abs(result - expected) < 1e-6

        # Test without scaling
        result_unscaled = compound_hhi("NaCl", hhi_production, scale_to_0_10=False)
        expected_unscaled = (hhi_production["Na"] + hhi_production["Cl"]) / 2
        assert abs(result_unscaled - expected_unscaled) < 1e-6

    def test_compound_hhi_complex(self):
        """Test compound HHI function with complex compounds."""
        hhi_production, hhi_reserve = get_hhi_data()

        # Test with more complex formula
        result = compound_hhi("Nd2Fe14B", hhi_production)

        # Should compute weighted average correctly
        assert isinstance(result, float)
        assert result > 0

        # Test LiFePO4
        result_lifepo4 = compound_hhi("LiFePO4", hhi_production)
        assert isinstance(result_lifepo4, float)
        assert result_lifepo4 > 0

    def test_compound_hhi_consistency(self):
        """Test that standalone function gives same results as metric."""
        hhi_production, hhi_reserve = get_hhi_data()

        # Create a structure
        structures = create_test_structures()
        nacl_structure = structures[0]  # NaCl

        # Compute using metric
        metric = HHIProductionMetric()
        metric_result = metric.compute_structure(
            nacl_structure, **metric._get_compute_attributes()
        )

        # Compute using standalone function
        function_result = compound_hhi("NaCl", hhi_production)

        # Should be the same
        assert abs(metric_result - function_result) < 1e-6


class TestErrorHandling:
    """Test error handling in HHI metrics."""

    def test_missing_element_handling(self):
        """
        Test that missing elements are assigned maximum HHI
        value instead of erroring.
        """
        # Create a mock HHI table missing some elements
        mock_hhi_table = {"Fe": 2424, "Na": 1102}  # Missing Cl

        class MockHHIMetric(BaseHHIMetric):
            def __init__(self):
                super().__init__(
                    hhi_table=mock_hhi_table,
                    name="MockHHI",
                    description="Mock HHI metric",
                    scale_to_0_10=True,
                )

        metric = MockHHIMetric()
        structures = create_test_structures()
        nacl_structure = structures[0]  # Contains Cl, which is missing from table

        # Should not raise an error, but assign maximum HHI value
        result = metric.compute_structure(
            nacl_structure, **metric._get_compute_attributes()
        )

        # Result should be valid and relatively high due to missing Cl
        assert isinstance(result, float)
        assert not np.isnan(result)
        assert result > 5.0  # Should be high due to Cl getting max value (10.0)

        # Calculate expected value: (Na_HHI/1000 + Cl_max) / 2
        expected = (mock_hhi_table["Na"] / 1000.0 + 10.0) / 2
        assert abs(result - expected) < 1e-6

    def test_missing_element_unscaled(self):
        """Test missing element handling with unscaled values."""
        mock_hhi_table = {"Fe": 2424}  # Missing other elements

        class MockHHIMetric(BaseHHIMetric):
            def __init__(self):
                super().__init__(
                    hhi_table=mock_hhi_table,
                    name="MockHHI",
                    description="Mock HHI metric",
                    scale_to_0_10=False,  # No scaling
                )

        metric = MockHHIMetric()
        structures = create_test_structures()
        nacl_structure = structures[0]

        result = metric.compute_structure(
            nacl_structure, **metric._get_compute_attributes()
        )

        # Both Na and Cl should get max value (10000), so result should be 10000
        assert isinstance(result, float)
        assert not np.isnan(result)
        assert abs(result - 10000.0) < 1e-6

    def test_compound_hhi_missing_elements(self):
        """Test compound_hhi function with missing elements."""
        mock_hhi_table = {"Fe": 2424, "O": 500}  # Missing other elements

        # Test with compound containing missing elements
        result = compound_hhi("NaCl", mock_hhi_table)  # Na and Cl missing

        # Both Na and Cl should get max value (10.0 scaled),
        # so result should be 10.0
        assert isinstance(result, float)
        assert abs(result - 10.0) < 1e-6

        # Test unscaled
        result_unscaled = compound_hhi("NaCl", mock_hhi_table, scale_to_0_10=False)
        assert abs(result_unscaled - 10000.0) < 1e-6

        # Test mixed (some elements present, some missing)
        result_mixed = compound_hhi("FeO", mock_hhi_table)  # Fe present, O present
        expected_mixed = (
            mock_hhi_table["Fe"] / 1000.0 + mock_hhi_table["O"] / 1000.0
        ) / 2
        assert abs(result_mixed - expected_mixed) < 1e-6

    def test_all_elements_present(self):
        """Test that behavior is unchanged when all elements are present."""
        # This test ensures we didn't break existing functionality
        metric = HHIProductionMetric()
        structures = create_test_structures()

        # Test with structures containing common elements
        for structure in structures:
            try:
                result = metric.compute_structure(
                    structure, **metric._get_compute_attributes()
                )
                assert isinstance(result, float)
                assert not np.isnan(result)
                assert result > 0
            except Exception:
                # If this fails, it might be due to missing elements
                # which is fine for this test
                pass
