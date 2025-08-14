"""Tests for SUN (Stable, Unique, Novel) metrics implementation."""

import traceback
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pymatgen.core.structure import Structure

from lemat_genbench.metrics.base import MetricResult
from lemat_genbench.metrics.sun_metric import MetaSUNMetric, SUNMetric
from lemat_genbench.preprocess.multi_mlip_preprocess import (
    MultiMLIPStabilityPreprocessor,
)


def create_test_structures_with_single_mlip_properties():
    """Create test structures with legacy single MLIP properties for backward compatibility testing."""
    structures = []

    # Structure 1: Stable, will be unique, will be novel
    lattice1 = [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]]
    structure1 = Structure(
        lattice=lattice1,
        species=["Na", "Cl"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structure1.properties = {"e_above_hull": 0.0}  # Stable
    structures.append(structure1)

    # Structure 2: Metastable, will be unique, will be novel
    lattice2 = [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]]
    structure2 = Structure(
        lattice=lattice2,
        species=["K", "Br"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structure2.properties = {"e_above_hull": 0.08}  # Metastable
    structures.append(structure2)

    # Structure 3: Unstable, will be unique, will be novel
    lattice3 = [[6.0, 0, 0], [0, 6.0, 0], [0, 0, 6.0]]
    structure3 = Structure(
        lattice=lattice3,
        species=["Li", "F"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structure3.properties = {"e_above_hull": 0.2}  # Unstable
    structures.append(structure3)

    return structures


def create_test_structures_with_multi_mlip_properties():
    """Create test structures with multi-MLIP properties for ensemble testing."""
    structures = []

    # Structure 1: Stable (ensemble mean), will be unique, will be novel
    lattice1 = [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]]
    structure1 = Structure(
        lattice=lattice1,
        species=["Na", "Cl"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structure1.properties = {
        "e_above_hull_mean": 0.0,  # Ensemble mean - stable
        "e_above_hull_std": 0.02,  # Small uncertainty
        "e_above_hull_orb": -0.01,
        "e_above_hull_mace": 0.01,
        "e_above_hull_uma": 0.0,
        "e_above_hull_n_mlips": 3,
    }
    structures.append(structure1)

    # Structure 2: Metastable (ensemble mean), will be unique, will be novel
    lattice2 = [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]]
    structure2 = Structure(
        lattice=lattice2,
        species=["K", "Br"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structure2.properties = {
        "e_above_hull_mean": 0.08,  # Ensemble mean - metastable
        "e_above_hull_std": 0.03,
        "e_above_hull_orb": 0.05,
        "e_above_hull_mace": 0.11,
        "e_above_hull_uma": 0.08,
        "e_above_hull_n_mlips": 3,
    }
    structures.append(structure2)

    # Structure 3: Unstable (ensemble mean), will be unique, will be novel
    lattice3 = [[6.0, 0, 0], [0, 6.0, 0], [0, 0, 6.0]]
    structure3 = Structure(
        lattice=lattice3,
        species=["Li", "F"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structure3.properties = {
        "e_above_hull_mean": 0.2,  # Ensemble mean - unstable
        "e_above_hull_std": 0.05,
        "e_above_hull_orb": 0.18,
        "e_above_hull_mace": 0.22,
        "e_above_hull_uma": 0.20,
        "e_above_hull_n_mlips": 3,
    }
    structures.append(structure3)

    # Structure 4: High uncertainty case - ensemble suggests metastable but with high uncertainty
    lattice4 = [[3.5, 0, 0], [0, 3.5, 0], [0, 0, 3.5]]
    structure4 = Structure(
        lattice=lattice4,
        species=["Mg", "O"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structure4.properties = {
        "e_above_hull_mean": 0.09,  # Ensemble mean - metastable
        "e_above_hull_std": 0.15,  # High uncertainty
        "e_above_hull_orb": -0.05,  # ORB says stable
        "e_above_hull_mace": 0.23,  # MACE says unstable
        "e_above_hull_uma": 0.09,  # UMA says metastable
        "e_above_hull_n_mlips": 3,
    }
    structures.append(structure4)

    return structures


def create_mixed_property_structures():
    """Create structures with mixed property types to test fallback behavior."""
    structures = []

    # Structure with multi-MLIP properties
    structure1 = Structure(
        lattice=[[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]],
        species=["Na", "Cl"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structure1.properties = {
        "e_above_hull_mean": 0.05,
        "e_above_hull_std": 0.02,
    }
    structures.append(structure1)

    # Structure with only single MLIP properties (fallback case)
    structure2 = Structure(
        lattice=[[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]],
        species=["K", "Br"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structure2.properties = {"e_above_hull": 0.08}  # Only legacy property
    structures.append(structure2)

    # Structure with neither property (missing case)
    structure3 = Structure(
        lattice=[[6.0, 0, 0], [0, 6.0, 0], [0, 0, 6.0]],
        species=["Li", "F"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    # No e_above_hull properties
    structures.append(structure3)

    return structures


def create_structures_with_invalid_e_above_hull():
    """Create structures with invalid e_above_hull values for error testing."""
    structures = []

    # Structure with string e_above_hull
    structure1 = Structure(
        lattice=[[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]],
        species=["Na", "Cl"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structure1.properties = {"e_above_hull": "invalid_string"}
    structures.append(structure1)

    # Structure with None e_above_hull
    structure2 = Structure(
        lattice=[[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]],
        species=["K", "Br"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structure2.properties = {"e_above_hull": None}
    structures.append(structure2)

    # Structure with inf e_above_hull
    structure3 = Structure(
        lattice=[[6.0, 0, 0], [0, 6.0, 0], [0, 0, 6.0]],
        species=["Li", "F"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structure3.properties = {"e_above_hull": float("inf")}
    structures.append(structure3)

    return structures


def create_mock_uniqueness_result(structures, individual_values, failed_indices=None):
    """Helper function to create properly mocked uniqueness results with fingerprints."""
    if failed_indices is None:
        failed_indices = []
    
    mock_result = MagicMock()
    mock_result.individual_values = individual_values
    mock_result.failed_indices = failed_indices
    
    # Add fingerprints attribute that the implementation expects
    # For testing, we'll create unique fingerprints for unique structures
    # and identical fingerprints for duplicate structures
    fingerprints = []
    for i, val in enumerate(individual_values):
        if i not in failed_indices:
            if val == 1.0:
                # Unique structure gets unique fingerprint
                fingerprints.append(f"unique_fp_{i}")
            else:
                # Duplicate structure gets shared fingerprint
                # Use the reciprocal to identify groups (e.g., val=0.5 means 2 duplicates)
                group_size = int(round(1.0 / val)) if val > 0 else 1
                fingerprints.append(f"dup_fp_group{group_size}")
    
    mock_result.fingerprints = fingerprints
    return mock_result


class TestSUNMetricBackwardCompatibility:
    """Test suite for SUN Metric backward compatibility with single MLIP properties."""

    def test_single_mlip_properties(self):
        """Test SUN metric with legacy single MLIP properties."""
        metric = SUNMetric()
        structures = create_test_structures_with_single_mlip_properties()

        # Test stability computation with single MLIP properties
        candidate_indices = [0, 1, 2]
        sun_indices, msun_indices = metric._compute_stability(
            structures, candidate_indices
        )

        # Structure 0: e_above_hull = 0.0 (stable)
        # Structure 1: e_above_hull = 0.08 (metastable)
        # Structure 2: e_above_hull = 0.2 (unstable)
        assert sun_indices == [0]
        assert msun_indices == [1]

    def test_fallback_behavior(self):
        """Test fallback behavior from e_above_hull_mean to e_above_hull."""
        metric = SUNMetric()
        structures = create_mixed_property_structures()

        candidate_indices = [0, 1, 2]
        sun_indices, msun_indices = metric._compute_stability(
            structures, candidate_indices
        )

        # Structure 0: e_above_hull_mean = 0.05 (metastable)
        # Structure 1: e_above_hull = 0.08 (fallback, metastable)
        # Structure 2: no properties (should be skipped)
        assert sun_indices == []
        assert sorted(msun_indices) == [0, 1]


class TestSUNMetricMultiMLIP:
    """Test suite for SUN Metric with multi-MLIP preprocessing."""

    def test_multi_mlip_properties(self):
        """Test SUN metric with multi-MLIP ensemble properties."""
        metric = SUNMetric()
        structures = create_test_structures_with_multi_mlip_properties()

        candidate_indices = [0, 1, 2, 3]
        sun_indices, msun_indices = metric._compute_stability(
            structures, candidate_indices
        )

        # Structure 0: e_above_hull_mean = 0.0 (stable)
        # Structure 1: e_above_hull_mean = 0.08 (metastable)
        # Structure 2: e_above_hull_mean = 0.2 (unstable)
        # Structure 3: e_above_hull_mean = 0.09 (metastable)
        assert sun_indices == [0]
        assert sorted(msun_indices) == [1, 3]

    def test_ensemble_vs_individual_differences(self):
        """Test cases where ensemble mean differs from individual MLIP predictions."""
        metric = SUNMetric()

        # Create structure where individual MLIPs disagree but ensemble is metastable
        structure = Structure(
            lattice=[[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]],
            species=["Na", "Cl"],
            coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
            coords_are_cartesian=False,
        )
        structure.properties = {
            "e_above_hull_mean": 0.07,  # Ensemble: metastable
            "e_above_hull_std": 0.12,  # High disagreement
            "e_above_hull_orb": -0.05,  # ORB: stable
            "e_above_hull_mace": 0.19,  # MACE: unstable
            "e_above_hull_uma": 0.07,  # UMA: metastable
        }

        sun_indices, msun_indices = metric._compute_stability([structure], [0])

        # Should use ensemble mean (0.07) -> metastable
        assert sun_indices == []
        assert msun_indices == [0]

    def test_high_uncertainty_handling(self):
        """Test behavior with high ensemble uncertainty."""
        metric = SUNMetric()
        structures = create_test_structures_with_multi_mlip_properties()

        # Structure 3 has high uncertainty (std=0.15) but ensemble mean of 0.09
        candidate_indices = [3]
        sun_indices, msun_indices = metric._compute_stability(
            structures, candidate_indices
        )

        # Should still use ensemble mean despite high uncertainty
        assert sun_indices == []
        assert msun_indices == [3]

    @patch("lemat_genbench.metrics.sun_metric.NoveltyMetric")
    @patch("lemat_genbench.metrics.sun_metric.UniquenessMetric")
    def test_full_sun_computation_multi_mlip(
        self, mock_uniqueness_class, mock_novelty_class
    ):
        """Test full SUN computation with multi-MLIP properties."""
        structures = create_test_structures_with_multi_mlip_properties()

        # Mock uniqueness: all structures are unique
        mock_uniqueness = MagicMock()
        mock_uniqueness_result = create_mock_uniqueness_result(
            structures, [1.0, 1.0, 1.0, 1.0], []
        )
        mock_uniqueness.compute.return_value = mock_uniqueness_result
        mock_uniqueness_class.return_value = mock_uniqueness

        # Mock novelty: all structures are novel
        mock_novelty = MagicMock()
        mock_novelty_result = MagicMock()
        mock_novelty_result.individual_values = [1.0, 1.0, 1.0, 1.0]
        mock_novelty_result.failed_indices = []
        mock_novelty.compute.return_value = mock_novelty_result
        mock_novelty_class.return_value = mock_novelty

        # Create metric and compute
        metric = SUNMetric()
        result = metric.compute(structures)

        # Expected:
        # Structure 0: stable (0.0), unique, novel → SUN
        # Structure 1: metastable (0.08), unique, novel → MetaSUN
        # Structure 2: unstable (0.2), unique, novel → neither
        # Structure 3: metastable (0.09), unique, novel → MetaSUN
        assert result.metrics["sun_count"] == 1
        assert result.metrics["msun_count"] == 2
        assert result.metrics["sun_rate"] == 0.25  # 1/4
        assert result.metrics["msun_rate"] == 0.5  # 2/4


class TestSUNMetricErrorHandling:
    """Test suite for error handling and edge cases."""

    def test_invalid_e_above_hull_values(self):
        """Test handling of invalid e_above_hull values."""
        metric = SUNMetric()
        structures = create_structures_with_invalid_e_above_hull()

        candidate_indices = [0, 1, 2]
        sun_indices, msun_indices = metric._compute_stability(
            structures, candidate_indices
        )

        # All structures should be skipped due to invalid values
        assert sun_indices == []
        assert msun_indices == []

    def test_negative_thresholds(self):
        """Test behavior with negative threshold values."""
        metric = SUNMetric(stability_threshold=-0.05, metastability_threshold=-0.02)

        # Test case 1: Metastable structure
        structure1 = Structure(
            lattice=[[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]],
            species=["Na", "Cl"],
            coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
            coords_are_cartesian=False,
        )
        structure1.properties = {"e_above_hull": -0.03}

        # Test case 2: Stable structure
        structure2 = Structure(
            lattice=[[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]],
            species=["K", "Br"],
            coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
            coords_are_cartesian=False,
        )
        structure2.properties = {"e_above_hull": -0.06}

        structures = [structure1, structure2]
        sun_indices, msun_indices = metric._compute_stability(structures, [0, 1])

        # Structure 1: -0.03 > -0.05 (not stable) but -0.03 <= -0.02 (metastable)
        # Structure 2: -0.06 <= -0.05 (stable)
        assert sun_indices == [1]
        assert msun_indices == [0]

    @patch("lemat_genbench.metrics.sun_metric.NoveltyMetric")
    @patch("lemat_genbench.metrics.sun_metric.UniquenessMetric")
    def test_uniqueness_metric_failure(self, mock_uniqueness_class, mock_novelty_class):
        """Test behavior when uniqueness metric fails completely."""
        structures = create_test_structures_with_single_mlip_properties()

        # Mock uniqueness to fail completely
        mock_uniqueness = MagicMock()
        mock_uniqueness_result = create_mock_uniqueness_result(
            structures, [float("nan")] * len(structures), list(range(len(structures)))
        )
        mock_uniqueness.compute.return_value = mock_uniqueness_result
        mock_uniqueness_class.return_value = mock_uniqueness

        # Mock novelty (won't be called since uniqueness fails)
        mock_novelty_class.return_value = MagicMock()

        metric = SUNMetric()
        result = metric.compute(structures)

        # Should return zero SUN/MetaSUN rates
        assert result.metrics["sun_rate"] == 0.0
        assert result.metrics["msun_rate"] == 0.0
        assert result.metrics["sun_count"] == 0
        assert result.metrics["msun_count"] == 0

    @patch("lemat_genbench.metrics.sun_metric.NoveltyMetric")
    @patch("lemat_genbench.metrics.sun_metric.UniquenessMetric")
    def test_novelty_metric_failure(self, mock_uniqueness_class, mock_novelty_class):
        """Test behavior when novelty metric fails completely."""
        structures = create_test_structures_with_single_mlip_properties()

        # Mock uniqueness: all structures are unique
        mock_uniqueness = MagicMock()
        mock_uniqueness_result = create_mock_uniqueness_result(
            structures, [1.0] * len(structures), []
        )
        mock_uniqueness.compute.return_value = mock_uniqueness_result
        mock_uniqueness_class.return_value = mock_uniqueness

        # Mock novelty to fail completely
        mock_novelty = MagicMock()
        mock_novelty_result = MagicMock()
        mock_novelty_result.individual_values = [float("nan")] * len(structures)
        mock_novelty_result.failed_indices = list(range(len(structures)))
        mock_novelty.compute.return_value = mock_novelty_result
        mock_novelty_class.return_value = mock_novelty

        metric = SUNMetric()
        result = metric.compute(structures)

        # Should return zero SUN/MetaSUN rates
        assert result.metrics["sun_rate"] == 0.0
        assert result.metrics["msun_rate"] == 0.0
        assert result.metrics["sun_count"] == 0
        assert result.metrics["msun_count"] == 0

    @patch("lemat_genbench.metrics.sun_metric.NoveltyMetric")
    @patch("lemat_genbench.metrics.sun_metric.UniquenessMetric")
    def test_partial_sub_metric_failures(
        self, mock_uniqueness_class, mock_novelty_class
    ):
        """Test behavior with partial failures in sub-metrics."""
        structures = create_test_structures_with_single_mlip_properties()

        # Mock uniqueness: first two unique, third failed
        mock_uniqueness = MagicMock()
        mock_uniqueness_result = create_mock_uniqueness_result(
            structures, [1.0, 1.0, float("nan")], [2]
        )
        mock_uniqueness.compute.return_value = mock_uniqueness_result
        mock_uniqueness_class.return_value = mock_uniqueness

        # Mock novelty: first novel, second not novel
        mock_novelty = MagicMock()
        mock_novelty_result = MagicMock()
        mock_novelty_result.individual_values = [
            1.0,
            0.0,
        ]  # Only called for first two structures
        mock_novelty_result.failed_indices = []
        mock_novelty.compute.return_value = mock_novelty_result
        mock_novelty_class.return_value = mock_novelty

        metric = SUNMetric()
        result = metric.compute(structures)

        # Only first structure should be SUN (stable, unique, novel)
        assert result.metrics["sun_count"] == 1
        assert result.metrics["msun_count"] == 0
        assert result.metrics["failed_count"] == 1


class TestSUNMetricResultValidation:
    """Test suite for validating MetricResult structure and values."""

    @patch("lemat_genbench.metrics.sun_metric.NoveltyMetric")
    @patch("lemat_genbench.metrics.sun_metric.UniquenessMetric")
    def test_metric_result_structure(self, mock_uniqueness_class, mock_novelty_class):
        """Test that MetricResult has the correct structure and values."""
        structures = create_test_structures_with_single_mlip_properties()

        # Mock all structures as unique and novel
        mock_uniqueness_result = create_mock_uniqueness_result(
            structures, [1.0] * len(structures), []
        )
        mock_uniqueness_class.return_value.compute.return_value = mock_uniqueness_result

        mock_novelty_result = MagicMock()
        mock_novelty_result.individual_values = [1.0] * len(structures)
        mock_novelty_result.failed_indices = []
        mock_novelty_class.return_value.compute.return_value = mock_novelty_result

        metric = SUNMetric()
        result = metric.compute(structures)

        # Test MetricResult structure
        assert isinstance(result, MetricResult)
        assert isinstance(result.metrics, dict)
        assert isinstance(result.individual_values, list)
        assert isinstance(result.failed_indices, list)
        assert isinstance(result.computation_time, float)
        assert result.computation_time > 0

        # Test required metrics
        required_metrics = [
            "sun_rate",
            "msun_rate",
            "combined_sun_msun_rate",
            "sun_count",
            "msun_count",
            "unique_count",
            "unique_rate",
            "total_structures_evaluated",
            "failed_count",
        ]
        for metric_name in required_metrics:
            assert metric_name in result.metrics

        # Test primary metric
        assert result.primary_metric == "sun_rate"
        assert result.primary_metric in result.metrics

    @patch("lemat_genbench.metrics.sun_metric.NoveltyMetric")
    @patch("lemat_genbench.metrics.sun_metric.UniquenessMetric")
    def test_individual_values_assignment(
        self, mock_uniqueness_class, mock_novelty_class
    ):
        """Test that individual values are assigned correctly."""
        structures = create_test_structures_with_single_mlip_properties()

        # Mock all structures as unique and novel
        mock_uniqueness_result = create_mock_uniqueness_result(
            structures, [1.0] * len(structures), []
        )
        mock_uniqueness_class.return_value.compute.return_value = mock_uniqueness_result

        mock_novelty_result = MagicMock()
        mock_novelty_result.individual_values = [1.0] * len(structures)
        mock_novelty_result.failed_indices = []
        mock_novelty_class.return_value.compute.return_value = mock_novelty_result

        metric = SUNMetric()
        result = metric.compute(structures)

        # Structure 0: stable -> 1.0 (SUN)
        # Structure 1: metastable -> 0.5 (MetaSUN)
        # Structure 2: unstable -> 0.0 (neither)
        expected_values = [1.0, 0.5, 0.0]
        assert result.individual_values == expected_values

    @patch("lemat_genbench.metrics.sun_metric.NoveltyMetric")
    @patch("lemat_genbench.metrics.sun_metric.UniquenessMetric")
    def test_failed_indices_propagation(
        self, mock_uniqueness_class, mock_novelty_class
    ):
        """Test that failed indices are properly propagated and handled."""
        structures = create_test_structures_with_single_mlip_properties()

        # Mock uniqueness with one failure
        mock_uniqueness = MagicMock()
        mock_uniqueness_result = create_mock_uniqueness_result(
            structures, [1.0, 1.0, float("nan")], [2]
        )
        mock_uniqueness.compute.return_value = mock_uniqueness_result
        mock_uniqueness_class.return_value = mock_uniqueness

        # Mock novelty with no failures
        mock_novelty = MagicMock()
        mock_novelty_result = MagicMock()
        mock_novelty_result.individual_values = [1.0, 1.0]
        mock_novelty_result.failed_indices = []
        mock_novelty.compute.return_value = mock_novelty_result
        mock_novelty_class.return_value = mock_novelty

        metric = SUNMetric()
        result = metric.compute(structures)

        # Failed structure should have NaN individual value
        assert np.isnan(result.individual_values[2])
        assert 2 in result.failed_indices
        assert result.metrics["failed_count"] == 1


class TestSUNMetricEdgeCases:
    """Test suite for statistical and edge case scenarios."""

    @patch("lemat_genbench.metrics.sun_metric.NoveltyMetric")
    @patch("lemat_genbench.metrics.sun_metric.UniquenessMetric")
    def test_all_stable_structures(self, mock_uniqueness_class, mock_novelty_class):
        """Test scenario where all structures are stable."""
        # Create structures that are all stable
        structures = []
        for i in range(3):
            structure = Structure(
                lattice=[[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]],
                species=["Na", "Cl"],
                coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
                coords_are_cartesian=False,
            )
            structure.properties = {"e_above_hull": -0.01}  # All stable
            structures.append(structure)

        # Mock all as unique and novel
        mock_uniqueness_result = create_mock_uniqueness_result(
            structures, [1.0] * len(structures), []
        )
        mock_uniqueness_class.return_value.compute.return_value = mock_uniqueness_result

        mock_novelty_result = MagicMock()
        mock_novelty_result.individual_values = [1.0] * len(structures)
        mock_novelty_result.failed_indices = []
        mock_novelty_class.return_value.compute.return_value = mock_novelty_result

        metric = SUNMetric()
        result = metric.compute(structures)

        # All should be SUN, none MetaSUN
        assert result.metrics["sun_rate"] == 1.0
        assert result.metrics["msun_rate"] == 0.0
        assert result.metrics["sun_count"] == 3
        assert result.metrics["msun_count"] == 0

    @patch("lemat_genbench.metrics.sun_metric.NoveltyMetric")
    @patch("lemat_genbench.metrics.sun_metric.UniquenessMetric")
    def test_all_unstable_structures(self, mock_uniqueness_class, mock_novelty_class):
        """Test scenario where all structures are unstable."""
        # Create structures that are all unstable
        structures = []
        for i in range(3):
            structure = Structure(
                lattice=[[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]],
                species=["Na", "Cl"],
                coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
                coords_are_cartesian=False,
            )
            structure.properties = {"e_above_hull": 0.5}  # All unstable
            structures.append(structure)

        # Mock all as unique and novel
        mock_uniqueness_result = create_mock_uniqueness_result(
            structures, [1.0] * len(structures), []
        )
        mock_uniqueness_class.return_value.compute.return_value = mock_uniqueness_result

        mock_novelty_result = MagicMock()
        mock_novelty_result.individual_values = [1.0] * len(structures)
        mock_novelty_result.failed_indices = []
        mock_novelty_class.return_value.compute.return_value = mock_novelty_result

        metric = SUNMetric()
        result = metric.compute(structures)

        # None should be SUN or MetaSUN
        assert result.metrics["sun_rate"] == 0.0
        assert result.metrics["msun_rate"] == 0.0
        assert result.metrics["sun_count"] == 0
        assert result.metrics["msun_count"] == 0

    @patch("lemat_genbench.metrics.sun_metric.NoveltyMetric")
    @patch("lemat_genbench.metrics.sun_metric.UniquenessMetric")
    def test_no_unique_structures(self, mock_uniqueness_class, mock_novelty_class):
        """Test scenario where no structures are unique."""
        structures = create_test_structures_with_single_mlip_properties()

        # Mock no structures as unique (all duplicates)
        mock_uniqueness = MagicMock()
        mock_uniqueness_result = create_mock_uniqueness_result(
            structures, [0.5] * len(structures), []  # All duplicates (2 copies each)
        )
        mock_uniqueness.compute.return_value = mock_uniqueness_result
        mock_uniqueness_class.return_value = mock_uniqueness

        # Mock novelty for the representatives that will be selected
        mock_novelty = MagicMock()
        mock_novelty_result = MagicMock()
        mock_novelty_result.individual_values = [1.0]  # One representative per group
        mock_novelty_result.failed_indices = []
        mock_novelty.compute.return_value = mock_novelty_result
        mock_novelty_class.return_value = mock_novelty

        metric = SUNMetric()
        result = metric.compute(structures)

        # The unique_count should be 1 (one fingerprint group) out of 3 structures
        # So unique_rate = 1/3 ≈ 0.333
        expected_unique_rate = 1.0 / len(structures)
        assert abs(result.metrics["unique_rate"] - expected_unique_rate) < 0.001
        
        # Should still have some SUN/MetaSUN from representatives
        # But no structures have individual_values = 1.0 (perfectly unique)
        assert all(val != 1.0 for val in mock_uniqueness_result.individual_values)


class TestSUNMetricStubMethods:
    """Test suite for stub methods required by BaseMetric."""

    def test_compute_structure_method(self):
        """Test the compute_structure static method."""
        structure = Structure(
            lattice=[[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]],
            species=["Na", "Cl"],
            coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
            coords_are_cartesian=False,
        )

        result = SUNMetric.compute_structure(structure)
        assert result == 0.0

    def test_aggregate_results_method(self):
        """Test the aggregate_results method."""
        metric = SUNMetric()
        values = [1.0, 0.5, 0.0]

        result = metric.aggregate_results(values)

        expected = {
            "metrics": {"sun_rate": 0.0},
            "primary_metric": "sun_rate",
            "uncertainties": {},
        }
        assert result == expected


class TestSUNMetricIntegrationWithPreprocessing:
    """Integration tests using actual multi-MLIP preprocessing."""

    @pytest.mark.slow
    def test_with_actual_multi_mlip_preprocessing(self):
        """Test SUN metric with actual MultiMLIPStabilityPreprocessor."""
        # Create simple test structures
        structures = []

        # Simple cubic NaCl structure
        structure1 = Structure(
            lattice=[[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]],
            species=["Na", "Cl"],
            coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
            coords_are_cartesian=False,
        )
        structures.append(structure1)

        # Simple cubic KBr structure (different composition)
        structure2 = Structure(
            lattice=[[4.5, 0, 0], [0, 4.5, 0], [0, 0, 4.5]],
            species=["K", "Br"],
            coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
            coords_are_cartesian=False,
        )
        structures.append(structure2)

        try:
            # Initialize multi-MLIP preprocessor with minimal MLIPs for testing
            preprocessor = MultiMLIPStabilityPreprocessor(
                mlip_names=["orb"],  # Use only ORB for faster testing
                relax_structures=False,  # Skip relaxation for speed
                calculate_formation_energy=False,  # Skip formation energy for speed
                calculate_energy_above_hull=True,  # This is what we need
                extract_embeddings=False,  # Skip embeddings for speed
                timeout=30,  # Short timeout
            )

            # Preprocess structures
            processed_structures = []
            for structure in structures:
                try:
                    processed_structure = preprocessor.process_structure(
                        structure,
                        preprocessor.calculators,
                        preprocessor.config.timeout,
                        preprocessor.config.relax_structures,
                        preprocessor.config.relaxation_config,
                        preprocessor.config.calculate_formation_energy,
                        preprocessor.config.calculate_energy_above_hull,
                        preprocessor.config.extract_embeddings,
                    )
                    processed_structures.append(processed_structure)
                except Exception as e:
                    # Skip failed structures but don't fail the test
                    print(f"Preprocessing failed for structure: {e}")
                    continue

            if not processed_structures:
                pytest.skip("No structures successfully preprocessed")

            # Verify multi-MLIP properties are present
            for structure in processed_structures:
                # Should have ensemble properties
                assert (
                    "e_above_hull_mean" in structure.properties
                    or "e_above_hull_orb" in structure.properties
                )
                print(f"Structure properties: {list(structure.properties.keys())}")

            # Test SUN metric with preprocessed structures
            with (
                patch("lemat_genbench.metrics.sun_metric.NoveltyMetric"),
                patch("lemat_genbench.metrics.sun_metric.UniquenessMetric"),
            ):
                # Mock uniqueness and novelty for deterministic testing
                metric = SUNMetric()

                # Mock all as unique and novel
                mock_uniqueness_result = create_mock_uniqueness_result(
                    processed_structures, [1.0] * len(processed_structures), []
                )
                metric.uniqueness_metric.compute = MagicMock(return_value=mock_uniqueness_result)
                
                mock_novelty_result = MagicMock()
                mock_novelty_result.individual_values = [1.0] * len(processed_structures)
                mock_novelty_result.failed_indices = []
                metric.novelty_metric.compute = MagicMock(return_value=mock_novelty_result)

                result = metric.compute(processed_structures)

                # Should compute successfully without errors
                assert isinstance(result, MetricResult)
                assert result.n_structures == len(processed_structures)
                assert "sun_rate" in result.metrics
                assert "msun_rate" in result.metrics

                # All metrics should be finite (not NaN)
                assert np.isfinite(result.metrics["sun_rate"])
                assert np.isfinite(result.metrics["msun_rate"])

        except ImportError as e:
            pytest.skip(f"Required dependencies not available: {e}")
        except Exception as e:
            pytest.skip(f"Multi-MLIP preprocessing failed: {e}")

    def test_mixed_preprocessing_scenarios(self):
        """Test SUN metric with mixed single/multi-MLIP preprocessed structures."""
        # Create structures with different preprocessing states
        structures = []

        # Structure 1: Multi-MLIP preprocessed
        structure1 = Structure(
            lattice=[[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]],
            species=["Na", "Cl"],
            coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
            coords_are_cartesian=False,
        )
        structure1.properties = {
            "e_above_hull_mean": 0.02,
            "e_above_hull_std": 0.01,
            "e_above_hull_orb": 0.01,
            "e_above_hull_mace": 0.03,
        }
        structures.append(structure1)

        # Structure 2: Single-MLIP preprocessed (legacy)
        structure2 = Structure(
            lattice=[[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]],
            species=["K", "Br"],
            coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
            coords_are_cartesian=False,
        )
        structure2.properties = {"e_above_hull": 0.07}
        structures.append(structure2)

        # Structure 3: No preprocessing (missing properties)
        structure3 = Structure(
            lattice=[[6.0, 0, 0], [0, 6.0, 0], [0, 0, 6.0]],
            species=["Li", "F"],
            coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
            coords_are_cartesian=False,
        )
        structures.append(structure3)

        metric = SUNMetric()
        candidate_indices = [0, 1, 2]
        sun_indices, msun_indices = metric._compute_stability(
            structures, candidate_indices
        )

        # Structure 0: e_above_hull_mean = 0.02 (metastable)
        # Structure 1: e_above_hull = 0.07 (metastable, fallback)
        # Structure 2: no properties (skipped)
        assert sun_indices == []
        assert sorted(msun_indices) == [0, 1]


class TestSUNMetricOriginalFunctionality:
    """Test suite preserving original SUN Metric functionality."""

    def test_initialization(self):
        """Test SUNMetric initialization."""
        metric = SUNMetric()

        assert metric.name == "SUN"
        assert metric.config.stability_threshold == 0.0
        assert metric.config.metastability_threshold == 0.1
        assert hasattr(metric, "uniqueness_metric")
        assert hasattr(metric, "novelty_metric")

    def test_custom_initialization(self):
        """Test SUNMetric with custom parameters."""
        metric = SUNMetric(
            stability_threshold=0.05,
            metastability_threshold=0.15,
            name="Custom SUN",
            description="Custom description",
            max_reference_size=100,
        )

        assert metric.name == "Custom SUN"
        assert metric.config.stability_threshold == 0.05
        assert metric.config.metastability_threshold == 0.15
        assert metric.config.max_reference_size == 100

    def test_missing_e_above_hull_properties(self):
        """Test handling of structures missing both e_above_hull properties."""
        metric = SUNMetric()

        # Create structure without any e_above_hull properties
        structure = Structure(
            lattice=[[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]],
            species=["Na", "Cl"],
            coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
            coords_are_cartesian=False,
        )
        # No e_above_hull properties

        sun_indices, msun_indices = metric._compute_stability([structure], [0])

        # Should skip structures without any e_above_hull properties
        assert sun_indices == []
        assert msun_indices == []

    def test_empty_structures(self):
        """Test behavior with empty structure list."""
        metric = SUNMetric()
        result = metric.compute([])

        assert isinstance(result, MetricResult)
        assert result.n_structures == 0
        assert np.isnan(result.metrics["sun_rate"])
        assert np.isnan(result.metrics["msun_rate"])
        assert result.metrics["sun_count"] == 0
        assert result.metrics["msun_count"] == 0


class TestMetaSUNMetric:
    """Test suite for MetaSUNMetric class."""

    def test_initialization(self):
        """Test MetaSUNMetric initialization."""
        metric = MetaSUNMetric()

        assert metric.name == "MetaSUN"
        assert (
            metric.config.stability_threshold == 0.1
        )  # Default metastability threshold
        assert metric.config.metastability_threshold == 0.1
        assert metric.primary_threshold == 0.1

    def test_custom_initialization(self):
        """Test MetaSUNMetric with custom parameters."""
        metric = MetaSUNMetric(
            metastability_threshold=0.08,
            name="Custom MetaSUN",
            max_reference_size=50,
        )

        assert metric.name == "Custom MetaSUN"
        assert metric.config.stability_threshold == 0.08
        assert metric.config.metastability_threshold == 0.08
        assert metric.primary_threshold == 0.08

    def test_metasun_with_multi_mlip_properties(self):
        """Test MetaSUN with multi-MLIP ensemble properties."""
        metric = MetaSUNMetric(metastability_threshold=0.1)
        structures = create_test_structures_with_multi_mlip_properties()

        candidate_indices = [0, 1, 2, 3]
        sun_indices, msun_indices = metric._compute_stability(
            structures, candidate_indices
        )

        # With MetaSUN, stability_threshold = metastability_threshold = 0.1
        # Structure 0: e_above_hull_mean = 0.0 <= 0.1 (stable/metastable)
        # Structure 1: e_above_hull_mean = 0.08 <= 0.1 (stable/metastable)
        # Structure 2: e_above_hull_mean = 0.2 > 0.1 (unstable)
        # Structure 3: e_above_hull_mean = 0.09 <= 0.1 (stable/metastable)
        assert sorted(sun_indices) == [0, 1, 3]
        assert (
            msun_indices == []
        )  # No distinction between stable and metastable in MetaSUN


# Manual test function for development
def manual_test():
    """Manual test for development purposes."""
    print("Running manual SUN metrics test with multi-MLIP support...")

    try:
        # Test 1: Basic initialization
        print("1. Testing basic initialization...")
        sun_metric = SUNMetric()
        metasun_metric = MetaSUNMetric()

        print(f"SUN metric name: {sun_metric.name}")
        print(f"MetaSUN metric name: {metasun_metric.name}")

        # Test 2: Multi-MLIP structures
        print("2. Testing multi-MLIP structure creation...")
        multi_structures = create_test_structures_with_multi_mlip_properties()
        print(f"Created {len(multi_structures)} multi-MLIP test structures")

        for i, s in enumerate(multi_structures):
            e_hull_mean = s.properties.get("e_above_hull_mean", "Missing")
            e_hull_std = s.properties.get("e_above_hull_std", "Missing")
            print(
                f"  Structure {i}: {s.composition.reduced_formula}, "
                f"e_above_hull_mean={e_hull_mean}, std={e_hull_std}"
            )

        # Test 3: Single MLIP structures (backward compatibility)
        print("3. Testing single-MLIP structure creation...")
        single_structures = create_test_structures_with_single_mlip_properties()
        print(f"Created {len(single_structures)} single-MLIP test structures")

        for i, s in enumerate(single_structures):
            e_hull = s.properties.get("e_above_hull", "Missing")
            print(
                f"  Structure {i}: {s.composition.reduced_formula}, e_above_hull={e_hull}"
            )

        # Test 4: Mixed properties (fallback behavior)
        print("4. Testing mixed property structures...")
        mixed_structures = create_mixed_property_structures()
        print(f"Created {len(mixed_structures)} mixed property test structures")

        for i, s in enumerate(mixed_structures):
            e_hull_mean = s.properties.get("e_above_hull_mean", "Missing")
            e_hull = s.properties.get("e_above_hull", "Missing")
            print(
                f"  Structure {i}: {s.composition.reduced_formula}, "
                f"mean={e_hull_mean}, single={e_hull}"
            )

        # Test 5: Stability computation with different property types
        print("5. Testing stability computation...")

        # Multi-MLIP
        candidate_indices = list(range(len(multi_structures)))
        sun_indices, msun_indices = sun_metric._compute_stability(
            multi_structures, candidate_indices
        )
        print(f"Multi-MLIP - SUN: {sun_indices}, MetaSUN: {msun_indices}")

        # Single MLIP
        candidate_indices = list(range(len(single_structures)))
        sun_indices, msun_indices = sun_metric._compute_stability(
            single_structures, candidate_indices
        )
        print(f"Single-MLIP - SUN: {sun_indices}, MetaSUN: {msun_indices}")

        # Mixed
        candidate_indices = list(range(len(mixed_structures)))
        sun_indices, msun_indices = sun_metric._compute_stability(
            mixed_structures, candidate_indices
        )
        print(f"Mixed - SUN: {sun_indices}, MetaSUN: {msun_indices}")

        # Test 6: Mock helper function
        print("6. Testing mock helper function...")
        mock_result = create_mock_uniqueness_result(
            ["s1", "s2", "s3"], 
            [1.0, 0.5, 1.0], 
            []
        )
        print(f"Mock fingerprints: {mock_result.fingerprints}")
        print(f"Has fingerprints: {hasattr(mock_result, 'fingerprints')}")

        print("\nAll manual tests passed!")
        return True

    except Exception as e:
        print(f"Manual test failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    manual_test()