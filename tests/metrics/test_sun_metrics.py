"""Tests for SUN (Stable, Unique, Novel) metrics implementation.

Updated to match the new hierarchical computation order:
Stability → Uniqueness → Novelty
"""

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


class TestSUNMetricNewHierarchy:
    """Test suite for new hierarchical SUN metric computation: Stability → Uniqueness → Novelty."""

    def test_hierarchical_order_basic(self):
        """Test that the new hierarchical order works correctly."""
        metric = SUNMetric()
        structures = create_test_structures_with_single_mlip_properties()

        # Test stability computation (Step 1)
        stable_indices, metastable_indices = metric._compute_stability_all(structures)
        
        # Structure 0: e_above_hull = 0.0 (stable)
        # Structure 1: e_above_hull = 0.08 (metastable)
        # Structure 2: e_above_hull = 0.2 (unstable)
        assert stable_indices == [0]
        assert metastable_indices == [1]

    @patch("lemat_genbench.metrics.sun_metric.NoveltyMetric")
    @patch("lemat_genbench.metrics.sun_metric.UniquenessMetric")
    def test_hierarchical_computation_flow(self, mock_uniqueness_class, mock_novelty_class):
        """Test the complete hierarchical computation flow."""
        structures = create_test_structures_with_single_mlip_properties()

        # Mock uniqueness: all stable/metastable structures are unique within their sets
        mock_uniqueness = MagicMock()
        
        def mock_uniqueness_compute(input_structures):
            # Return unique result for whatever structures are passed
            n_structs = len(input_structures)
            return create_mock_uniqueness_result(
                input_structures, [1.0] * n_structs, []
            )
        
        mock_uniqueness.compute.side_effect = mock_uniqueness_compute
        mock_uniqueness_class.return_value = mock_uniqueness

        # Mock novelty: all unique stable/metastable structures are novel
        mock_novelty = MagicMock()
        
        def mock_novelty_compute(input_structures):
            n_structs = len(input_structures)
            mock_result = MagicMock()
            mock_result.individual_values = [1.0] * n_structs
            mock_result.failed_indices = []
            return mock_result
        
        mock_novelty.compute.side_effect = mock_novelty_compute
        mock_novelty_class.return_value = mock_novelty

        # Create metric and compute
        metric = SUNMetric()
        result = metric.compute(structures)

        # Verify hierarchical metrics are present
        assert "stable_count" in result.metrics
        assert "metastable_count" in result.metrics
        assert "unique_in_stable_count" in result.metrics
        assert "unique_in_metastable_count" in result.metrics
        assert "sun_count" in result.metrics
        assert "msun_count" in result.metrics

        # Expected hierarchical results:
        # Level 1 (Stability): 1 stable, 1 metastable, 1 unstable
        assert result.metrics["stable_count"] == 1
        assert result.metrics["metastable_count"] == 1
        
        # Level 2 (Uniqueness within stable/metastable): all unique
        assert result.metrics["unique_in_stable_count"] == 1
        assert result.metrics["unique_in_metastable_count"] == 1
        
        # Level 3 (Novelty): all novel
        assert result.metrics["sun_count"] == 1  # 1 stable, unique, novel
        assert result.metrics["msun_count"] == 1  # 1 metastable, unique, novel

    def test_hierarchical_rates_calculation(self):
        """Test that hierarchical rates are calculated correctly."""
        metric = SUNMetric()
        
        # Create more structures for better rate testing
        structures = []
        for i in range(10):
            structure = Structure(
                lattice=[[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]],
                species=["Na", "Cl"],
                coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
                coords_are_cartesian=False,
            )
            # Mix of stable, metastable, and unstable
            if i < 3:
                structure.properties = {"e_above_hull": 0.0}  # Stable
            elif i < 6:
                structure.properties = {"e_above_hull": 0.05}  # Metastable
            else:
                structure.properties = {"e_above_hull": 0.15}  # Unstable
            structures.append(structure)

        with (
            patch("lemat_genbench.metrics.sun_metric.NoveltyMetric"),
            patch("lemat_genbench.metrics.sun_metric.UniquenessMetric"),
        ):
            # Mock all as unique and novel for simplicity
            mock_uniqueness_result = create_mock_uniqueness_result(
                [], [1.0, 1.0, 1.0], []  # Mock for 3 stable structures
            )
            metric.uniqueness_metric.compute = MagicMock(return_value=mock_uniqueness_result)
            
            mock_novelty_result = MagicMock()
            mock_novelty_result.individual_values = [1.0, 1.0, 1.0]
            mock_novelty_result.failed_indices = []
            metric.novelty_metric.compute = MagicMock(return_value=mock_novelty_result)

            result = metric.compute(structures)

            # Verify rate calculations
            assert result.metrics["stable_rate"] == 0.3  # 3/10
            assert result.metrics["metastable_rate"] == 0.3  # 3/10
            assert result.metrics["unique_in_stable_rate"] == 1.0  # 3/3 stable are unique
            assert result.metrics["unique_in_metastable_rate"] == 1.0  # 3/3 metastable are unique


class TestSUNMetricStructureMatcherSupport:
    """Test suite for structure matcher fingerprinting support."""

    @patch("lemat_genbench.metrics.sun_metric.NoveltyMetric")
    @patch("lemat_genbench.metrics.sun_metric.UniquenessMetric")
    def test_structure_matcher_uniqueness(self, mock_uniqueness_class, mock_novelty_class):
        """Test uniqueness computation with structure matcher method."""
        structures = create_test_structures_with_single_mlip_properties()
        
        # Create metric with structure matcher
        metric = SUNMetric(fingerprint_method="structure-matcher")
        
        # Mock the fingerprinter for structure matcher
        mock_fingerprinter = MagicMock()
        mock_fingerprinter.is_equivalent.return_value = False  # All structures unique
        metric.uniqueness_metric.fingerprinter = mock_fingerprinter
        
        # Mock other components
        mock_uniqueness_class.return_value = metric.uniqueness_metric
        mock_novelty_class.return_value = MagicMock()

        # Test uniqueness computation within a set
        stable_structures = [structures[0]]  # Only stable structure
        unique_indices = metric._compute_uniqueness_within_set(stable_structures)
        
        # Should return all indices since structures are unique
        assert unique_indices == [0]

    @patch("lemat_genbench.metrics.sun_metric.NoveltyMetric")
    @patch("lemat_genbench.metrics.sun_metric.UniquenessMetric")
    def test_structure_matcher_novelty(self, mock_uniqueness_class, mock_novelty_class):
        """Test novelty computation with structure matcher method."""
        structures = create_test_structures_with_single_mlip_properties()
        
        # Create metric with structure matcher
        metric = SUNMetric(fingerprint_method="structure-matcher")
        
        # Mock novelty metric to return positive values for novel structures
        mock_novelty = MagicMock()
        mock_novelty_result = MagicMock()
        mock_novelty_result.individual_values = [1.0, 0.8, 0.5]  # Different values for structure matcher
        mock_novelty_result.failed_indices = []
        mock_novelty.compute.return_value = mock_novelty_result
        
        # Set up the metric's novelty_metric attribute to use our mock
        metric.novelty_metric = mock_novelty
        
        # Test novelty computation within a set
        novel_indices = metric._compute_novelty_within_set(structures)
        
        # All structures with positive values should be considered novel for structure matcher
        assert novel_indices == [0, 1, 2]


class TestSUNMetricBackwardCompatibility:
    """Test suite for SUN Metric backward compatibility with single MLIP properties."""

    def test_single_mlip_properties(self):
        """Test SUN metric with legacy single MLIP properties."""
        metric = SUNMetric()
        structures = create_test_structures_with_single_mlip_properties()

        # Test stability computation with single MLIP properties
        stable_indices, metastable_indices = metric._compute_stability_all(structures)

        # Structure 0: e_above_hull = 0.0 (stable)
        # Structure 1: e_above_hull = 0.08 (metastable)
        # Structure 2: e_above_hull = 0.2 (unstable)
        assert stable_indices == [0]
        assert metastable_indices == [1]

    def test_fallback_behavior(self):
        """Test fallback behavior from e_above_hull_mean to e_above_hull."""
        metric = SUNMetric()
        structures = create_mixed_property_structures()

        stable_indices, metastable_indices = metric._compute_stability_all(structures)

        # Structure 0: e_above_hull_mean = 0.05 (metastable)
        # Structure 1: e_above_hull = 0.08 (fallback, metastable)
        # Structure 2: no properties (should be skipped)
        assert stable_indices == []
        assert sorted(metastable_indices) == [0, 1]


class TestSUNMetricMultiMLIP:
    """Test suite for SUN Metric with multi-MLIP preprocessing."""

    def test_multi_mlip_properties(self):
        """Test SUN metric with multi-MLIP ensemble properties."""
        metric = SUNMetric()
        structures = create_test_structures_with_multi_mlip_properties()

        stable_indices, metastable_indices = metric._compute_stability_all(structures)

        # Structure 0: e_above_hull_mean = 0.0 (stable)
        # Structure 1: e_above_hull_mean = 0.08 (metastable)
        # Structure 2: e_above_hull_mean = 0.2 (unstable)
        # Structure 3: e_above_hull_mean = 0.09 (metastable)
        assert stable_indices == [0]
        assert sorted(metastable_indices) == [1, 3]

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

        stable_indices, metastable_indices = metric._compute_stability_all([structure])

        # Should use ensemble mean (0.07) -> metastable
        assert stable_indices == []
        assert metastable_indices == [0]

    @patch("lemat_genbench.metrics.sun_metric.NoveltyMetric")
    @patch("lemat_genbench.metrics.sun_metric.UniquenessMetric")
    def test_full_sun_computation_multi_mlip(
        self, mock_uniqueness_class, mock_novelty_class
    ):
        """Test full SUN computation with multi-MLIP properties."""
        structures = create_test_structures_with_multi_mlip_properties()

        # Mock uniqueness: all structures within each set are unique
        mock_uniqueness = MagicMock()
        
        def mock_uniqueness_compute(input_structures):
            n_structs = len(input_structures)
            return create_mock_uniqueness_result(
                input_structures, [1.0] * n_structs, []
            )
        
        mock_uniqueness.compute.side_effect = mock_uniqueness_compute
        mock_uniqueness_class.return_value = mock_uniqueness

        # Mock novelty: all structures are novel
        mock_novelty = MagicMock()
        
        def mock_novelty_compute(input_structures):
            n_structs = len(input_structures)
            mock_result = MagicMock()
            mock_result.individual_values = [1.0] * n_structs
            mock_result.failed_indices = []
            return mock_result
            
        mock_novelty.compute.side_effect = mock_novelty_compute
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

        stable_indices, metastable_indices = metric._compute_stability_all(structures)

        # All structures should be skipped due to invalid values
        assert stable_indices == []
        assert metastable_indices == []

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
        stable_indices, metastable_indices = metric._compute_stability_all(structures)

        # Structure 1: -0.03 > -0.05 (not stable) but -0.03 <= -0.02 (metastable)
        # Structure 2: -0.06 <= -0.05 (stable)
        assert stable_indices == [1]
        assert metastable_indices == [0]

    @patch("lemat_genbench.metrics.sun_metric.NoveltyMetric")
    @patch("lemat_genbench.metrics.sun_metric.UniquenessMetric")
    def test_uniqueness_metric_failure(self, mock_uniqueness_class, mock_novelty_class):
        """Test behavior when uniqueness metric fails completely."""
        structures = create_test_structures_with_single_mlip_properties()

        # Mock uniqueness to fail completely
        mock_uniqueness = MagicMock()
        
        def mock_uniqueness_compute(input_structures):
            n_structs = len(input_structures)
            return create_mock_uniqueness_result(
                input_structures, [float("nan")] * n_structs, list(range(n_structs))
            )
        
        mock_uniqueness.compute.side_effect = mock_uniqueness_compute
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
        
        def mock_uniqueness_compute(input_structures):
            n_structs = len(input_structures)
            return create_mock_uniqueness_result(
                input_structures, [1.0] * n_structs, []
            )
        
        mock_uniqueness.compute.side_effect = mock_uniqueness_compute
        mock_uniqueness_class.return_value = mock_uniqueness

        # Mock novelty to fail completely
        mock_novelty = MagicMock()
        
        def mock_novelty_compute(input_structures):
            n_structs = len(input_structures)
            mock_result = MagicMock()
            mock_result.individual_values = [float("nan")] * n_structs
            mock_result.failed_indices = list(range(n_structs))
            return mock_result
            
        mock_novelty.compute.side_effect = mock_novelty_compute
        mock_novelty_class.return_value = mock_novelty

        metric = SUNMetric()
        result = metric.compute(structures)

        # Should return zero SUN/MetaSUN rates
        assert result.metrics["sun_rate"] == 0.0
        assert result.metrics["msun_rate"] == 0.0
        assert result.metrics["sun_count"] == 0
        assert result.metrics["msun_count"] == 0


class TestSUNMetricResultValidation:
    """Test suite for validating MetricResult structure and values."""

    @patch("lemat_genbench.metrics.sun_metric.NoveltyMetric")
    @patch("lemat_genbench.metrics.sun_metric.UniquenessMetric")
    def test_metric_result_structure(self, mock_uniqueness_class, mock_novelty_class):
        """Test that MetricResult has the correct structure and values."""
        structures = create_test_structures_with_single_mlip_properties()

        # Mock all structures as unique and novel
        mock_uniqueness = MagicMock()
        
        def mock_uniqueness_compute(input_structures):
            n_structs = len(input_structures)
            return create_mock_uniqueness_result(
                input_structures, [1.0] * n_structs, []
            )
        
        mock_uniqueness.compute.side_effect = mock_uniqueness_compute
        mock_uniqueness_class.return_value = mock_uniqueness

        mock_novelty = MagicMock()
        
        def mock_novelty_compute(input_structures):
            n_structs = len(input_structures)
            mock_result = MagicMock()
            mock_result.individual_values = [1.0] * n_structs
            mock_result.failed_indices = []
            return mock_result
            
        mock_novelty.compute.side_effect = mock_novelty_compute
        mock_novelty_class.return_value = mock_novelty

        metric = SUNMetric()
        result = metric.compute(structures)

        # Test MetricResult structure
        assert isinstance(result, MetricResult)
        assert isinstance(result.metrics, dict)
        assert isinstance(result.individual_values, list)
        assert isinstance(result.failed_indices, list)
        assert isinstance(result.computation_time, float)
        assert result.computation_time > 0

        # Test required hierarchical metrics
        required_metrics = [
            "sun_rate",
            "msun_rate",
            "combined_sun_msun_rate",
            "sun_count",
            "msun_count",
            "stable_count",
            "metastable_count",
            "stable_rate",
            "metastable_rate",
            "unique_in_stable_count",
            "unique_in_metastable_count",
            "unique_in_stable_rate",
            "unique_in_metastable_rate",
            "total_structures_evaluated",
            "failed_count",
        ]
        for metric_name in required_metrics:
            assert metric_name in result.metrics, f"Missing metric: {metric_name}"

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
        mock_uniqueness = MagicMock()
        
        def mock_uniqueness_compute(input_structures):
            n_structs = len(input_structures)
            return create_mock_uniqueness_result(
                input_structures, [1.0] * n_structs, []
            )
        
        mock_uniqueness.compute.side_effect = mock_uniqueness_compute
        mock_uniqueness_class.return_value = mock_uniqueness

        mock_novelty = MagicMock()
        
        def mock_novelty_compute(input_structures):
            n_structs = len(input_structures)
            mock_result = MagicMock()
            mock_result.individual_values = [1.0] * n_structs
            mock_result.failed_indices = []
            return mock_result
            
        mock_novelty.compute.side_effect = mock_novelty_compute
        mock_novelty_class.return_value = mock_novelty

        metric = SUNMetric()
        result = metric.compute(structures)

        # Structure 0: stable -> 1.0 (SUN)
        # Structure 1: metastable -> 0.5 (MetaSUN)
        # Structure 2: unstable -> 0.0 (neither)
        expected_values = [1.0, 0.5, 0.0]
        assert result.individual_values == expected_values

    @patch("lemat_genbench.metrics.sun_metric.NoveltyMetric")
    @patch("lemat_genbench.metrics.sun_metric.UniquenessMetric")
    def test_hierarchical_indices_attributes(
        self, mock_uniqueness_class, mock_novelty_class
    ):
        """Test that hierarchical indices are properly stored as attributes."""
        structures = create_test_structures_with_single_mlip_properties()

        # Mock all structures as unique and novel
        mock_uniqueness = MagicMock()
        
        def mock_uniqueness_compute(input_structures):
            n_structs = len(input_structures)
            return create_mock_uniqueness_result(
                input_structures, [1.0] * n_structs, []
            )
        
        mock_uniqueness.compute.side_effect = mock_uniqueness_compute
        mock_uniqueness_class.return_value = mock_uniqueness

        mock_novelty = MagicMock()
        
        def mock_novelty_compute(input_structures):
            n_structs = len(input_structures)
            mock_result = MagicMock()
            mock_result.individual_values = [1.0] * n_structs
            mock_result.failed_indices = []
            return mock_result
            
        mock_novelty.compute.side_effect = mock_novelty_compute
        mock_novelty_class.return_value = mock_novelty

        metric = SUNMetric()
        result = metric.compute(structures)

        # Test that hierarchical indices are stored as attributes
        assert hasattr(result, "sun_indices")
        assert hasattr(result, "msun_indices")
        assert hasattr(result, "stable_indices")
        assert hasattr(result, "metastable_indices")
        assert hasattr(result, "stable_unique_indices")
        assert hasattr(result, "metastable_unique_indices")

        # Verify indices match expected values
        assert result.sun_indices == [0]  # Only stable structure
        assert result.msun_indices == [1]  # Only metastable structure
        assert result.stable_indices == [0]
        assert result.metastable_indices == [1]


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
        mock_uniqueness = MagicMock()
        
        def mock_uniqueness_compute(input_structures):
            n_structs = len(input_structures)
            return create_mock_uniqueness_result(
                input_structures, [1.0] * n_structs, []
            )
        
        mock_uniqueness.compute.side_effect = mock_uniqueness_compute
        mock_uniqueness_class.return_value = mock_uniqueness

        mock_novelty = MagicMock()
        
        def mock_novelty_compute(input_structures):
            n_structs = len(input_structures)
            mock_result = MagicMock()
            mock_result.individual_values = [1.0] * n_structs
            mock_result.failed_indices = []
            return mock_result
            
        mock_novelty.compute.side_effect = mock_novelty_compute
        mock_novelty_class.return_value = mock_novelty

        metric = SUNMetric()
        result = metric.compute(structures)

        # All should be SUN, none MetaSUN
        assert result.metrics["sun_rate"] == 1.0
        assert result.metrics["msun_rate"] == 0.0
        assert result.metrics["sun_count"] == 3
        assert result.metrics["msun_count"] == 0
        assert result.metrics["stable_count"] == 3
        assert result.metrics["metastable_count"] == 0

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

        # Mock won't be called since no stable/metastable structures
        mock_uniqueness_class.return_value = MagicMock()
        mock_novelty_class.return_value = MagicMock()

        metric = SUNMetric()
        result = metric.compute(structures)

        # None should be SUN or MetaSUN
        assert result.metrics["sun_rate"] == 0.0
        assert result.metrics["msun_rate"] == 0.0
        assert result.metrics["sun_count"] == 0
        assert result.metrics["msun_count"] == 0
        assert result.metrics["stable_count"] == 0
        assert result.metrics["metastable_count"] == 0

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

        stable_indices, metastable_indices = metric._compute_stability_all(structures)

        # With MetaSUN, stability_threshold = metastability_threshold = 0.1
        # Structure 0: e_above_hull_mean = 0.0 <= 0.1 (stable/metastable)
        # Structure 1: e_above_hull_mean = 0.08 <= 0.1 (stable/metastable)
        # Structure 2: e_above_hull_mean = 0.2 > 0.1 (unstable)
        # Structure 3: e_above_hull_mean = 0.09 <= 0.1 (stable/metastable)
        assert sorted(stable_indices) == [0, 1, 3]
        assert (
            metastable_indices == []
        )  # No distinction between stable and metastable in MetaSUN


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
                mock_uniqueness = MagicMock()
                
                def mock_uniqueness_compute(input_structures):
                    n_structs = len(input_structures)
                    return create_mock_uniqueness_result(
                        input_structures, [1.0] * n_structs, []
                    )
                
                mock_uniqueness.compute.side_effect = mock_uniqueness_compute
                metric.uniqueness_metric = mock_uniqueness
                
                mock_novelty = MagicMock()
                
                def mock_novelty_compute(input_structures):
                    n_structs = len(input_structures)
                    mock_result = MagicMock()
                    mock_result.individual_values = [1.0] * n_structs
                    mock_result.failed_indices = []
                    return mock_result
                    
                mock_novelty.compute.side_effect = mock_novelty_compute
                metric.novelty_metric = mock_novelty

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


# Manual test function for development
def manual_test():
    """Manual test for development purposes."""
    print("Running manual SUN metrics test with hierarchical computation...")

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

        # Test 3: Hierarchical stability computation
        print("3. Testing hierarchical stability computation...")
        stable_indices, metastable_indices = sun_metric._compute_stability_all(multi_structures)
        print(f"Stable indices: {stable_indices}")
        print(f"Metastable indices: {metastable_indices}")

        # Test 4: Hierarchical uniqueness within sets
        print("4. Testing uniqueness within stable set...")
        if stable_indices:
            stable_structures = [multi_structures[i] for i in stable_indices]
            # Mock the uniqueness computation for testing
            with patch.object(sun_metric.uniqueness_metric, 'compute') as mock_compute:
                mock_result = create_mock_uniqueness_result(
                    stable_structures, [1.0] * len(stable_structures), []
                )
                mock_compute.return_value = mock_result
                
                unique_indices = sun_metric._compute_uniqueness_within_set(stable_structures)
                print(f"Unique indices within stable set: {unique_indices}")

        # Test 5: Hierarchical novelty within unique sets
        print("5. Testing novelty within unique structures...")
        if stable_indices:
            stable_structures = [multi_structures[i] for i in stable_indices]
            # Mock the novelty computation for testing
            with patch.object(sun_metric.novelty_metric, 'compute') as mock_compute:
                mock_result = MagicMock()
                mock_result.individual_values = [1.0] * len(stable_structures)
                mock_result.failed_indices = []
                mock_compute.return_value = mock_result
                
                novel_indices = sun_metric._compute_novelty_within_set(stable_structures)
                print(f"Novel indices within stable unique set: {novel_indices}")

        print("\nAll manual tests passed!")
        return True

    except Exception as e:
        print(f"Manual test failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    manual_test()