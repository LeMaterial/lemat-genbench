"""Tests for new SUN (Stable, Unique, Novel) metrics using augmented fingerprinting."""

import traceback
from unittest.mock import MagicMock, patch

import numpy as np
from pymatgen.core.structure import Structure

from lemat_genbench.metrics.base import MetricResult
from lemat_genbench.metrics.sun_new_metric import MetaSUNNewMetric, SUNNewMetric


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


def create_test_structures_with_known_duplicates():
    """Create test structures with known duplicate pattern: A, B, C, C, D, D, D."""
    lattice = [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]]

    # Create unique structures
    struct_a = Structure(
        lattice=lattice,
        species=["Na", "Cl"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )

    struct_b = Structure(
        lattice=lattice,
        species=["K", "Br"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )

    struct_c = Structure(
        lattice=lattice,
        species=["Li", "F"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )

    struct_d = Structure(
        lattice=lattice,
        species=["Cs", "I"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )

    # Create pattern: A, B, C, C, D, D, D (4 unique structures)
    structures = [
        struct_a,  # Index 0: A (unique)
        struct_b,  # Index 1: B (unique)
        struct_c,  # Index 2: C (first occurrence)
        struct_c.copy(),  # Index 3: C (duplicate)
        struct_d,  # Index 4: D (first occurrence)
        struct_d.copy(),  # Index 5: D (duplicate)
        struct_d.copy(),  # Index 6: D (duplicate)
    ]

    return structures


def create_structures_with_stability_data():
    """Create structures with e_above_hull properties for SUN testing."""
    structures = create_test_structures_with_known_duplicates()

    # Add stability data
    stability_values = [0.0, 0.08, 0.0, 0.0, 0.2, 0.2, 0.2]
    for i, struct in enumerate(structures):
        struct.properties["e_above_hull"] = stability_values[i]

    return structures


class TestSUNNewMetricBackwardCompatibility:
    """Test suite for SUN New Metric backward compatibility with single MLIP properties."""

    def test_single_mlip_properties(self):
        """Test SUN metric with legacy single MLIP properties."""
        metric = SUNNewMetric()
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

    def test_multi_mlip_properties(self):
        """Test SUN metric with multi-MLIP ensemble properties."""
        metric = SUNNewMetric()
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

    def test_initialization(self):
        """Test SUNNewMetric initialization."""
        metric = SUNNewMetric()

        assert metric.name == "SUNNew"
        assert metric.config.stability_threshold == 0.0
        assert metric.config.metastability_threshold == 0.1
        assert hasattr(metric, "uniqueness_metric")
        assert hasattr(metric, "novelty_metric")

    def test_custom_initialization(self):
        """Test SUNNewMetric with custom parameters."""
        metric = SUNNewMetric(
            stability_threshold=0.05,
            metastability_threshold=0.15,
            name="Custom SUNNew",
            description="Custom description",
            fingerprint_source="property",
            symprec=0.1,
            angle_tolerance=10.0,
        )

        assert metric.name == "Custom SUNNew"
        assert metric.config.stability_threshold == 0.05
        assert metric.config.metastability_threshold == 0.15
        assert metric.config.fingerprint_source == "property"
        assert metric.config.symprec == 0.1
        assert metric.config.angle_tolerance == 10.0


class TestSUNNewMetricAugmentedFingerprinting:
    """Test suite for SUN New Metric with augmented fingerprinting functionality."""

    @patch("lemat_genbench.metrics.sun_new_metric.AugmentedNoveltyMetric")
    @patch("lemat_genbench.metrics.sun_new_metric.UniquenessNewMetric")
    def test_full_sun_computation_augmented(
        self, mock_uniqueness_class, mock_novelty_class
    ):
        """Test full SUN computation with augmented fingerprinting."""
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
        metric = SUNNewMetric()
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

    @patch("lemat_genbench.metrics.sun_new_metric.AugmentedNoveltyMetric")
    @patch("lemat_genbench.metrics.sun_new_metric.UniquenessNewMetric")
    def test_get_unique_structure_indices_with_fingerprints(
        self, mock_uniqueness_class, mock_novelty_class
    ):
        """Test _get_unique_structure_indices with fingerprint data."""
        _ = create_test_structures_with_known_duplicates()
        
        metric = SUNNewMetric()

        # Mock uniqueness result with known fingerprints
        uniqueness_result = MagicMock()
        uniqueness_result.fingerprints = [
            "fp_A",
            "fp_B",
            "fp_C",
            "fp_C",
            "fp_D",
            "fp_D",
            "fp_D",
        ]
        uniqueness_result.individual_values = [1.0, 1.0, 0.5, 0.5, 1/3, 1/3, 1/3]
        failed_indices = []

        selected_indices = metric._get_unique_structure_indices(
            uniqueness_result, failed_indices
        )

        # Should select indices [0, 1, 2, 4] - first occurrence of each unique fingerprint
        expected_indices = [0, 1, 2, 4]
        assert selected_indices == expected_indices

    @patch("lemat_genbench.metrics.sun_new_metric.AugmentedNoveltyMetric")
    @patch("lemat_genbench.metrics.sun_new_metric.UniquenessNewMetric")
    def test_fingerprint_fallback_behavior(
        self, mock_uniqueness_class, mock_novelty_class
    ):
        """Test fallback behavior when fingerprints not available."""
        metric = SUNNewMetric()

        # Mock uniqueness result without fingerprints
        uniqueness_result = MagicMock()
        uniqueness_result.individual_values = [1.0, 1.0, 0.5, 0.5, 0.25, 0.25, 0.25]
        # Remove fingerprints attribute to test fallback
        if hasattr(uniqueness_result, "fingerprints"):
            delattr(uniqueness_result, "fingerprints")

        with patch("lemat_genbench.metrics.sun_new_metric.logger") as mock_logger:
            selected_indices = metric._get_unique_structure_indices(
                uniqueness_result, []
            )

            # Should log warning about fallback
            mock_logger.warning.assert_called_once()
            assert "Fingerprints not available" in str(mock_logger.warning.call_args)

        # Should still select some representatives using individual values
        assert len(selected_indices) > 0

    def test_empty_structures(self):
        """Test behavior with empty structure list."""
        metric = SUNNewMetric()
        result = metric.compute([])

        assert isinstance(result, MetricResult)
        assert result.n_structures == 0
        assert np.isnan(result.metrics["sun_rate"])
        assert np.isnan(result.metrics["msun_rate"])
        assert result.metrics["sun_count"] == 0
        assert result.metrics["msun_count"] == 0


class TestSUNNewMetricErrorHandling:
    """Test suite for error handling and edge cases."""

    @patch("lemat_genbench.metrics.sun_new_metric.AugmentedNoveltyMetric")
    @patch("lemat_genbench.metrics.sun_new_metric.UniquenessNewMetric")
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

        metric = SUNNewMetric()
        result = metric.compute(structures)

        # Should return zero SUN/MetaSUN rates
        assert result.metrics["sun_rate"] == 0.0
        assert result.metrics["msun_rate"] == 0.0
        assert result.metrics["sun_count"] == 0
        assert result.metrics["msun_count"] == 0

    @patch("lemat_genbench.metrics.sun_new_metric.AugmentedNoveltyMetric")
    @patch("lemat_genbench.metrics.sun_new_metric.UniquenessNewMetric")
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

        metric = SUNNewMetric()
        result = metric.compute(structures)

        # Should return zero SUN/MetaSUN rates
        assert result.metrics["sun_rate"] == 0.0
        assert result.metrics["msun_rate"] == 0.0
        assert result.metrics["sun_count"] == 0
        assert result.metrics["msun_count"] == 0

    def test_missing_e_above_hull_properties(self):
        """Test handling of structures missing both e_above_hull properties."""
        metric = SUNNewMetric()

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


class TestMetaSUNNewMetric:
    """Test suite for MetaSUNNewMetric class."""

    def test_initialization(self):
        """Test MetaSUNNewMetric initialization."""
        metric = MetaSUNNewMetric()

        assert metric.name == "MetaSUNNew"
        assert (
            metric.config.stability_threshold == 0.1
        )  # Default metastability threshold
        assert metric.config.metastability_threshold == 0.1
        assert metric.primary_threshold == 0.1

    def test_custom_initialization(self):
        """Test MetaSUNNewMetric with custom parameters."""
        metric = MetaSUNNewMetric(
            metastability_threshold=0.08,
            name="Custom MetaSUNNew",
            fingerprint_source="property",
        )

        assert metric.name == "Custom MetaSUNNew"
        assert metric.config.stability_threshold == 0.08
        assert metric.config.metastability_threshold == 0.08
        assert metric.primary_threshold == 0.08
        assert metric.config.fingerprint_source == "property"

    def test_metasun_with_multi_mlip_properties(self):
        """Test MetaSUN with multi-MLIP ensemble properties."""
        metric = MetaSUNNewMetric(metastability_threshold=0.1)
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


class TestSUNNewMetricFactoryFunctions:
    """Test suite for factory functions."""

    def test_create_property_based_sun_metric(self):
        """Test property-based SUN metric factory function."""
        from lemat_genbench.metrics.sun_new_metric import (
            create_property_based_sun_metric,
        )
        
        metric = create_property_based_sun_metric()
        
        assert isinstance(metric, SUNNewMetric)
        assert metric.config.fingerprint_source == "property"
        assert not metric.config.fallback_to_computation

    def test_create_computation_based_sun_metric(self):
        """Test computation-based SUN metric factory function."""
        from lemat_genbench.metrics.sun_new_metric import (
            create_computation_based_sun_metric,
        )
        
        metric = create_computation_based_sun_metric()
        
        assert isinstance(metric, SUNNewMetric)
        assert metric.config.fingerprint_source == "compute"

    def test_create_robust_sun_metric(self):
        """Test robust SUN metric factory function."""
        from lemat_genbench.metrics.sun_new_metric import create_robust_sun_metric
        
        metric = create_robust_sun_metric()
        
        assert isinstance(metric, SUNNewMetric)
        assert metric.config.fingerprint_source == "auto"
        assert metric.config.symprec == 0.1
        assert metric.config.angle_tolerance == 10.0
        assert metric.config.fallback_to_computation


class TestSUNNewMetricResultValidation:
    """Test suite for validating MetricResult structure and values."""

    @patch("lemat_genbench.metrics.sun_new_metric.AugmentedNoveltyMetric")
    @patch("lemat_genbench.metrics.sun_new_metric.UniquenessNewMetric")
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

        metric = SUNNewMetric()
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

    @patch("lemat_genbench.metrics.sun_new_metric.AugmentedNoveltyMetric")
    @patch("lemat_genbench.metrics.sun_new_metric.UniquenessNewMetric")
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

        metric = SUNNewMetric()
        result = metric.compute(structures)

        # Structure 0: stable -> 1.0 (SUN)
        # Structure 1: metastable -> 0.5 (MetaSUN)
        # Structure 2: unstable -> 0.0 (neither)
        expected_values = [1.0, 0.5, 0.0]
        assert result.individual_values == expected_values


class TestSUNNewMetricEdgeCases:
    """Test suite for statistical and edge case scenarios."""

    @patch("lemat_genbench.metrics.sun_new_metric.AugmentedNoveltyMetric")
    @patch("lemat_genbench.metrics.sun_new_metric.UniquenessNewMetric")
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

        metric = SUNNewMetric()
        result = metric.compute(structures)

        # All should be SUN, none MetaSUN
        assert result.metrics["sun_rate"] == 1.0
        assert result.metrics["msun_rate"] == 0.0
        assert result.metrics["sun_count"] == 3
        assert result.metrics["msun_count"] == 0

    @patch("lemat_genbench.metrics.sun_new_metric.AugmentedNoveltyMetric")
    @patch("lemat_genbench.metrics.sun_new_metric.UniquenessNewMetric")
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

        metric = SUNNewMetric()
        result = metric.compute(structures)

        # None should be SUN or MetaSUN
        assert result.metrics["sun_rate"] == 0.0
        assert result.metrics["msun_rate"] == 0.0
        assert result.metrics["sun_count"] == 0
        assert result.metrics["msun_count"] == 0


class TestSUNNewMetricStubMethods:
    """Test suite for stub methods required by BaseMetric."""

    def test_compute_structure_method(self):
        """Test the compute_structure static method."""
        structure = Structure(
            lattice=[[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]],
            species=["Na", "Cl"],
            coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
            coords_are_cartesian=False,
        )

        result = SUNNewMetric.compute_structure(structure)
        assert result == 0.0

    def test_aggregate_results_method(self):
        """Test the aggregate_results method."""
        metric = SUNNewMetric()
        values = [1.0, 0.5, 0.0]

        result = metric.aggregate_results(values)

        expected = {
            "metrics": {"sun_rate": 0.0},
            "primary_metric": "sun_rate",
            "uncertainties": {},
        }
        assert result == expected


# Manual test function for development
def manual_test():
    """Manual test for development purposes."""
    print("Running manual SUN New metrics test with augmented fingerprinting...")

    try:
        # Test 1: Basic initialization
        print("1. Testing basic initialization...")
        sun_metric = SUNNewMetric()
        metasun_metric = MetaSUNNewMetric()

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

        # Test 4: Stability computation with different property types
        print("4. Testing stability computation...")

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

        # Test 5: Mock helper function
        print("5. Testing mock helper function...")
        mock_result = create_mock_uniqueness_result(
            ["s1", "s2", "s3"], 
            [1.0, 0.5, 1.0], 
            []
        )
        print(f"Mock fingerprints: {mock_result.fingerprints}")
        print(f"Has fingerprints: {hasattr(mock_result, 'fingerprints')}")

        # Test 6: Factory functions
        print("6. Testing factory functions...")
        from lemat_genbench.metrics.sun_new_metric import (
            create_computation_based_sun_metric,
            create_property_based_sun_metric,
            create_robust_sun_metric,
        )
        
        prop_metric = create_property_based_sun_metric()
        comp_metric = create_computation_based_sun_metric()
        robust_metric = create_robust_sun_metric()
        
        print(f"Property-based metric fingerprint source: {prop_metric.config.fingerprint_source}")
        print(f"Computation-based metric fingerprint source: {comp_metric.config.fingerprint_source}")
        print(f"Robust metric symprec: {robust_metric.config.symprec}")

        print("\nAll manual tests passed!")
        return True

    except Exception as e:
        print(f"Manual test failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    manual_test()