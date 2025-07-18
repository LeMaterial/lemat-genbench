"""Tests for SUN (Stable, Unique, Novel) metrics implementation."""

import traceback
from unittest.mock import MagicMock, patch

import numpy as np
from pymatgen.core.structure import Structure

from lematerial_forgebench.metrics.base import MetricResult
from lematerial_forgebench.metrics.sun_metric import MetaSUNMetric, SUNMetric


def create_test_structures_with_properties():
    """Create test structures with known properties for SUN testing."""
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

    # Structure 4: Duplicate of structure 1 (same lattice and species)
    structure4 = Structure(
        lattice=lattice1,  # Same as structure1
        species=["Na", "Cl"],  # Same as structure1
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structure4.properties = {"e_above_hull": 0.0}  # Stable but not unique
    structures.append(structure4)

    # Structure 5: Stable, will be unique, but will be in reference (not novel)
    lattice5 = [[4.2, 0, 0], [0, 4.2, 0], [0, 0, 4.2]]
    structure5 = Structure(
        lattice=lattice5,
        species=["Mg", "O"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structure5.properties = {"e_above_hull": 0.0}  # Stable but will be known
    structures.append(structure5)

    return structures


class TestSUNMetric:
    """Test suite for SUNMetric class."""

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

    @patch("lematerial_forgebench.metrics.sun_metric.NoveltyMetric")
    @patch("lematerial_forgebench.metrics.sun_metric.UniquenessMetric")
    def test_compute_sun_basic(self, mock_uniqueness_class, mock_novelty_class):
        """Test basic SUN computation with mocked sub-metrics."""

        # Create test structures
        structures = create_test_structures_with_properties()

        # Mock uniqueness metric
        mock_uniqueness = MagicMock()
        mock_uniqueness_result = MagicMock()
        # Structures 0, 1, 2, 4 are unique (structure 3 is duplicate of 0)
        mock_uniqueness_result.individual_values = [1.0, 1.0, 1.0, 0.5, 1.0]
        mock_uniqueness_result.failed_indices = []
        mock_uniqueness.compute.return_value = mock_uniqueness_result
        mock_uniqueness_class.return_value = mock_uniqueness

        # Mock novelty metric
        mock_novelty = MagicMock()
        mock_novelty_result = MagicMock()
        # Of the unique structures [0, 1, 2, 4], assume [0, 1, 2] are novel, [4] is known
        mock_novelty_result.individual_values = [
            1.0,
            1.0,
            1.0,
            0.0,
        ]  # 4 unique structures
        mock_novelty_result.failed_indices = []
        mock_novelty.compute.return_value = mock_novelty_result
        mock_novelty_class.return_value = mock_novelty

        # Create metric and compute
        metric = SUNMetric()
        result = metric.compute(structures)

        # Verify calls
        mock_uniqueness.compute.assert_called_once_with(structures)
        mock_novelty.compute.assert_called_once()

        # Check results
        assert isinstance(result, MetricResult)
        assert result.n_structures == 5

        # Expected: Structure 0 (stable, unique, novel) = SUN
        #           Structure 1 (metastable, unique, novel) = MetaSUN
        #           Structure 2 (unstable, unique, novel) = neither
        #           Structure 3 (stable, duplicate) = not unique
        #           Structure 4 (stable, unique, known) = not novel
        assert result.metrics["sun_count"] == 1  # Only structure 0
        assert result.metrics["msun_count"] == 1  # Only structure 1
        assert result.metrics["sun_rate"] == 0.2  # 1/5
        assert result.metrics["msun_rate"] == 0.2  # 1/5

    def test_compute_stability_extraction(self):
        """Test _compute_stability method with real structures."""
        metric = SUNMetric()
        structures = create_test_structures_with_properties()

        # Test with all structures as candidates
        candidate_indices = [0, 1, 2, 3, 4]
        sun_indices, msun_indices = metric._compute_stability(
            structures, candidate_indices
        )

        # Structure 0, 3, 4 have e_above_hull = 0.0 (stable)
        # Structure 1 has e_above_hull = 0.08 (metastable)
        # Structure 2 has e_above_hull = 0.2 (unstable)

        expected_sun = [0, 3, 4]  # e_above_hull <= 0.0
        expected_msun = [1]  # 0.0 < e_above_hull <= 0.1

        assert sorted(sun_indices) == sorted(expected_sun)
        assert sorted(msun_indices) == sorted(expected_msun)

    def test_compute_stability_thresholds(self):
        """Test stability computation with custom thresholds."""
        metric = SUNMetric(stability_threshold=0.05, metastability_threshold=0.15)
        structures = create_test_structures_with_properties()

        candidate_indices = [0, 1, 2]
        sun_indices, msun_indices = metric._compute_stability(
            structures, candidate_indices
        )

        # With stability_threshold=0.05:
        # Structure 0: e_above_hull=0.0 <= 0.05 (stable)
        # Structure 1: e_above_hull=0.08 > 0.05 but <= 0.15 (metastable)
        # Structure 2: e_above_hull=0.2 > 0.15 (unstable)

        assert sun_indices == [0]
        assert msun_indices == [1]

    def test_missing_e_above_hull(self):
        """Test handling of structures missing e_above_hull properties."""
        metric = SUNMetric()

        # Create structure without e_above_hull
        structure = Structure(
            lattice=[[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]],
            species=["Na", "Cl"],
            coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
            coords_are_cartesian=False,
        )
        # No e_above_hull property

        sun_indices, msun_indices = metric._compute_stability([structure], [0])

        # Should skip structures without e_above_hull
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

    def test_individual_values_assignment(self):
        """Test individual values assignment."""
        metric = SUNMetric()

        # Create mock result
        start_time = 0.0
        n_structures = 5
        sun_indices = [0, 2]  # Structures 0 and 2 are SUN
        msun_indices = [1]  # Structure 1 is MetaSUN
        unique_indices = [0, 1, 2, 3]  # Structure 4 is not unique
        failed_indices = [4]  # Structure 4 failed

        result = metric._create_result(
            start_time,
            n_structures,
            sun_indices,
            msun_indices,
            unique_indices,
            failed_indices,
        )

        expected_individual = [1.0, 0.5, 1.0, 0.0, float("nan")]
        # Structure 0: SUN = 1.0
        # Structure 1: MetaSUN = 0.5
        # Structure 2: SUN = 1.0
        # Structure 3: Neither = 0.0
        # Structure 4: Failed = NaN

        assert len(result.individual_values) == 5
        for i, (actual, expected) in enumerate(
            zip(result.individual_values, expected_individual)
        ):
            if np.isnan(expected):
                assert np.isnan(actual), f"Structure {i}: expected NaN, got {actual}"
            else:
                assert actual == expected, (
                    f"Structure {i}: expected {expected}, got {actual}"
                )

    def test_metric_result_properties(self):
        """Test that MetricResult has all expected properties."""
        with (
            patch("lematerial_forgebench.metrics.sun_metric.NoveltyMetric"),
            patch("lematerial_forgebench.metrics.sun_metric.UniquenessMetric"),
        ):
            metric = SUNMetric(max_reference_size=10)
            structures = create_test_structures_with_properties()[:2]

            # Mock the sub-metrics to return simple results
            metric.uniqueness_metric.compute = MagicMock()
            metric.uniqueness_metric.compute.return_value = MagicMock(
                individual_values=[1.0, 1.0], failed_indices=[]
            )
            metric.novelty_metric.compute = MagicMock()
            metric.novelty_metric.compute.return_value = MagicMock(
                individual_values=[1.0, 1.0], failed_indices=[]
            )

            result = metric.compute(structures)

            # Check MetricResult properties
            assert isinstance(result, MetricResult)
            assert hasattr(result, "metrics")
            assert hasattr(result, "primary_metric")
            assert hasattr(result, "uncertainties")
            assert hasattr(result, "config")
            assert hasattr(result, "computation_time")
            assert hasattr(result, "individual_values")
            assert hasattr(result, "n_structures")
            assert hasattr(result, "failed_indices")
            assert hasattr(result, "warnings")

            # Check specific content
            assert result.primary_metric == "sun_rate"
            assert result.computation_time > 0


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

    def test_metasun_vs_sun_behavior(self):
        """Test that MetaSUN behaves differently from SUN."""
        sun_metric = SUNMetric(stability_threshold=0.0)
        metasun_metric = MetaSUNMetric(metastability_threshold=0.1)

        # Create structure with e_above_hull = 0.05
        structure = Structure(
            lattice=[[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]],
            species=["Na", "Cl"],
            coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
            coords_are_cartesian=False,
        )
        structure.properties = {"e_above_hull": 0.05}

        # For SUN: 0.05 > 0.0, so not stable
        sun_indices, _ = sun_metric._compute_stability([structure], [0])
        assert sun_indices == []

        # For MetaSUN: 0.05 <= 0.1, so metastable
        metasun_indices, _ = metasun_metric._compute_stability([structure], [0])
        assert metasun_indices == [0]


class TestSUNIntegration:
    """Integration tests for SUN metrics."""

    def test_stability_exclusive_sets(self):
        """Test that stable and metastable are mutually exclusive sets."""
        metric = SUNMetric(stability_threshold=0.0, metastability_threshold=0.1)

        # Create structures with different e_above_hull values
        structures = []
        e_above_hull_values = [0.0, 0.05, 0.1, 0.15]

        for i, e_val in enumerate(e_above_hull_values):
            structure = Structure(
                lattice=[[5.0 + i, 0, 0], [0, 5.0, 0], [0, 0, 5.0]],
                species=["Na", "Cl"],
                coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
                coords_are_cartesian=False,
            )
            structure.properties = {"e_above_hull": e_val}
            structures.append(structure)

        candidate_indices = [0, 1, 2, 3]
        sun_indices, msun_indices = metric._compute_stability(
            structures, candidate_indices
        )

        # Check mutual exclusivity
        sun_set = set(sun_indices)
        msun_set = set(msun_indices)
        assert sun_set.isdisjoint(msun_set), (
            "SUN and MetaSUN sets should be mutually exclusive"
        )

        # Expected:
        # Structure 0: e_above_hull = 0.0 <= 0.0 → SUN
        # Structure 1: e_above_hull = 0.05 > 0.0 but <= 0.1 → MetaSUN
        # Structure 2: e_above_hull = 0.1 > 0.0 but <= 0.1 → MetaSUN
        # Structure 3: e_above_hull = 0.15 > 0.1 → Neither

        assert sun_indices == [0]
        assert sorted(msun_indices) == [1, 2]

    def test_realistic_sun_calculation(self):
        """Test realistic SUN calculation with known expected results."""
        with (
            patch("lematerial_forgebench.metrics.sun_metric.NoveltyMetric"),
            patch("lematerial_forgebench.metrics.sun_metric.UniquenessMetric"),
        ):
            metric = SUNMetric()
            structures = create_test_structures_with_properties()

            # Mock uniqueness: structures 0, 1, 2, 4 are unique (3 is duplicate of 0)
            mock_uniqueness_result = MagicMock()
            mock_uniqueness_result.individual_values = [1.0, 1.0, 1.0, 0.5, 1.0]
            mock_uniqueness_result.failed_indices = []
            metric.uniqueness_metric.compute = MagicMock(
                return_value=mock_uniqueness_result
            )

            # Mock novelty: of unique structures [0, 1, 2, 4], structures [0, 1, 2] are novel
            mock_novelty_result = MagicMock()
            mock_novelty_result.individual_values = [
                1.0,
                1.0,
                1.0,
                0.0,
            ]  # Corresponds to structures [0,1,2,4]
            mock_novelty_result.failed_indices = []
            metric.novelty_metric.compute = MagicMock(return_value=mock_novelty_result)

            result = metric.compute(structures)

            # Expected results:
            # Structure 0: stable (0.0), unique, novel → SUN ✓
            # Structure 1: metastable (0.08), unique, novel → MetaSUN ✓
            # Structure 2: unstable (0.2), unique, novel → Neither
            # Structure 3: stable (0.0), duplicate → not unique
            # Structure 4: stable (0.0), unique, not novel → not novel

            assert result.metrics["sun_count"] == 1
            assert result.metrics["msun_count"] == 1
            assert result.metrics["sun_rate"] == 0.2  # 1 out of 5
            assert result.metrics["msun_rate"] == 0.2  # 1 out of 5
            assert result.metrics["combined_sun_msun_rate"] == 0.4  # 2 out of 5

            # Check individual values
            expected_individual = [1.0, 0.5, 0.0, 0.0, 0.0]
            for i, (actual, expected) in enumerate(
                zip(result.individual_values, expected_individual)
            ):
                assert actual == expected, (
                    f"Structure {i}: expected {expected}, got {actual}"
                )

    def test_edge_case_all_unstable(self):
        """Test case where all structures are unstable."""
        with (
            patch("lematerial_forgebench.metrics.sun_metric.NoveltyMetric"),
            patch("lematerial_forgebench.metrics.sun_metric.UniquenessMetric"),
        ):
            metric = SUNMetric()

            # Create structures that are all unstable
            structures = []
            for i in range(3):
                structure = Structure(
                    lattice=[[5.0 + i, 0, 0], [0, 5.0, 0], [0, 0, 5.0]],
                    species=["Na", "Cl"],
                    coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
                    coords_are_cartesian=False,
                )
                structure.properties = {"e_above_hull": 0.5}  # All unstable
                structures.append(structure)

            # Mock all as unique and novel
            mock_result = MagicMock()
            mock_result.individual_values = [1.0, 1.0, 1.0]
            mock_result.failed_indices = []
            metric.uniqueness_metric.compute = MagicMock(return_value=mock_result)
            metric.novelty_metric.compute = MagicMock(return_value=mock_result)

            result = metric.compute(structures)

            # All are unstable, so no SUN or MetaSUN
            assert result.metrics["sun_count"] == 0
            assert result.metrics["msun_count"] == 0
            assert result.metrics["sun_rate"] == 0.0
            assert result.metrics["msun_rate"] == 0.0

    def test_edge_case_no_unique_structures(self):
        """Test case where no structures are unique."""
        with (
            patch("lematerial_forgebench.metrics.sun_metric.NoveltyMetric"),
            patch("lematerial_forgebench.metrics.sun_metric.UniquenessMetric"),
        ):
            metric = SUNMetric()
            structures = create_test_structures_with_properties()[:3]

            # Mock all as duplicates (no individual value = 1.0)
            mock_uniqueness_result = MagicMock()
            mock_uniqueness_result.individual_values = [
                0.5,
                0.33,
                0.33,
            ]  # All duplicates
            mock_uniqueness_result.failed_indices = []
            metric.uniqueness_metric.compute = MagicMock(
                return_value=mock_uniqueness_result
            )

            result = metric.compute(structures)

            # No unique structures, so no SUN or MetaSUN possible
            assert result.metrics["sun_count"] == 0
            assert result.metrics["msun_count"] == 0
            assert result.metrics["sun_rate"] == 0.0
            assert result.metrics["msun_rate"] == 0.0

    def test_callable_interface(self):
        """Test the __call__ interface inherited from BaseMetric."""
        with (
            patch("lematerial_forgebench.metrics.sun_metric.NoveltyMetric"),
            patch("lematerial_forgebench.metrics.sun_metric.UniquenessMetric"),
        ):
            metric = SUNMetric()
            structures = create_test_structures_with_properties()[:2]

            # Mock simple results
            mock_result = MagicMock()
            mock_result.individual_values = [1.0, 1.0]
            mock_result.failed_indices = []
            metric.uniqueness_metric.compute = MagicMock(return_value=mock_result)
            metric.novelty_metric.compute = MagicMock(return_value=mock_result)

            # Test callable interface
            result = metric(structures)

            # Should return MetricResult same as compute()
            assert isinstance(result, MetricResult)
            assert "sun_rate" in result.metrics


# Manual test function for development
def manual_test():
    """Manual test for development purposes."""
    print("Running manual SUN metrics test...")

    try:
        # Test 1: Basic initialization
        print("1. Testing basic initialization...")
        sun_metric = SUNMetric()
        metasun_metric = MetaSUNMetric()

        print(f"SUN metric name: {sun_metric.name}")
        print(f"MetaSUN metric name: {metasun_metric.name}")
        print(f"SUN stability threshold: {sun_metric.config.stability_threshold}")
        print(
            f"MetaSUN stability threshold: {metasun_metric.config.stability_threshold}"
        )

        # Test 2: Structure creation and property setting
        print("2. Testing structure creation...")
        structures = create_test_structures_with_properties()
        print(f"Created {len(structures)} test structures")

        for i, s in enumerate(structures):
            e_hull = s.properties.get("e_above_hull", "Missing")
            print(
                f"  Structure {i}: {s.composition.reduced_formula}, e_above_hull={e_hull}"
            )

        # Test 3: Stability computation
        print("3. Testing stability computation...")
        candidate_indices = [0, 1, 2, 3, 4]
        sun_indices, msun_indices = sun_metric._compute_stability(
            structures, candidate_indices
        )

        print(f"SUN structures (stable): {sun_indices}")
        print(f"MetaSUN structures (metastable): {msun_indices}")

        # Test 4: Mutual exclusivity
        print("4. Testing mutual exclusivity...")
        sun_set = set(sun_indices)
        msun_set = set(msun_indices)
        is_exclusive = sun_set.isdisjoint(msun_set)
        print(f"SUN and MetaSUN are mutually exclusive: {is_exclusive}")

        # Test 5: Result creation
        print("5. Testing result creation...")
        result = sun_metric._create_result(
            start_time=0.0,
            n_structures=5,
            sun_indices=sun_indices,
            msun_indices=msun_indices,
            unique_indices=[0, 1, 2, 4],
            failed_indices=[],
        )

        print(f"SUN rate: {result.metrics['sun_rate']:.3f}")
        print(f"MetaSUN rate: {result.metrics['msun_rate']:.3f}")
        print(f"Combined rate: {result.metrics['combined_sun_msun_rate']:.3f}")
        print(f"Individual values: {result.individual_values}")

        print("\nAll manual tests passed!")
        return True

    except Exception as e:
        print(f"Manual test failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    manual_test()
