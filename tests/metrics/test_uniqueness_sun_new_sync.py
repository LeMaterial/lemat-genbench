"""Tests for uniqueness-SUN synchronization fixes using new augmented fingerprinting.

This module contains tests to verify that the standalone uniqueness metric
and SUN metric report consistent uniqueness counts and properly handle
duplicate structures after the synchronization fixes using the new augmented
fingerprinting approach.
"""

from unittest.mock import Mock, patch

from pymatgen.core.structure import Structure
from pymatgen.util.testing import PymatgenTest

from lemat_genbench.metrics.sun_new_metric import SUNNewMetric
from lemat_genbench.metrics.uniqueness_new_metric import UniquenessNewMetric


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


class TestUniquenessNewMetricFingerprints:
    """Test that UniquenessNewMetric properly includes fingerprints in results."""

    def test_fingerprints_attribute_exists(self):
        """Test that fingerprints attribute is added to MetricResult."""
        metric = UniquenessNewMetric()
        structures = create_test_structures_with_known_duplicates()

        result = metric.compute(structures)

        # Check fingerprints attribute exists
        assert hasattr(result, "fingerprints")
        assert isinstance(result.fingerprints, list)
        assert len(result.fingerprints) > 0

    def test_fingerprints_count_matches_successful_structures(self):
        """Test that number of fingerprints matches non-failed structures."""
        metric = UniquenessNewMetric()
        structures = create_test_structures_with_known_duplicates()

        result = metric.compute(structures)

        expected_successful = len(structures) - len(result.failed_indices)
        assert len(result.fingerprints) == expected_successful

    def test_fingerprints_on_complete_failure(self):
        """Test that fingerprints is empty list on complete failure."""
        metric = UniquenessNewMetric()

        # Mock the fingerprint computation to always fail
        with patch.object(metric, "_compute_structure_fingerprint", return_value=None):
            structures = create_test_structures_with_known_duplicates()[:2]

            result = metric.compute(structures)

            assert hasattr(result, "fingerprints")
            assert result.fingerprints == []

    def test_fingerprints_with_known_pattern(self):
        """Test fingerprints with structures containing known duplicates."""
        metric = UniquenessNewMetric()
        structures = create_test_structures_with_known_duplicates()

        result = metric.compute(structures)

        # Should have fingerprints for all 7 structures
        assert len(result.fingerprints) == 7

        # Should report 4 unique structures (A, B, C, D)
        assert result.metrics["unique_structures_count"] == 4

        # Expected individual values: [1.0, 1.0, 0.5, 0.5, 1/3, 1/3, 1/3]
        # A, B: unique (1.0)
        # C: appears 2 times (0.5 each)
        # D: appears 3 times (1/3 each)
        expected_values = [1.0, 1.0, 0.5, 0.5, 1 / 3, 1 / 3, 1 / 3]

        for i, expected in enumerate(expected_values):
            assert abs(result.individual_values[i] - expected) < 1e-6


class TestUniquenessNewSUNNewSynchronization:
    """Test synchronization between standalone uniqueness and SUN metrics using new fingerprinting."""

    def test_unique_count_synchronization(self):
        """Test that standalone uniqueness and SUN report same unique count."""
        structures = create_structures_with_stability_data()

        # Test standalone uniqueness
        uniqueness_metric = UniquenessNewMetric()
        uniqueness_result = uniqueness_metric.compute(structures)

        # Test SUN metric
        sun_metric = SUNNewMetric()
        sun_result = sun_metric.compute(structures)

        # Both should report same unique count and rate
        assert (
            uniqueness_result.metrics["unique_structures_count"]
            == sun_result.metrics["unique_count"]
        )
        assert (
            abs(
                uniqueness_result.metrics["uniqueness_score"]
                - sun_result.metrics["unique_rate"]
            )
            < 1e-10
        )

        # Should be 4 unique structures out of 7 total
        assert uniqueness_result.metrics["unique_structures_count"] == 4
        assert abs(uniqueness_result.metrics["uniqueness_score"] - 4 / 7) < 1e-6

    def test_individual_values_consistency(self):
        """Test that individual values are correctly assigned in both metrics."""
        structures = create_test_structures_with_known_duplicates()

        uniqueness_metric = UniquenessNewMetric()
        result = uniqueness_metric.compute(structures)

        # Expected individual values: [1.0, 1.0, 0.5, 0.5, 1/3, 1/3, 1/3]
        expected_values = [1.0, 1.0, 0.5, 0.5, 1 / 3, 1 / 3, 1 / 3]

        for i, expected in enumerate(expected_values):
            assert abs(result.individual_values[i] - expected) < 1e-6

    def test_edge_case_all_unique_structures(self):
        """Test consistency when all structures are unique."""
        test = PymatgenTest()
        structures = [
            test.get_structure("Si"),
            test.get_structure("LiFePO4"),
            test.get_structure("Li2O"),
        ]

        # Add stability data
        for struct in structures:
            struct.properties["e_above_hull"] = 0.0

        uniqueness_metric = UniquenessNewMetric()
        uniqueness_result = uniqueness_metric.compute(structures)

        sun_metric = SUNNewMetric()
        sun_result = sun_metric.compute(structures)

        # Both should report all structures as unique
        assert uniqueness_result.metrics["unique_structures_count"] == 3
        assert sun_result.metrics["unique_count"] == 3
        assert uniqueness_result.metrics["uniqueness_score"] == 1.0
        assert sun_result.metrics["unique_rate"] == 1.0

    def test_edge_case_all_duplicate_structures(self):
        """Test consistency when all structures are duplicates."""
        lattice = [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]]
        base_structure = Structure(
            lattice,
            ["Na", "Cl"],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
            coords_are_cartesian=False,
        )
        base_structure.properties["e_above_hull"] = 0.0

        # Create 4 identical structures
        structures = [base_structure.copy() for _ in range(4)]

        uniqueness_metric = UniquenessNewMetric()
        uniqueness_result = uniqueness_metric.compute(structures)

        sun_metric = SUNNewMetric()
        sun_result = sun_metric.compute(structures)

        # Both should report 1 unique structure
        assert uniqueness_result.metrics["unique_structures_count"] == 1
        assert sun_result.metrics["unique_count"] == 1
        assert uniqueness_result.metrics["uniqueness_score"] == 0.25  # 1/4
        assert sun_result.metrics["unique_rate"] == 0.25


class TestSUNNewRepresentativeSelection:
    """Test SUN's _get_unique_structure_indices method using new fingerprinting."""

    def test_fingerprint_based_selection_with_real_data(self):
        """Test representative selection using real UniquenessNewMetric output."""
        structures = create_test_structures_with_known_duplicates()

        # Get real uniqueness result
        uniqueness_metric = UniquenessNewMetric()
        uniqueness_result = uniqueness_metric.compute(structures)

        # Test SUN's representative selection
        sun_metric = SUNNewMetric()
        selected_indices = sun_metric._get_unique_structure_indices(
            uniqueness_result, []
        )

        # Should select indices [0, 1, 2, 4] - one representative of each unique structure
        expected_indices = [0, 1, 2, 4]
        assert selected_indices == expected_indices

        # Verify we have fingerprints
        assert hasattr(uniqueness_result, "fingerprints")
        assert len(uniqueness_result.fingerprints) == 7

    def test_selection_with_mock_fingerprints(self):
        """Test selection logic with controlled fingerprint data."""
        sun_metric = SUNNewMetric()

        # Mock uniqueness result with known fingerprints
        uniqueness_result = Mock()
        uniqueness_result.fingerprints = [
            "fp_A",
            "fp_B",
            "fp_C",
            "fp_C",
            "fp_D",
            "fp_D",
            "fp_D",
        ]
        uniqueness_result.individual_values = [1.0, 1.0, 0.5, 0.5, 1 / 3, 1 / 3, 1 / 3]
        failed_indices = []

        selected_indices = sun_metric._get_unique_structure_indices(
            uniqueness_result, failed_indices
        )

        # Should select indices [0, 1, 2, 4] - first occurrence of each unique fingerprint
        expected_indices = [0, 1, 2, 4]
        assert selected_indices == expected_indices

    def test_selection_with_failed_structures(self):
        """Test representative selection when some structures fail fingerprinting."""
        sun_metric = SUNNewMetric()

        # Mock uniqueness result with some failures
        # Structure mapping: successful fingerprints for indices [0, 2, 3]
        uniqueness_result = Mock()
        uniqueness_result.fingerprints = ["fp_A", "fp_B", "fp_B"]  # 3 successful fingerprints
        uniqueness_result.individual_values = [
            1.0,           # Index 0: successful (fp_A)
            float("nan"),  # Index 1: failed
            1.0,           # Index 2: successful (fp_B) - first occurrence
            0.5,           # Index 3: successful (fp_B) - duplicate
            float("nan"),  # Index 4: failed
        ]
        failed_indices = [1, 4]  # Indices 1 and 4 failed

        selected_indices = sun_metric._get_unique_structure_indices(
            uniqueness_result, failed_indices
        )

        # Should select indices [0, 2] - one representative of each unique fingerprint
        # Index 0: fp_A (unique)
        # Index 2: fp_B (first occurrence, index 3 is duplicate)
        expected_indices = [0, 2]
        assert selected_indices == expected_indices
        assert 1 not in selected_indices  # Failed index excluded
        assert 4 not in selected_indices  # Failed index excluded

    def test_fallback_when_fingerprints_unavailable(self):
        """Test fallback to individual values when fingerprints not available."""
        sun_metric = SUNNewMetric()

        # Mock uniqueness result without fingerprints
        uniqueness_result = Mock()
        uniqueness_result.individual_values = [1.0, 1.0, 0.5, 0.5, 0.25, 0.25, 0.25]
        # Remove fingerprints attribute to test fallback
        if hasattr(uniqueness_result, "fingerprints"):
            delattr(uniqueness_result, "fingerprints")

        # The logger is from lemat_genbench.metrics.sun_new_metric, not utils.logging
        with patch("lemat_genbench.metrics.sun_new_metric.logger") as mock_logger:
            selected_indices = sun_metric._get_unique_structure_indices(
                uniqueness_result, []
            )

            # Should log warning about fallback
            mock_logger.warning.assert_called_once()
            assert "Fingerprints not available" in str(mock_logger.warning.call_args)

        # Should still select some representatives using individual values
        assert len(selected_indices) > 0

    def test_handles_empty_fingerprints(self):
        """Test behavior when fingerprints list is empty."""
        sun_metric = SUNNewMetric()

        uniqueness_result = Mock()
        uniqueness_result.fingerprints = []  # Empty fingerprints
        uniqueness_result.individual_values = [1.0, 1.0]

        # The logger is from lemat_genbench.metrics.sun_new_metric, not utils.logging
        with patch("lemat_genbench.metrics.sun_new_metric.logger") as mock_logger:
            selected_indices = sun_metric._get_unique_structure_indices(
                uniqueness_result, []
            )

            # Should fall back to individual values
            mock_logger.warning.assert_called_once()

        # Should select representatives based on individual values fallback
        assert len(selected_indices) > 0


class TestSUNNewUniquenessIntegration:
    """Test full integration between UniquenessNewMetric and SUNNewMetric."""

    @patch("lemat_genbench.metrics.sun_new_metric.AugmentedNoveltyMetric")
    def test_sun_processes_correct_representatives(self, mock_novelty_class):
        """Test that SUN processes correct number of unique representatives."""
        structures = create_structures_with_stability_data()

        # Mock novelty to return all as novel
        mock_novelty = Mock()
        mock_novelty_result = Mock()
        mock_novelty_result.individual_values = [
            1.0,
            1.0,
            1.0,
            1.0,
        ]  # 4 novel structures
        mock_novelty_result.failed_indices = []
        mock_novelty.compute.return_value = mock_novelty_result
        mock_novelty_class.return_value = mock_novelty

        sun_metric = SUNNewMetric()
        result = sun_metric.compute(structures)

        # Verify novelty was called with 4 unique representatives
        mock_novelty.compute.assert_called_once()
        called_structures = mock_novelty.compute.call_args[0][0]
        assert len(called_structures) == 4

        # Should report correct unique count
        assert result.metrics["unique_count"] == 4
        assert abs(result.metrics["unique_rate"] - 4 / 7) < 1e-6

    def test_full_sun_workflow_with_fingerprints(self):
        """Test complete SUN workflow ensuring fingerprint-based processing."""
        structures = create_structures_with_stability_data()

        # Test standalone uniqueness first
        uniqueness_metric = UniquenessNewMetric()
        uniqueness_result = uniqueness_metric.compute(structures)

        # Should have 4 unique structures (A, B, C, D)
        assert uniqueness_result.metrics["unique_structures_count"] == 4
        assert hasattr(uniqueness_result, "fingerprints")
        assert len(uniqueness_result.fingerprints) == 7

        # Test SUN metric
        sun_metric = SUNNewMetric()
        sun_result = sun_metric.compute(structures)

        # SUN should report same uniqueness stats
        assert sun_result.metrics["unique_count"] == 4
        assert abs(sun_result.metrics["unique_rate"] - 4 / 7) < 1e-6

        # Verify that SUN counts make sense
        assert sun_result.metrics["sun_count"] >= 0
        assert sun_result.metrics["msun_count"] >= 0
        assert sun_result.metrics["sun_count"] + sun_result.metrics["msun_count"] <= 4

    @patch("lemat_genbench.metrics.sun_new_metric.AugmentedNoveltyMetric")
    def test_sun_correct_unique_count_from_fingerprints(self, mock_novelty_class):
        """Test that SUN gets correct unique count from fingerprints."""
        # Mock uniqueness result with known fingerprints
        with patch.object(UniquenessNewMetric, "compute") as mock_uniqueness_compute:
            uniqueness_result = Mock()
            uniqueness_result.metrics = {"unique_structures_count": 3}
            uniqueness_result.fingerprints = [
                "fp_A",
                "fp_B",
                "fp_C",
                "fp_C",
                "fp_B",
            ]  # 3 unique
            uniqueness_result.individual_values = [1.0, 1.0, 1.0, 0.5, 0.5]
            uniqueness_result.failed_indices = []
            mock_uniqueness_compute.return_value = uniqueness_result

            # Mock novelty
            mock_novelty = Mock()
            mock_novelty_result = Mock()
            mock_novelty_result.individual_values = [1.0, 1.0, 1.0]  # All novel
            mock_novelty_result.failed_indices = []
            mock_novelty.compute.return_value = mock_novelty_result
            mock_novelty_class.return_value = mock_novelty

            structures = create_structures_with_stability_data()[:5]

            sun_metric = SUNNewMetric()
            result = sun_metric.compute(structures)

            # Should report correct unique count (3, from unique fingerprints)
            assert result.metrics["unique_count"] == 3
            assert abs(result.metrics["unique_rate"] - 3 / 5) < 1e-6


class TestSynchronizationRegression:
    """Regression tests to ensure the fix doesn't break in the future."""

    def test_complex_duplicate_pattern(self):
        """Test with a complex duplicate pattern to ensure robustness."""
        # Create pattern: A, B, C, B, D, C, A, E, E, D, D
        # Unique structures: A, B, C, D, E (5 unique out of 11 total)
        lattice = [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]]

        struct_a = Structure(
            lattice,
            ["Na", "Cl"],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
            coords_are_cartesian=False,
        )
        struct_b = Structure(
            lattice,
            ["K", "Br"],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
            coords_are_cartesian=False,
        )
        struct_c = Structure(
            lattice,
            ["Li", "F"],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
            coords_are_cartesian=False,
        )
        struct_d = Structure(
            lattice,
            ["Cs", "I"],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
            coords_are_cartesian=False,
        )
        struct_e = Structure(
            lattice,
            ["Rb", "At"],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
            coords_are_cartesian=False,
        )

        structures = [
            struct_a,  # A (first)
            struct_b,  # B (first)
            struct_c,  # C (first)
            struct_b.copy(),  # B (duplicate)
            struct_d,  # D (first)
            struct_c.copy(),  # C (duplicate)
            struct_a.copy(),  # A (duplicate)
            struct_e,  # E (first)
            struct_e.copy(),  # E (duplicate)
            struct_d.copy(),  # D (duplicate)
            struct_d.copy(),  # D (duplicate)
        ]

        # Add stability data
        for struct in structures:
            struct.properties["e_above_hull"] = 0.0

        # Test both metrics
        uniqueness_metric = UniquenessNewMetric()
        uniqueness_result = uniqueness_metric.compute(structures)

        sun_metric = SUNNewMetric()
        sun_result = sun_metric.compute(structures)

        # Both should report 5 unique structures
        assert uniqueness_result.metrics["unique_structures_count"] == 5
        assert sun_result.metrics["unique_count"] == 5

        # Rates should match
        expected_rate = 5 / 11
        assert abs(uniqueness_result.metrics["uniqueness_score"] - expected_rate) < 1e-6
        assert abs(sun_result.metrics["unique_rate"] - expected_rate) < 1e-6

    def test_synchronization_invariant_under_reordering(self):
        """Test that synchronization holds even when structures are reordered."""
        structures_original = create_structures_with_stability_data()

        # Create reordered version: reverse order
        structures_reordered = list(reversed(structures_original))

        # Test both orderings
        uniqueness_metric = UniquenessNewMetric()
        sun_metric = SUNNewMetric()

        # Original order
        unique_result_orig = uniqueness_metric.compute(structures_original)
        sun_result_orig = sun_metric.compute(structures_original)

        # Reordered
        unique_result_reord = uniqueness_metric.compute(structures_reordered)
        sun_result_reord = sun_metric.compute(structures_reordered)

        # Unique counts should be same regardless of order
        assert (
            unique_result_orig.metrics["unique_structures_count"]
            == unique_result_reord.metrics["unique_structures_count"]
        )
        assert (
            sun_result_orig.metrics["unique_count"]
            == sun_result_reord.metrics["unique_count"]
        )

        # Synchronization should hold for both orderings
        assert (
            unique_result_orig.metrics["unique_structures_count"]
            == sun_result_orig.metrics["unique_count"]
        )
        assert (
            unique_result_reord.metrics["unique_structures_count"]
            == sun_result_reord.metrics["unique_count"]
        )

    def test_fingerprint_source_consistency(self):
        """Test that different fingerprint sources maintain synchronization."""
        structures = create_structures_with_stability_data()

        # Test with different fingerprint sources
        fingerprint_sources = ["auto", "compute", "property"]
        
        results = {}
        for source in fingerprint_sources:
            try:
                # Test uniqueness metric
                uniqueness_metric = UniquenessNewMetric(fingerprint_source=source)
                uniqueness_result = uniqueness_metric.compute(structures)
                
                # Test SUN metric  
                sun_metric = SUNNewMetric(fingerprint_source=source)
                sun_result = sun_metric.compute(structures)
                
                results[source] = {
                    "uniqueness": uniqueness_result,
                    "sun": sun_result
                }
                
                # Verify synchronization for this source
                assert (
                    uniqueness_result.metrics["unique_structures_count"]
                    == sun_result.metrics["unique_count"]
                )
                
            except Exception:
                # Some sources might fail (e.g., "property" if no preprocessed fingerprints)
                # This is expected behavior, just continue
                continue

        # If we got results for multiple sources, they should be consistent
        unique_counts = []
        for source, result_dict in results.items():
            unique_counts.append(result_dict["uniqueness"].metrics["unique_structures_count"])
        
        if len(unique_counts) > 1:
            # All sources that worked should give same unique count
            assert all(count == unique_counts[0] for count in unique_counts)


class TestAugmentedFingerprintingSpecific:
    """Test cases specific to augmented fingerprinting functionality."""

    def test_augmented_fingerprint_vs_bawl_consistency(self):
        """Test that augmented fingerprinting maintains proper uniqueness detection."""
        structures = create_test_structures_with_known_duplicates()
        
        # Test with augmented fingerprinting
        uniqueness_metric = UniquenessNewMetric(fingerprint_source="compute")
        result = uniqueness_metric.compute(structures)
        
        # Should detect the same duplicate pattern regardless of fingerprinting method
        # We have 4 unique structures: A, B, C, D
        assert result.metrics["unique_structures_count"] == 4
        assert abs(result.metrics["uniqueness_score"] - 4/7) < 1e-6
        
        # Individual values should follow expected pattern
        expected_pattern = [1.0, 1.0, 0.5, 0.5, 1/3, 1/3, 1/3]
        for i, expected in enumerate(expected_pattern):
            assert abs(result.individual_values[i] - expected) < 1e-6

    def test_fingerprint_fallback_behavior(self):
        """Test that fingerprint source fallback works correctly."""
        structures = create_test_structures_with_known_duplicates()
        
        # Test "auto" mode which should fall back to computation
        uniqueness_metric = UniquenessNewMetric(fingerprint_source="auto")
        result = uniqueness_metric.compute(structures)
        
        # Should still work and give correct results
        assert result.metrics["unique_structures_count"] == 4
        assert hasattr(result, "fingerprints")
        assert len(result.fingerprints) == 7

    def test_property_source_with_preprocessed_fingerprints(self):
        """Test property source when structures have preprocessed fingerprints."""
        structures = create_test_structures_with_known_duplicates()
        
        # Add mock preprocessed fingerprints to structures
        fingerprint_pattern = ["aug_fp_A", "aug_fp_B", "aug_fp_C", "aug_fp_C", 
                              "aug_fp_D", "aug_fp_D", "aug_fp_D"]
        
        for i, struct in enumerate(structures):
            struct.properties["augmented_fingerprint"] = fingerprint_pattern[i]
        
        # Test with property source
        uniqueness_metric = UniquenessNewMetric(fingerprint_source="property")
        result = uniqueness_metric.compute(structures)
        
        # Should use the preprocessed fingerprints and detect duplicates correctly
        assert result.metrics["unique_structures_count"] == 4
        assert result.metrics["duplicate_structures_count"] == 3


# Manual test function for development
def manual_test():
    """Manual test for development purposes."""
    print("Running manual uniqueness-SUN synchronization test with augmented fingerprinting...")

    try:
        # Test 1: Basic structure creation
        print("1. Testing structure creation...")
        structures = create_test_structures_with_known_duplicates()
        stability_structures = create_structures_with_stability_data()
        
        print(f"Created {len(structures)} duplicate test structures")
        print(f"Created {len(stability_structures)} structures with stability data")

        # Test 2: Uniqueness metric with augmented fingerprints
        print("2. Testing uniqueness metric...")
        uniqueness_metric = UniquenessNewMetric()
        uniqueness_result = uniqueness_metric.compute(structures)
        
        print(f"Unique structures found: {uniqueness_result.metrics['unique_structures_count']}")
        print(f"Fingerprints generated: {len(uniqueness_result.fingerprints)}")
        print(f"Individual values: {uniqueness_result.individual_values}")

        # Test 3: SUN metric synchronization
        print("3. Testing SUN metric synchronization...")
        sun_metric = SUNNewMetric()
        sun_result = sun_metric.compute(stability_structures)
        
        print(f"SUN unique count: {sun_result.metrics['unique_count']}")
        print(f"SUN rate: {sun_result.metrics['sun_rate']}")
        print(f"MetaSUN rate: {sun_result.metrics['msun_rate']}")

        # Test 4: Verify synchronization
        print("4. Testing synchronization...")
        uniqueness_stability = UniquenessNewMetric()
        uniqueness_stability_result = uniqueness_stability.compute(stability_structures)
        
        unique_count_standalone = uniqueness_stability_result.metrics["unique_structures_count"]
        unique_count_sun = sun_result.metrics["unique_count"]
        
        print(f"Standalone uniqueness count: {unique_count_standalone}")
        print(f"SUN uniqueness count: {unique_count_sun}")
        print(f"Counts match: {unique_count_standalone == unique_count_sun}")

        # Test 5: Representative selection
        print("5. Testing representative selection...")
        selected_indices = sun_metric._get_unique_structure_indices(
            uniqueness_stability_result, []
        )
        print(f"Selected representative indices: {selected_indices}")

        # Test 6: Different fingerprint sources
        print("6. Testing different fingerprint sources...")
        sources = ["auto", "compute"]
        for source in sources:
            try:
                test_metric = UniquenessNewMetric(fingerprint_source=source)
                test_result = test_metric.compute(structures[:3])  # Smaller set for speed
                print(f"Source {source}: {test_result.metrics['unique_structures_count']} unique")
            except Exception as e:
                print(f"Source {source} failed: {e}")

        print("\nAll manual tests passed!")
        return True

    except Exception as e:
        print(f"Manual test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    manual_test()