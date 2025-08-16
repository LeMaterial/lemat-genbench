"""Tests for the enhanced novelty metric using augmented fingerprinting."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pymatgen.core.structure import Structure

from lemat_genbench.metrics.base import MetricResult
from lemat_genbench.metrics.novelty_new_metric import (
    AugmentedNoveltyMetric,
    create_augmented_novelty_metric,
    create_computation_based_novelty_metric,
    create_property_based_novelty_metric,
    create_robust_novelty_metric,
)


def create_test_structures_with_fingerprints():
    """Create test structures with augmented fingerprints attached."""
    structures = []

    # Structure 1: NaCl with augmented fingerprint
    lattice1 = [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]]
    structure1 = Structure(
        lattice=lattice1,
        species=["Na", "Cl"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structure1.properties["augmented_fingerprint"] = "AUG_225_Na1Cl1_test1"
    structures.append(structure1)

    # Structure 2: CsCl with different fingerprint
    lattice2 = [[4.2, 0, 0], [0, 4.2, 0], [0, 0, 4.2]]
    structure2 = Structure(
        lattice=lattice2,
        species=["Cs", "Cl"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structure2.properties["augmented_fingerprint"] = "AUG_221_Cs1Cl1_test2"
    structures.append(structure2)

    # Structure 3: Another NaCl variant (same composition, different fingerprint)
    lattice3 = [[3.8, 0, 0], [0, 3.8, 0], [0, 0, 3.8]]
    structure3 = Structure(
        lattice=lattice3,
        species=["Na", "Cl"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structure3.properties["augmented_fingerprint"] = "AUG_225_Na1Cl1_test3"
    structures.append(structure3)

    # Structure 4: Without fingerprint (for testing computation fallback)
    lattice4 = [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]]
    structure4 = Structure(
        lattice=lattice4,
        species=["K", "Br"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    # No fingerprint attached for testing computation
    structures.append(structure4)

    return structures


class TestAugmentedNoveltyMetric:
    """Test suite for AugmentedNoveltyMetric class."""

    def test_initialization(self):
        """Test metric initialization with default parameters."""
        metric = AugmentedNoveltyMetric()
        
        assert metric.name == "AugmentedNovelty"
        assert metric.config.reference_dataset_name == "LeMat-Bulk"
        assert metric.config.fingerprint_source == "auto"
        assert metric.config.symprec == 0.01
        assert metric.config.angle_tolerance == 5.0
        assert metric.config.fallback_to_computation is True

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        metric = AugmentedNoveltyMetric(
            reference_fingerprints_path="/custom/path",
            reference_dataset_name="CustomDataset",
            fingerprint_source="property",
            symprec=0.1,
            angle_tolerance=10.0,
            fallback_to_computation=False,
            name="CustomNoveltyMetric",
        )
        
        assert metric.name == "CustomNoveltyMetric"
        assert metric.config.reference_fingerprints_path == "/custom/path"
        assert metric.config.reference_dataset_name == "CustomDataset"
        assert metric.config.fingerprint_source == "property"
        assert metric.config.symprec == 0.1
        assert metric.config.angle_tolerance == 10.0
        assert metric.config.fallback_to_computation is False

    def test_load_reference_fingerprints_success(self):
        """Test successful loading of reference fingerprints."""
        # Create a temporary parquet file for testing
        import pandas as pd
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock parquet file
            test_fingerprints = [
                "AUG_225_Na1Cl1_ref1", 
                "AUG_221_Cs1Cl1_ref2",
                "AUG_227_Ca1F2_ref3"
            ]
            df = pd.DataFrame({"values": test_fingerprints})
            parquet_path = Path(temp_dir) / "unique_fingerprints.parquet"
            df.to_parquet(parquet_path)
            
            # Test loading
            metric = AugmentedNoveltyMetric(reference_fingerprints_path=temp_dir)
            fingerprints = metric._load_reference_fingerprints()
            
            assert len(fingerprints) == 3
            assert "AUG_225_Na1Cl1_ref1" in fingerprints
            assert "AUG_221_Cs1Cl1_ref2" in fingerprints
            assert "AUG_227_Ca1F2_ref3" in fingerprints
            assert metric._reference_loaded is True

    def test_load_reference_fingerprints_empty_directory(self):
        """Test handling when no fingerprint files exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Empty directory
            metric = AugmentedNoveltyMetric(reference_fingerprints_path=temp_dir)
            fingerprints = metric._load_reference_fingerprints()
            
            assert len(fingerprints) == 0
            assert isinstance(fingerprints, set)

    def test_load_reference_fingerprints_pickle_fallback(self):
        """Test fallback to pickle file when parquet doesn't exist."""
        import pickle
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock pickle file
            test_fingerprints = {"AUG_225_test1", "AUG_227_test2"}
            pickle_path = Path(temp_dir) / "unique_fingerprints.pkl"
            with open(pickle_path, 'wb') as f:
                pickle.dump(test_fingerprints, f)
            
            # Test loading
            metric = AugmentedNoveltyMetric(reference_fingerprints_path=temp_dir)
            fingerprints = metric._load_reference_fingerprints()
            
            assert len(fingerprints) == 2
            assert "AUG_225_test1" in fingerprints
            assert "AUG_227_test2" in fingerprints

    def test_get_structure_fingerprint_from_property(self):
        """Test fingerprint extraction from structure properties."""
        structures = create_test_structures_with_fingerprints()
        metric = AugmentedNoveltyMetric(fingerprint_source="property")
        
        # Should get fingerprint from properties
        fp = metric._get_structure_fingerprint(structures[0])
        assert fp == "AUG_225_Na1Cl1_test1"
        
        # Structure without fingerprint should return None
        fp_none = metric._get_structure_fingerprint(structures[3])
        assert fp_none is None

    @patch("lemat_genbench.metrics.novelty_new_metric.get_augmented_fingerprint")
    def test_get_structure_fingerprint_computation(self, mock_get_fp):
        """Test fingerprint computation fallback."""
        mock_get_fp.return_value = "AUG_computed_fingerprint"
        
        structures = create_test_structures_with_fingerprints()
        metric = AugmentedNoveltyMetric(fingerprint_source="compute")
        
        # Should compute fingerprint
        fp = metric._get_structure_fingerprint(structures[3])
        assert fp == "AUG_computed_fingerprint"
        mock_get_fp.assert_called_once()

    @patch("lemat_genbench.metrics.novelty_new_metric.get_augmented_fingerprint")
    def test_get_structure_fingerprint_auto_mode(self, mock_get_fp):
        """Test auto mode: property first, then computation."""
        mock_get_fp.return_value = "AUG_computed_fingerprint"
        
        structures = create_test_structures_with_fingerprints()
        metric = AugmentedNoveltyMetric(fingerprint_source="auto")
        
        # Structure with property - should use property
        fp1 = metric._get_structure_fingerprint(structures[0])
        assert fp1 == "AUG_225_Na1Cl1_test1"
        mock_get_fp.assert_not_called()
        
        # Structure without property - should compute
        fp2 = metric._get_structure_fingerprint(structures[3])
        assert fp2 == "AUG_computed_fingerprint"
        mock_get_fp.assert_called_once()

    def test_compute_structure_novel(self):
        """Test compute_structure for novel structures."""
        structures = create_test_structures_with_fingerprints()
        
        # Mock reference set without the test fingerprints
        reference_fps = {"AUG_ref1", "AUG_ref2", "AUG_ref3"}
        
        # Test novel structure
        result = AugmentedNoveltyMetric.compute_structure(
            structures[0], reference_fps, "property", 0.01, 5.0, True
        )
        assert result == 1.0  # Novel

    def test_compute_structure_known(self):
        """Test compute_structure for known structures."""
        structures = create_test_structures_with_fingerprints()
        
        # Mock reference set containing the test fingerprint
        reference_fps = {"AUG_225_Na1Cl1_test1", "AUG_ref2", "AUG_ref3"}
        
        # Test known structure
        result = AugmentedNoveltyMetric.compute_structure(
            structures[0], reference_fps, "property", 0.01, 5.0, True
        )
        assert result == 0.0  # Known

    @patch("lemat_genbench.metrics.novelty_new_metric.get_augmented_fingerprint")
    def test_compute_structure_computation_mode(self, mock_get_fp):
        """Test compute_structure with computation mode."""
        mock_get_fp.return_value = "AUG_computed_test"
        
        structures = create_test_structures_with_fingerprints()
        reference_fps = {"AUG_ref1", "AUG_ref2"}
        
        # Should compute and find novel
        result = AugmentedNoveltyMetric.compute_structure(
            structures[3], reference_fps, "compute", 0.01, 5.0, True
        )
        assert result == 1.0
        mock_get_fp.assert_called_once()

    def test_compute_structure_error_handling(self):
        """Test error handling in compute_structure."""
        structures = create_test_structures_with_fingerprints()
        reference_fps = {"AUG_ref1"}
        
        # Remove fingerprint to test missing fingerprint case
        structure_no_fp = structures[3]  # This one has no fingerprint
        
        result = AugmentedNoveltyMetric.compute_structure(
            structure_no_fp, reference_fps, "property", 0.01, 5.0, False
        )
        # Should return NaN when fingerprint unavailable and computation disabled
        assert np.isnan(result)

    def test_aggregate_results(self):
        """Test result aggregation."""
        metric = AugmentedNoveltyMetric()
        
        # Test with mixed results
        values = [1.0, 0.0, 1.0, 1.0]  # 3 novel, 1 known
        result = metric.aggregate_results(values)
        
        assert result["metrics"]["novelty_score"] == 0.75
        assert result["metrics"]["novel_structures_count"] == 3
        assert result["metrics"]["total_structures_evaluated"] == 4
        assert result["metrics"]["total_structures_attempted"] == 4
        assert result["metrics"]["fingerprinting_success_rate"] == 1.0
        assert result["primary_metric"] == "novelty_score"

    def test_aggregate_results_with_nan(self):
        """Test aggregation with failed fingerprinting (NaN values)."""
        metric = AugmentedNoveltyMetric()
        
        # Include NaN values for failed fingerprinting
        values = [1.0, 0.0, float("nan"), 1.0, float("nan")]
        result = metric.aggregate_results(values)
        
        assert result["metrics"]["novelty_score"] == 2.0 / 3.0  # 2 novel out of 3 valid
        assert result["metrics"]["novel_structures_count"] == 2
        assert result["metrics"]["total_structures_evaluated"] == 3  # Only valid ones
        assert result["metrics"]["total_structures_attempted"] == 5  # All attempted
        assert result["metrics"]["fingerprinting_success_rate"] == 0.6  # 3/5

    def test_aggregate_results_all_failed(self):
        """Test aggregation when all fingerprinting failed."""
        metric = AugmentedNoveltyMetric()
        
        values = [float("nan"), float("nan"), float("nan")]
        result = metric.aggregate_results(values)
        
        assert np.isnan(result["metrics"]["novelty_score"])
        assert result["metrics"]["novel_structures_count"] == 0
        assert result["metrics"]["total_structures_evaluated"] == 0
        assert result["metrics"]["total_structures_attempted"] == 3
        assert result["metrics"]["fingerprinting_success_rate"] == 0.0

    def test_full_workflow_with_temp_data(self):
        """Test complete workflow with temporary test data."""
        import pandas as pd
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create reference fingerprints
            reference_fps = [
                "AUG_ref1", "AUG_ref2", "AUG_225_Na1Cl1_test1"  # Include one known
            ]
            df = pd.DataFrame({"values": reference_fps})
            parquet_path = Path(temp_dir) / "unique_fingerprints.parquet"
            df.to_parquet(parquet_path)
            
            # Create metric and structures
            metric = AugmentedNoveltyMetric(
                reference_fingerprints_path=temp_dir,
                fingerprint_source="property"
            )
            structures = create_test_structures_with_fingerprints()[:3]  # Use first 3
            
            # Evaluate
            result = metric.compute(structures)
            
            # Check result
            assert isinstance(result, MetricResult)
            assert "novelty_score" in result.metrics
            assert result.metrics["total_structures_evaluated"] == 3
            # First structure should be known (0), others novel (1)
            assert result.metrics["novel_structures_count"] == 2
            assert result.metrics["novelty_score"] == 2.0 / 3.0

    def test_factory_functions(self):
        """Test factory functions for creating metrics."""
        # Test basic factory
        metric1 = create_augmented_novelty_metric()
        assert isinstance(metric1, AugmentedNoveltyMetric)
        assert metric1.config.fingerprint_source == "auto"
        
        # Test property-based factory
        metric2 = create_property_based_novelty_metric()
        assert metric2.config.fingerprint_source == "property"
        assert metric2.config.fallback_to_computation is False
        
        # Test computation-based factory
        metric3 = create_computation_based_novelty_metric()
        assert metric3.config.fingerprint_source == "compute"
        
        # Test robust factory
        metric4 = create_robust_novelty_metric()
        assert metric4.config.symprec == 0.1
        assert metric4.config.angle_tolerance == 10.0

    def test_caching_behavior(self):
        """Test that reference fingerprints are cached properly."""
        import pandas as pd
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data
            test_fps = ["AUG_ref1", "AUG_ref2"]
            df = pd.DataFrame({"values": test_fps})
            parquet_path = Path(temp_dir) / "unique_fingerprints.parquet"
            df.to_parquet(parquet_path)
            
            metric = AugmentedNoveltyMetric(reference_fingerprints_path=temp_dir)
            
            # First call should load
            fps1 = metric._load_reference_fingerprints()
            assert len(fps1) == 2
            
            # Second call should use cache
            fps2 = metric._load_reference_fingerprints()
            assert fps1 == fps2
            assert metric._reference_loaded is True

    def test_custom_reference_path(self):
        """Test using custom reference fingerprints path."""
        custom_path = "/custom/fingerprints/path"
        metric = AugmentedNoveltyMetric(reference_fingerprints_path=custom_path)
        
        assert metric.config.reference_fingerprints_path == custom_path

    def test_error_handling_with_malformed_fingerprints(self):
        """Test handling of malformed fingerprints."""
        metric = AugmentedNoveltyMetric(fingerprint_source="property")
        
        # Create structure with malformed fingerprint
        lattice = [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]]
        structure = Structure(
            lattice=lattice,
            species=["Na", "Cl"],
            coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
            coords_are_cartesian=False,
        )
        
        # Test various malformed fingerprint cases
        test_cases = [
            None,  # No fingerprint
            "",    # Empty string
            123,   # Wrong type
            [],    # Wrong type
        ]
        
        for bad_fp in test_cases:
            structure.properties["augmented_fingerprint"] = bad_fp
            fp = metric._get_structure_fingerprint(structure)
            # Should handle gracefully, either return None or string representation
            if fp is not None:
                assert isinstance(fp, str)

    def test_fingerprint_filtering(self):
        """Test that invalid fingerprints are filtered out."""
        import pandas as pd
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create data with some invalid fingerprints
            test_fingerprints = [
                "AUG_valid_fp_1",
                "",  # Empty string
                "AUG_valid_fp_2", 
                None,  # None value
                "   ",  # Whitespace only
                "AUG_valid_fp_3"
            ]
            df = pd.DataFrame({"values": test_fingerprints})
            parquet_path = Path(temp_dir) / "unique_fingerprints.parquet"
            df.to_parquet(parquet_path)
            
            metric = AugmentedNoveltyMetric(reference_fingerprints_path=temp_dir)
            fingerprints = metric._load_reference_fingerprints()
            
            # Should only have valid fingerprints
            assert len(fingerprints) == 3
            assert "AUG_valid_fp_1" in fingerprints
            assert "AUG_valid_fp_2" in fingerprints
            assert "AUG_valid_fp_3" in fingerprints
            assert "" not in fingerprints
            assert None not in fingerprints


class TestIntegrationWithRealFingerprints:
    """Integration tests that would work with actual fingerprint files."""
    
    def test_metric_with_actual_data_structure(self):
        """Test metric behavior with data structures like the real implementation."""
        # This test simulates the structure of actual fingerprint data
        metric = AugmentedNoveltyMetric()
        
        # Test with fingerprints that look like real augmented fingerprints
        test_structures = []
        
        # Create structures with realistic fingerprints
        lattice = [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]]
        for i, fp in enumerate([
            "AUG_225_Na:4a:1_1|Cl:4b:1_1",
            "AUG_227_Ca:4a:1_1|F:8c:1_2",
            "AUG_229_Li:1a:1_1|H:1b:1_1"
        ]):
            structure = Structure(
                lattice=lattice,
                species=["Na", "Cl"] if i == 0 else ["Ca", "F"] if i == 1 else ["Li", "H"],
                coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
                coords_are_cartesian=False,
            )
            structure.properties["augmented_fingerprint"] = fp
            test_structures.append(structure)
        
        # Test fingerprint extraction
        for structure in test_structures:
            fp = metric._get_structure_fingerprint(structure)
            assert fp is not None
            assert fp.startswith("AUG_")
            assert "_" in fp  # Should have the expected format


# Manual test function for development and debugging
def manual_test_augmented_novelty():
    """Manual test function for development."""
    print("🧪 Running manual augmented novelty metric test...")
    
    try:
        # Test basic functionality
        print("1. Testing basic initialization...")
        metric = AugmentedNoveltyMetric()
        print("✅ Metric initialized successfully")
        
        # Test fingerprint extraction
        print("2. Testing fingerprint extraction...")
        structures = create_test_structures_with_fingerprints()
        
        for i, structure in enumerate(structures[:3]):
            fp = metric._get_structure_fingerprint(structure)
            print(f"   Structure {i+1}: {fp}")
        
        # Test with temporary reference data
        print("3. Testing with temporary reference...")
        import pandas as pd
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock reference
            mock_reference = ["AUG_ref1", "AUG_ref2", "AUG_225_Na1Cl1_test1"]
            df = pd.DataFrame({"values": mock_reference})
            parquet_path = Path(temp_dir) / "unique_fingerprints.parquet"
            df.to_parquet(parquet_path)
            
            metric_with_ref = AugmentedNoveltyMetric(reference_fingerprints_path=temp_dir)
            result = metric_with_ref.compute(structures[:3])
            print(f"   Novelty score: {result.metrics['novelty_score']:.3f}")
            print(f"   Novel count: {result.metrics['novel_structures_count']}")
            print(f"   Total evaluated: {result.metrics['total_structures_evaluated']}")
        
        print("\n✅ All manual tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run manual test for development
    manual_test_augmented_novelty()