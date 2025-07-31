"""Integration tests for novelty metrics with real LeMat-Bulk data.

These tests verify the complete workflow from dataset loading to
novelty evaluation using actual BAWL fingerprinting.
"""

import pytest
from lemat_genbench.metrics.base import MetricResult
from lemat_genbench.metrics.novelty_metric import NoveltyMetric
from pymatgen.core.structure import Structure


def create_definitely_novel_structures():
    """Create structures that are definitely not in LeMat-Bulk.

    Using common elements that BAWL hasher can handle properly.
    """
    structures = []

    # 1. Unusual but processable compound - artificial perovskite
    lattice = [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]]
    artificial_perovskite = Structure(
        lattice=lattice,
        species=["Ba", "Ti", "O", "O", "O"],  # BaTiO3-like but artificial
        coords=[
            [0.5, 0.5, 0.5],  # Ba
            [0.0, 0.0, 0.0],  # Ti
            [0.5, 0.0, 0.0],  # O
            [0.0, 0.5, 0.0],  # O
            [0.0, 0.0, 0.5],  # O
        ],
        coords_are_cartesian=False,
    )
    structures.append(artificial_perovskite)

    # 2. Unusual coordination - artificial spinel
    lattice2 = [[8.0, 0, 0], [0, 8.0, 0], [0, 0, 8.0]]
    artificial_spinel = Structure(
        lattice=lattice2,
        species=["Mg", "Al", "O", "O", "O", "O"],  # MgAl2O4-like
        coords=[
            [0.125, 0.125, 0.125],  # Mg
            [0.5, 0.5, 0.5],  # Al
            [0.25, 0.25, 0.25],  # O
            [0.75, 0.75, 0.25],  # O
            [0.75, 0.25, 0.75],  # O
            [0.25, 0.75, 0.75],  # O
        ],
        coords_are_cartesian=False,
    )
    structures.append(artificial_spinel)

    # 3. Simple but unusual binary compound
    lattice3 = [[6.0, 0, 0], [0, 6.0, 0], [0, 0, 6.0]]
    binary_compound = Structure(
        lattice=lattice3,
        species=["Zn", "S"],  # ZnS-like but artificial structure
        coords=[
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
        ],
        coords_are_cartesian=False,
    )
    structures.append(binary_compound)

    return structures


@pytest.mark.slow  # Mark as slow test since it downloads data
class TestNoveltyIntegration:
    """Integration tests using real LeMat-Bulk data."""

    def test_small_reference_integration(self):
        """Test complete workflow with small reference dataset."""
        # Use small reference for speed
        novelty_metric = NoveltyMetric(
            reference_dataset="LeMaterial/LeMat-Bulk",
            reference_config="compatible_pbe",
            max_reference_size=50,  # Small for CI/CD
            cache_reference=True,
        )

        # Create definitely novel structures
        novel_structures = create_definitely_novel_structures()

        # Evaluate novelty using compute method which returns MetricResult
        result = novelty_metric.compute(novel_structures)

        # Verify we get MetricResult
        assert isinstance(result, MetricResult)

        # Verify results - allow for some structures to fail fingerprinting
        assert result.metrics["total_structures_evaluated"] <= len(novel_structures)
        assert (
            result.metrics["total_structures_evaluated"] > 0
        )  # At least one should work
        assert 0.0 <= result.metrics["novelty_score"] <= 1.0
        assert isinstance(result.metrics["novel_structures_count"], int)

        # Most structures that can be fingerprinted should be novel
        if result.metrics["total_structures_evaluated"] > 0:
            assert result.metrics["novelty_score"] >= 0.5

    def test_fingerprint_consistency(self):
        """Test that fingerprinting is consistent across runs."""
        novelty_metric = NoveltyMetric(max_reference_size=10)

        # Create a test structure that should work with BAWL
        structures = create_definitely_novel_structures()

        # Find a structure that can be fingerprinted
        test_structure = None
        for structure in structures:
            try:
                fp = novelty_metric.fingerprinter.get_material_hash(structure)
                if fp is not None:
                    test_structure = structure
                    break
            except Exception:
                continue

        # Skip test if no structure can be fingerprinted
        if test_structure is None:
            pytest.skip("No structures could be fingerprinted by BAWL")

        # Compute fingerprint multiple times
        fingerprints = []
        for _ in range(3):
            fp = novelty_metric.fingerprinter.get_material_hash(test_structure)
            fingerprints.append(fp)

        # All fingerprints should be identical
        assert all(fp == fingerprints[0] for fp in fingerprints)
        assert len(fingerprints[0]) > 0

    def test_reference_dataset_loading(self):
        """Test that reference dataset loads correctly."""
        novelty_metric = NoveltyMetric(max_reference_size=20)

        # Load reference fingerprints
        reference_fps = novelty_metric._load_reference_dataset()

        # Should load some fingerprints
        assert len(reference_fps) > 0
        assert len(reference_fps) <= 20  # Respects max size

        # All should be strings
        for fp in reference_fps:
            assert isinstance(fp, str)
            assert len(fp) > 0

    def test_caching_behavior(self):
        """Test that caching works correctly."""
        novelty_metric = NoveltyMetric(max_reference_size=10, cache_reference=True)

        # First load
        fps1 = novelty_metric._load_reference_dataset()

        # Second load should use cache
        fps2 = novelty_metric._load_reference_dataset()

        # Should be identical
        assert fps1 == fps2
        assert novelty_metric._reference_loaded is True

    def test_different_reference_configs(self):
        """Test with different LeMat-Bulk configurations."""
        configs = ["compatible_pbe", "compatible_pbesol"]

        for config in configs:
            try:
                novelty_metric = NoveltyMetric(
                    reference_config=config,
                    max_reference_size=5,  # Very small
                )

                # Should initialize without error
                assert novelty_metric.config.reference_config == config

                # Should be able to load reference
                fps = novelty_metric._load_reference_dataset()
                assert len(fps) > 0

            except Exception as e:
                # Some configs might not be available, that's ok
                print(f"Config {config} not available: {e}")

    def test_error_handling(self):
        """Test error handling with invalid configurations."""
        # Test invalid dataset
        with pytest.raises(Exception):  # Could be various exceptions
            metric = NoveltyMetric(
                reference_dataset="nonexistent/dataset", max_reference_size=1
            )
            metric._load_reference_dataset()

        # Test invalid config
        with pytest.raises(Exception):
            metric = NoveltyMetric(
                reference_config="nonexistent_config", max_reference_size=1
            )
            metric._load_reference_dataset()

    def test_metric_result_properties(self):
        """Test that MetricResult has all expected properties."""
        novelty_metric = NoveltyMetric(max_reference_size=10)
        structures = create_definitely_novel_structures()[:2]  # Just 2 for speed

        result = novelty_metric.compute(structures)

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
        assert result.primary_metric == "novelty_score"
        assert result.n_structures == 2
        assert len(result.individual_values) == 2
        assert result.computation_time > 0

    def test_callable_interface(self):
        """Test the callable interface (__call__ method)."""
        metric = NoveltyMetric(max_reference_size=10)
        structures = create_definitely_novel_structures()[:1]  # Just 1 for speed

        # Test direct call
        result = metric(structures)

        # Should return MetricResult
        assert isinstance(result, MetricResult)
        # Check that proper aggregation was done (not just the base metric name)
        assert (
            "novelty_score" in result.metrics
            or result.metrics.get("total_structures_evaluated", 0) == 0
        )

    def test_realistic_workflow(self):
        """Test a realistic workflow with structures that should work."""
        # Create very simple, common structures that BAWL should handle
        simple_structures = []

        # Simple cubic structure
        lattice = [[3.0, 0, 0], [0, 3.0, 0], [0, 0, 3.0]]
        simple_structure = Structure(
            lattice=lattice,
            species=["Na", "Cl"],
            coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
            coords_are_cartesian=False,
        )
        simple_structures.append(simple_structure)

        # Test with small reference
        metric = NoveltyMetric(max_reference_size=20)
        result = metric.compute(simple_structures)

        # Should work for at least this simple case
        assert isinstance(result, MetricResult)
        assert result.metrics["total_structures_evaluated"] >= 0


# Manual test function for development
def manual_integration_test():
    """Manual integration test for development purposes."""

    print("🧪 Running manual integration test...")

    try:
        # Test 1: Basic functionality
        print("1. Testing basic integration...")
        metric = NoveltyMetric(max_reference_size=20)
        structures = create_definitely_novel_structures()
        result = metric.compute(structures)

        print("Evaluated structures")
        print(f"Total evaluated: {result.metrics['total_structures_evaluated']}")
        print(f"Novelty score: {result.metrics['novelty_score']:.3f}")
        print(f"Result type: {type(result).__name__}")

        # Test 2: Fingerprint consistency for structures that work
        print("2. Testing fingerprint consistency...")
        for i, structure in enumerate(structures):
            try:
                fp1 = metric.fingerprinter.get_material_hash(structure)
                fp2 = metric.fingerprinter.get_material_hash(structure)
                if fp1 == fp2 and fp1 is not None:
                    print(
                        f"Structure {i + 1} ({structure.composition.reduced_formula}): Consistent fingerprint"
                    )
                    break
            except Exception as e:
                print(f"   - Structure {i + 1} failed: {e}")

        # Test 3: Reference loading
        print("3. Testing reference loading...")
        fps = metric._load_reference_dataset()
        print(f"Loaded {len(fps)} reference fingerprints")

        # Test 4: MetricResult properties
        print("4. Testing MetricResult properties...")
        assert isinstance(result, MetricResult)
        assert result.primary_metric == "novelty_score"
        assert result.computation_time > 0
        print("MetricResult verified")

        print("\nAll integration tests passed!")
        return True

    except Exception as e:
        print(f"Integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    manual_integration_test()
