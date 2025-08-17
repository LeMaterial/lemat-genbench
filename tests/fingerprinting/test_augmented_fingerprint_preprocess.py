"""Tests for augmented fingerprint preprocessor functionality."""

import pytest
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

# Use the newer MatSciTest to avoid deprecation warnings
try:
    from pymatgen.util.testing import MatSciTest as PymatgenTest
except ImportError:
    # Fallback to older PymatgenTest if MatSciTest is not available
    from pymatgen.util.testing import PymatgenTest

from lemat_genbench.preprocess.augmented_fingerprint_preprocess import (
    AugmentedFingerprintPreprocessor,
    create_augmented_fingerprint_preprocessor,
    create_high_precision_fingerprint_preprocessor,
    create_robust_fingerprint_preprocessor,
)


@pytest.fixture
def test_structures():
    """Create test structures for fingerprint preprocessing."""
    structures = {}
    
    # Simple NaCl structure
    lattice_nacl = Lattice.cubic(5.64)
    nacl = Structure(
        lattice_nacl,
        ["Na", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    structures["nacl"] = nacl
    
    # CsCl structure
    lattice_cscl = Lattice.cubic(4.11)
    cscl = Structure(
        lattice_cscl, 
        ["Cs", "Cl"], 
        [[0, 0, 0], [0.5, 0.5, 0.5]], 
        coords_are_cartesian=False
    )
    structures["cscl"] = cscl
    
    # Fluorite structure (more complex)
    lattice_caf2 = Lattice.cubic(5.46)
    caf2 = Structure(
        lattice_caf2,
        ["Ca", "F", "F"],
        [[0, 0, 0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
        coords_are_cartesian=False,
    )
    structures["caf2"] = caf2
    
    return structures


@pytest.fixture
def pymatgen_test_structures():
    """Create test structures using pymatgen testing utilities."""
    test = PymatgenTest()
    return [
        test.get_structure("Si"),
        test.get_structure("LiFePO4"),
    ]


class TestAugmentedFingerprintPreprocessor:
    """Test cases for AugmentedFingerprintPreprocessor."""

    def test_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = AugmentedFingerprintPreprocessor()
        assert preprocessor.config.symprec == 0.01
        assert preprocessor.config.angle_tolerance == 5.0
        assert preprocessor.config.include_fallback_properties is True
        assert preprocessor.config.name == "AugmentedFingerprintPreprocessor"

        # Test custom parameters
        preprocessor_custom = AugmentedFingerprintPreprocessor(
            symprec=0.1, 
            angle_tolerance=10.0,
            include_fallback_properties=False,
            name="CustomFingerprinter"
        )
        assert preprocessor_custom.config.symprec == 0.1
        assert preprocessor_custom.config.angle_tolerance == 10.0
        assert preprocessor_custom.config.include_fallback_properties is False
        assert preprocessor_custom.config.name == "CustomFingerprinter"

    def test_process_single_structure(self, test_structures):
        """Test processing a single structure."""
        preprocessor = AugmentedFingerprintPreprocessor()
        
        # Test NaCl structure
        nacl = test_structures["nacl"]
        processed_nacl = preprocessor.process_structure(nacl)
        
        # Check that fingerprint properties were added
        assert "augmented_fingerprint" in processed_nacl.properties
        assert "augmented_fingerprint_properties" in processed_nacl.properties
        
        fingerprint_props = processed_nacl.properties["augmented_fingerprint_properties"]
        assert "augmented_fingerprint_success" in fingerprint_props
        assert "augmented_fingerprint_symprec" in fingerprint_props
        assert "augmented_fingerprint_angle_tolerance" in fingerprint_props
        
        # If fingerprinting succeeded, check fingerprint format
        fingerprint = processed_nacl.properties["augmented_fingerprint"]
        if fingerprint is not None:
            assert isinstance(fingerprint, str)
            assert fingerprint.startswith("AUG_")
            print(f"NaCl fingerprint: {fingerprint}")
        else:
            print("NaCl fingerprinting failed, checking fallback properties")
            
    def test_process_structure_with_fallback(self, test_structures):
        """Test processing with fallback properties enabled."""
        preprocessor = AugmentedFingerprintPreprocessor(include_fallback_properties=True)
        
        cscl = test_structures["cscl"]
        processed_cscl = preprocessor.process_structure(cscl)
        
        fingerprint_props = processed_cscl.properties["augmented_fingerprint_properties"]
        
        # Should have either successful fingerprint or fallback properties
        if not fingerprint_props["augmented_fingerprint_success"]:
            # Check for fallback properties
            assert any(key.startswith("augmented_fingerprint_fallback") 
                      for key in fingerprint_props.keys())

    def test_process_multiple_structures(self, pymatgen_test_structures):
        """Test processing multiple structures using the preprocessor."""
        preprocessor = AugmentedFingerprintPreprocessor()
        
        # Process structures using the preprocessor's run method
        result = preprocessor(pymatgen_test_structures)
        
        # Check result structure
        assert len(result.processed_structures) <= len(pymatgen_test_structures)
        assert result.config.name == "AugmentedFingerprintPreprocessor"
        assert result.computation_time > 0
        
        # Check that all processed structures have fingerprint properties
        for structure in result.processed_structures:
            assert "augmented_fingerprint" in structure.properties
            assert "augmented_fingerprint_properties" in structure.properties
            
            fingerprint = structure.properties["augmented_fingerprint"]
            if fingerprint is not None:
                assert isinstance(fingerprint, str)
                print(f"Structure {structure.formula} fingerprint: {fingerprint[:50]}...")

    def test_fingerprint_consistency(self, test_structures):
        """Test that fingerprints are consistent across multiple calls."""
        preprocessor = AugmentedFingerprintPreprocessor()
        
        nacl = test_structures["nacl"]
        
        # Process the same structure multiple times
        processed_1 = preprocessor.process_structure(nacl.copy())
        processed_2 = preprocessor.process_structure(nacl.copy())
        
        fingerprint_1 = processed_1.properties["augmented_fingerprint"]
        fingerprint_2 = processed_2.properties["augmented_fingerprint"]
        
        # Fingerprints should be identical for the same structure
        assert fingerprint_1 == fingerprint_2
        
    def test_different_structures_different_fingerprints(self, test_structures):
        """Test that different structures produce different fingerprints."""
        preprocessor = AugmentedFingerprintPreprocessor()
        
        nacl = test_structures["nacl"]
        cscl = test_structures["cscl"]
        
        processed_nacl = preprocessor.process_structure(nacl)
        processed_cscl = preprocessor.process_structure(cscl)
        
        fingerprint_nacl = processed_nacl.properties["augmented_fingerprint"]
        fingerprint_cscl = processed_cscl.properties["augmented_fingerprint"]
        
        # Different structures should have different fingerprints (if both succeeded)
        if fingerprint_nacl is not None and fingerprint_cscl is not None:
            assert fingerprint_nacl != fingerprint_cscl
            print(f"NaCl: {fingerprint_nacl}")
            print(f"CsCl: {fingerprint_cscl}")

    def test_complex_structure_fingerprinting(self, test_structures):
        """Test fingerprinting of more complex structures."""
        preprocessor = AugmentedFingerprintPreprocessor()
        
        caf2 = test_structures["caf2"]
        processed_caf2 = preprocessor.process_structure(caf2)
        
        fingerprint = processed_caf2.properties["augmented_fingerprint"]
        fingerprint_props = processed_caf2.properties["augmented_fingerprint_properties"]
        
        if fingerprint is not None:
            # Complex structures should have substantial fingerprints
            assert len(fingerprint) > 30
            print(f"CaF2 fingerprint length: {len(fingerprint)} characters")
            
            # Should have enumeration information
            if "augmented_fingerprint_n_enumerations" in fingerprint_props:
                n_enumerations = fingerprint_props["augmented_fingerprint_n_enumerations"]
                assert n_enumerations > 0
                print(f"CaF2 has {n_enumerations} enumerations")

    def test_factory_functions(self):
        """Test factory function creation."""
        # Test default factory
        preprocessor = create_augmented_fingerprint_preprocessor()
        assert preprocessor.config.symprec == 0.01
        assert preprocessor.config.angle_tolerance == 5.0
        
        # Test high precision factory
        high_prec = create_high_precision_fingerprint_preprocessor()
        assert high_prec.config.symprec == 0.001
        assert high_prec.config.angle_tolerance == 1.0
        
        # Test robust factory
        robust = create_robust_fingerprint_preprocessor()
        assert robust.config.symprec == 0.1
        assert robust.config.angle_tolerance == 10.0
        assert robust.config.include_fallback_properties is True

    def test_parallel_processing(self, pymatgen_test_structures):
        """Test parallel processing with multiple jobs."""
        preprocessor = AugmentedFingerprintPreprocessor(n_jobs=2)
        
        result = preprocessor(pymatgen_test_structures)
        
        # Should process successfully with parallel jobs
        assert len(result.processed_structures) <= len(pymatgen_test_structures)
        
        # All structures should have fingerprint properties
        for structure in result.processed_structures:
            assert "augmented_fingerprint" in structure.properties

    def test_error_handling(self):
        """Test error handling with problematic structures."""
        preprocessor = AugmentedFingerprintPreprocessor(include_fallback_properties=True)
        
        # Create a minimal/problematic structure
        minimal_lattice = Lattice.cubic(1.0)
        minimal_structure = Structure(
            minimal_lattice, 
            ["H"], 
            [[0, 0, 0]], 
            coords_are_cartesian=False
        )
        
        processed = preprocessor.process_structure(minimal_structure)
        
        # Should handle errors gracefully
        assert "augmented_fingerprint" in processed.properties
        assert "augmented_fingerprint_properties" in processed.properties
        
        fingerprint_props = processed.properties["augmented_fingerprint_properties"]
        assert "augmented_fingerprint_success" in fingerprint_props
        
        # If fingerprinting failed, should have error information
        if not fingerprint_props["augmented_fingerprint_success"]:
            # Should have fallback properties or error information
            assert (any(key.startswith("augmented_fingerprint_fallback") 
                       for key in fingerprint_props.keys()) or
                    "augmented_fingerprint_error" in fingerprint_props)

    def test_config_to_dict(self):
        """Test configuration serialization."""
        preprocessor = AugmentedFingerprintPreprocessor(
            symprec=0.05,
            angle_tolerance=7.5,
            include_fallback_properties=False
        )
        
        config_dict = preprocessor.config.to_dict()
        
        # Check that all expected keys are present
        expected_keys = [
            "name", "description", "n_jobs", "symprec", 
            "angle_tolerance", "include_fallback_properties"
        ]
        for key in expected_keys:
            assert key in config_dict
        
        # Check values
        assert config_dict["symprec"] == 0.05
        assert config_dict["angle_tolerance"] == 7.5
        assert config_dict["include_fallback_properties"] is False

    def test_get_process_attributes(self):
        """Test that process attributes are correctly extracted."""
        preprocessor = AugmentedFingerprintPreprocessor(
            symprec=0.02,
            angle_tolerance=3.0,
            include_fallback_properties=False
        )
        
        attrs = preprocessor._get_process_attributes()
        
        assert attrs["symprec"] == 0.02
        assert attrs["angle_tolerance"] == 3.0
        assert attrs["include_fallback_properties"] is False

    def test_empty_structure_list(self):
        """Test handling of empty structure list."""
        preprocessor = AugmentedFingerprintPreprocessor()
        
        result = preprocessor([])
        
        assert len(result.processed_structures) == 0
        assert result.n_input_structures == 0
        assert len(result.failed_indices) == 0
        assert result.computation_time >= 0

    def test_single_structure_in_list(self, test_structures):
        """Test processing a single structure in a list."""
        preprocessor = AugmentedFingerprintPreprocessor()
        
        nacl = test_structures["nacl"]
        result = preprocessor([nacl])
        
        assert len(result.processed_structures) <= 1
        assert result.n_input_structures == 1
        
        if result.processed_structures:
            structure = result.processed_structures[0]
            assert "augmented_fingerprint" in structure.properties


def test_module_imports():
    """Test that all required imports work correctly."""
    # Test that we can import all the factory functions
    from lemat_genbench.preprocess.augmented_fingerprint_preprocess import (
        AugmentedFingerprintPreprocessor,
        create_augmented_fingerprint_preprocessor,
        create_high_precision_fingerprint_preprocessor,
        create_robust_fingerprint_preprocessor,
    )
    
    # Test that we can create instances
    assert AugmentedFingerprintPreprocessor is not None
    assert create_augmented_fingerprint_preprocessor is not None
    assert create_high_precision_fingerprint_preprocessor is not None
    assert create_robust_fingerprint_preprocessor is not None


def test_integration_with_base_preprocessor():
    """Test that the preprocessor properly inherits from BasePreprocessor."""
    from lemat_genbench.preprocess.base import BasePreprocessor
    
    preprocessor = AugmentedFingerprintPreprocessor()
    
    # Should be an instance of BasePreprocessor
    assert isinstance(preprocessor, BasePreprocessor)
    
    # Should have all required attributes
    assert hasattr(preprocessor, "config")
    assert hasattr(preprocessor, "name")
    assert hasattr(preprocessor, "description")
    
    # Should have callable interface
    assert callable(preprocessor)


if __name__ == "__main__":
    # Quick test when run directly
    try:
        from pymatgen.util.testing import MatSciTest as PymatgenTest
    except ImportError:
        from pymatgen.util.testing import PymatgenTest
    
    test = PymatgenTest()
    structures = [test.get_structure("Si"), test.get_structure("LiFePO4")]
    
    preprocessor = AugmentedFingerprintPreprocessor()
    result = preprocessor(structures)
    
    print(f"Processed {len(result.processed_structures)} structures")
    print(f"Failed indices: {result.failed_indices}")
    print(f"Computation time: {result.computation_time:.2f}s")
    
    for i, structure in enumerate(result.processed_structures):
        fingerprint = structure.properties.get("augmented_fingerprint")
        print(f"Structure {i+1} ({structure.formula}): {fingerprint[:100] if fingerprint else 'None'}...")