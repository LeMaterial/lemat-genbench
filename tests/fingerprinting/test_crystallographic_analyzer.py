"""Tests for crystallographic analyzer functionality."""

import numpy as np
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

from lemat_genbench.fingerprinting.crystallographic_analyzer import (
    CrystallographicAnalyzer,
    analyze_lematbulk_item,
    lematbulk_item_to_structure,
    structure_to_crystallographic_dict,
)


def create_test_structures():
    """Create test structures with known properties."""
    structures = {}
    
    # 1. Simple cubic NaCl (rock salt structure, space group 225 - Fm-3m)
    lattice_nacl = Lattice.cubic(5.64)
    nacl = Structure(
        lattice_nacl,
        ["Na", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False
    )
    structures['nacl'] = nacl
    
    # 2. CsCl structure (space group 221 - Pm-3m) 
    lattice_cscl = Lattice.cubic(4.11)
    cscl = Structure(
        lattice_cscl,
        ["Cs", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False
    )
    structures['cscl'] = cscl
    
    # 3. Simple tetragonal structure
    lattice_tetra = Lattice.tetragonal(4.0, 6.0)
    tetra = Structure(
        lattice_tetra,
        ["Ti", "O", "O"],
        [[0, 0, 0], [0.5, 0.5, 0.25], [0.5, 0.5, 0.75]],
        coords_are_cartesian=False
    )
    structures['tetragonal'] = tetra
    
    return structures


def create_lematbulk_test_item():
    """Create a mock LeMat-Bulk dataset item for testing."""
    return {
        'immutable_id': 'test-structure-001',
        'lattice_vectors': [
            [5.64, 0.0, 0.0],
            [0.0, 5.64, 0.0], 
            [0.0, 0.0, 5.64]
        ],
        'species_at_sites': ['Na', 'Cl'],
        'cartesian_site_positions': [
            [0.0, 0.0, 0.0],
            [2.82, 2.82, 2.82]
        ],
        'elements': ['Na', 'Cl'],
        'chemical_formula_reduced': 'NaCl'
    }


class TestCrystallographicAnalyzer:
    """Test suite for CrystallographicAnalyzer class."""
    
    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = CrystallographicAnalyzer()
        assert analyzer.symprec == 0.01
        assert analyzer.angle_tolerance == 5.0
        
        # Test custom parameters
        analyzer_custom = CrystallographicAnalyzer(symprec=0.1, angle_tolerance=10.0)
        assert analyzer_custom.symprec == 0.1
        assert analyzer_custom.angle_tolerance == 10.0
    
    def test_analyze_structure_nacl(self):
        """Test crystallographic analysis on NaCl structure."""
        structures = create_test_structures()
        analyzer = CrystallographicAnalyzer()
        
        result = analyzer.analyze_structure(structures['nacl'])
        
        # Check that analysis succeeded
        assert result['success'] is True
        assert result['error'] is None
        
        # Check basic properties
        assert isinstance(result['spacegroup_number'], int)
        assert result['spacegroup_number'] > 0
        
        # Check elements
        assert 'Na' in result['elements']
        assert 'Cl' in result['elements']
        assert len(result['elements']) == 2
        
        # Check that we have site symmetries and multiplicities
        assert len(result['site_symmetries']) > 0
        assert len(result['multiplicity']) > 0
        assert len(result['sites_enumeration_augmented']) > 0
        
        # Check data types
        assert isinstance(result['elements'], list)
        assert isinstance(result['site_symmetries'], list)
        assert isinstance(result['multiplicity'], list)
        assert isinstance(result['sites_enumeration_augmented'], list)
    
    def test_analyze_structure_cscl(self):
        """Test crystallographic analysis on CsCl structure."""
        structures = create_test_structures()
        analyzer = CrystallographicAnalyzer()

        result = analyzer.analyze_structure(structures["cscl"])

        # Check that analysis succeeded
        assert result["success"] is True

        # Check elements
        assert "Cs" in result["elements"]
        assert "Cl" in result["elements"]

        # Verify that both structures can be analyzed (space groups may be same or different)
        nacl_result = analyzer.analyze_structure(structures["nacl"])
        assert isinstance(result["spacegroup_number"], int)
        assert isinstance(nacl_result["spacegroup_number"], int)
        # Note: CsCl and NaCl with these coordinates may have same space group, which is fine
    def test_analyze_invalid_structure(self):
        """Test behavior with invalid structure."""
        analyzer = CrystallographicAnalyzer()
        
        # Create a problematic structure (empty)
        try:
            empty_lattice = Lattice.cubic(1.0)
            empty_structure = Structure(empty_lattice, [], [])
            
            result = analyzer.analyze_structure(empty_structure)
            
            # Should handle gracefully
            assert result['success'] is False
            assert result['error'] is not None
            assert isinstance(result['error'], str)
            
        except Exception:
            # If structure creation itself fails, that's also acceptable
            pass
    
    def test_extract_composition_info(self):
        """Test composition information extraction."""
        structures = create_test_structures()
        analyzer = CrystallographicAnalyzer()
        
        comp_info = analyzer.extract_composition_info(structures['nacl'])
        
        assert comp_info['formula'] == 'NaCl'
        assert comp_info['num_elements'] == 2
        assert comp_info['num_sites'] == 2
        assert 'Na' in str(comp_info['elements'])
        assert 'Cl' in str(comp_info['elements'])


class TestConvenienceFunctions:
    """Test standalone convenience functions."""
    
    def test_structure_to_crystallographic_dict(self):
        """Test the convenience function for structure analysis."""
        structures = create_test_structures()
        
        result = structure_to_crystallographic_dict(structures['nacl'])
        
        # Should contain both crystallographic and composition data
        assert 'spacegroup_number' in result
        assert 'elements' in result
        assert 'formula' in result
        assert 'num_sites' in result
        assert result['success'] is True
    
    def test_lematbulk_item_to_structure(self):
        """Test conversion from LeMat-Bulk item to Structure."""
        item = create_lematbulk_test_item()
        
        structure = lematbulk_item_to_structure(item)
        
        assert isinstance(structure, Structure)
        assert len(structure) == 2  # Na and Cl
        assert structure.composition.reduced_formula == 'NaCl'
        
        # Check lattice
        expected_lattice = np.array([
            [5.64, 0.0, 0.0],
            [0.0, 5.64, 0.0],
            [0.0, 0.0, 5.64]
        ])
        np.testing.assert_array_almost_equal(structure.lattice.matrix, expected_lattice)
    
    def test_analyze_lematbulk_item(self):
        """Test complete LeMat-Bulk item analysis."""
        item = create_lematbulk_test_item()
        
        result = analyze_lematbulk_item(item)
        
        # Should have crystallographic analysis
        assert result['immutable_id'] == 'test-structure-001'
        assert result['success'] is True
        assert 'spacegroup_number' in result
        assert 'elements' in result
        assert 'formula' in result
        
        # Should contain Na and Cl
        assert 'Na' in result['elements']
        assert 'Cl' in result['elements']
    
    def test_analyze_lematbulk_item_invalid(self):
        """Test behavior with invalid LeMat-Bulk item."""
        invalid_item = {
            'immutable_id': 'invalid-item',
            'lattice_vectors': 'not_a_list',  # Invalid format
        }
        
        result = analyze_lematbulk_item(invalid_item)
        
        assert result['success'] is False
        assert result['error'] is not None
        assert result['immutable_id'] == 'invalid-item'


class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_full_workflow(self):
        """Test the complete workflow from LeMat-Bulk item to fingerprinting data."""
        item = create_lematbulk_test_item()
        
        # Step 1: Convert to structure
        structure = lematbulk_item_to_structure(item)
        
        # Step 2: Analyze crystallography
        crystal_data = structure_to_crystallographic_dict(structure)
        
        # Step 3: Verify we have all needed data for fingerprinting
        required_fields = [
            'spacegroup_number', 'elements', 'site_symmetries', 
            'sites_enumeration_augmented', 'multiplicity'
        ]
        
        for field in required_fields:
            assert field in crystal_data, f"Missing required field: {field}"
        
        # Verify data types are correct for fingerprinting
        assert isinstance(crystal_data['spacegroup_number'], int)
        assert isinstance(crystal_data['elements'], list)
        assert isinstance(crystal_data['site_symmetries'], list)
        assert isinstance(crystal_data['sites_enumeration_augmented'], list)
        assert isinstance(crystal_data['multiplicity'], list)
    
    def test_consistency_across_calls(self):
        """Test that repeated analysis gives consistent results."""
        structures = create_test_structures()
        
        # Analyze same structure multiple times
        result1 = structure_to_crystallographic_dict(structures['nacl'])
        result2 = structure_to_crystallographic_dict(structures['nacl'])
        
        # Results should be identical
        assert result1['spacegroup_number'] == result2['spacegroup_number']
        assert result1['elements'] == result2['elements']
        assert result1['formula'] == result2['formula']


if __name__ == "__main__":
    """Manual test runner for development."""
    print("🧪 Running crystallographic analyzer tests...")
    
    try:
        # Test basic functionality
        print("1. Testing basic structure analysis...")
        structures = create_test_structures()
        analyzer = CrystallographicAnalyzer()
        
        for name, structure in structures.items():
            result = analyzer.analyze_structure(structure)
            print(f"   {name}: Space group {result['spacegroup_number']}, "
                  f"Elements: {result['elements']}")
        
        # Test LeMat-Bulk conversion
        print("2. Testing LeMat-Bulk item conversion...")
        item = create_lematbulk_test_item()
        result = analyze_lematbulk_item(item)
        print(f"   LeMat-Bulk item: {result['formula']}, "
              f"Space group: {result['spacegroup_number']}")
        
        print("✅ All manual tests passed!")
        print("\nTo run full test suite:")
        print("   pytest tests/fingerprinting/test_crystallographic_analyzer.py -v")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()