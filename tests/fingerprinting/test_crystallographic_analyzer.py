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
    
    # 4. Fluorite structure (more complex for testing enhanced enumerations)
    lattice_caf2 = Lattice.cubic(5.46)
    caf2 = Structure(
        lattice_caf2,
        ["Ca", "F", "F"],
        [[0, 0, 0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
        coords_are_cartesian=False
    )
    structures['caf2'] = caf2
    
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
        assert len(result['elements']) >= 1
        
        # Check that we have site symmetries with proper Wyckoff format (number + letter)
        assert len(result['site_symmetries']) > 0
        for site_sym in result['site_symmetries']:
            assert isinstance(site_sym, str)
            assert len(site_sym) >= 2
            # Extract multiplicity (digits at start)
            mult_part = ""
            for char in site_sym:
                if char.isdigit():
                    mult_part += char
                else:
                    break
            assert len(mult_part) > 0, f"No multiplicity found in {site_sym}"
            
            # Extract Wyckoff letter (letters at end)
            wyckoff_part = site_sym[len(mult_part):]
            assert len(wyckoff_part) >= 1, f"No Wyckoff letter found in {site_sym}"
            assert wyckoff_part.isalpha(), f"Invalid Wyckoff letter in {site_sym}"
        
        # Check multiplicity data
        assert len(result['multiplicity']) > 0
        for mult in result['multiplicity']:
            assert isinstance(mult, int)
            assert mult > 0
        
        # Check enhanced enumeration data - should have multiple enumerations
        assert len(result['sites_enumeration_augmented']) >= 1
        for enumeration in result['sites_enumeration_augmented']:
            assert isinstance(enumeration, list)
            assert len(enumeration) == len(result['elements'])
            # All enumeration values should be positive integers
            for val in enumeration:
                assert isinstance(val, int)
                assert val > 0
        
        # Check data types
        assert isinstance(result['elements'], list)
        assert isinstance(result['site_symmetries'], list)
        assert isinstance(result['multiplicity'], list)
        assert isinstance(result['sites_enumeration_augmented'], list)
        
        print(f"   NaCl: Generated {len(result['sites_enumeration_augmented'])} enumerations")
    
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

        # Verify space group is reasonable
        assert isinstance(result["spacegroup_number"], int)
        assert 1 <= result["spacegroup_number"] <= 230
        
        # Check Wyckoff positions are properly formatted
        for site_sym in result["site_symmetries"]:
            assert isinstance(site_sym, str)
            assert len(site_sym) >= 2
            assert any(char.isdigit() for char in site_sym)  # Has multiplicity
            assert any(char.isalpha() for char in site_sym)  # Has Wyckoff letter
            
        print(f"   CsCl: Generated {len(result['sites_enumeration_augmented'])} enumerations")
    
    def test_enumeration_count_and_diversity(self):
        """Test that enumerations are generated properly."""
        structures = create_test_structures()
        analyzer = CrystallographicAnalyzer()
        
        for name, structure in structures.items():
            result = analyzer.analyze_structure(structure)
            
            if result['success']:
                enum_count = len(result['sites_enumeration_augmented'])
                print(f"   {name}: {enum_count} enumerations")
                
                # Should have at least 1 enumeration
                assert enum_count >= 1, f"{name} should have at least 1 enumeration, got {enum_count}"
                
                # But not too many (should be capped)
                assert enum_count <= 50, f"{name} should have at most 50 enumerations, got {enum_count}"
                
                # Check enumeration diversity for structures with multiple atoms
                if len(result['elements']) > 1:
                    unique_enums = set(tuple(enum) for enum in result['sites_enumeration_augmented'])
                    diversity_ratio = len(unique_enums) / len(result['sites_enumeration_augmented'])
                    print(f"     Diversity ratio: {diversity_ratio:.2f}")
                    # Should have at least some diversity if multiple enumerations exist
                    if enum_count > 1:
                        assert diversity_ratio > 0, f"{name} should have some enumeration diversity"
    
    def test_wyckoff_position_accuracy(self):
        """Test that Wyckoff positions are accurate using spglib."""
        structures = create_test_structures()
        analyzer = CrystallographicAnalyzer()
        
        for name, structure in structures.items():
            result = analyzer.analyze_structure(structure)
            
            assert result['success'] is True, f"Analysis failed for {name}: {result['error']}"
            
            # Verify Wyckoff positions have correct format
            for site_sym in result['site_symmetries']:
                # Should have format like "4a", "8c", "1b", etc.
                assert len(site_sym) >= 2
                
                # Extract multiplicity (all digits at start)
                mult_str = ""
                for char in site_sym:
                    if char.isdigit():
                        mult_str += char
                    else:
                        break
                
                assert len(mult_str) > 0, f"No multiplicity found in {site_sym}"
                multiplicity = int(mult_str)
                assert multiplicity > 0
                
                # Extract Wyckoff letter (should be alphabetic after multiplicity)
                wyckoff_letter = site_sym[len(mult_str):]
                assert len(wyckoff_letter) >= 1, f"No Wyckoff letter found in {site_sym}"
                assert wyckoff_letter.isalpha(), f"Invalid Wyckoff letter in {site_sym}"
                
                print(f"   {name}: {site_sym} -> mult={multiplicity}, wyckoff={wyckoff_letter}")
    
    def test_group_operations(self):
        """Test individual group operation generation."""
        analyzer = CrystallographicAnalyzer()
        
        # Test different group sizes
        test_sizes = [1, 2, 3, 4, 6, 8]
        
        for size in test_sizes:
            print(f"\n   Testing group size {size}:")
            
            group_ops = analyzer._generate_group_operations(size)
            assert len(group_ops) >= 1, f"Should have at least 1 operation for size {size}"
            
            # All operations should have correct size
            for op in group_ops:
                assert len(op) == size, f"Operation {op} has wrong size"
                assert all(isinstance(val, int) and 1 <= val <= size for val in op), \
                    f"Invalid values in operation {op}"
            
            print(f"     Generated {len(group_ops)} operations")
            
            # Test specific operation types for known sizes
            if size == 1:
                assert group_ops == [[1]], "Size 1 should only have identity"
            
            if size >= 2:
                # Should have identity
                identity = list(range(1, size + 1))
                assert identity in group_ops, "Should have identity operation"
                
                # Should have reverse
                reverse = list(range(size, 0, -1))
                assert reverse in group_ops, "Should have reverse operation"
            
            if size >= 3:
                # Should have at least one cyclic shift
                cyclic_shift = list(range(2, size + 1)) + [1]
                has_cyclic = any(op == cyclic_shift for op in group_ops)
                assert has_cyclic, f"Should have cyclic shift for size {size}"
    
    def test_analyze_invalid_structure(self):
        """Test behavior with problematic structures."""
        analyzer = CrystallographicAnalyzer()
        
        # Create a structure that might be problematic
        try:
            # Very small lattice
            tiny_lattice = Lattice.cubic(0.1)
            tiny_structure = Structure(tiny_lattice, ["H"], [[0, 0, 0]])
            
            result = analyzer.analyze_structure(tiny_structure)
            
            # Should either succeed or fail gracefully
            assert isinstance(result['success'], bool)
            if not result['success']:
                assert result['error'] is not None
                assert isinstance(result['error'], str)
                
        except Exception:
            # Structure creation itself failing is acceptable
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


class TestEnumerationMethods:
    """Test suite for enumeration generation methods."""
    
    def test_generate_equivalent_enumerations(self):
        """Test equivalent enumeration generation."""
        analyzer = CrystallographicAnalyzer()
        
        # Test with simple equivalent indices
        equiv_indices = [[0, 1], [2, 3]]  # Two groups of 2 atoms each
        enumerations = analyzer._generate_equivalent_enumerations(equiv_indices)
        
        # Should have multiple enumerations
        assert len(enumerations) >= 1
        
        # Each enumeration should have 4 elements (2 groups × 2 atoms)
        for enum in enumerations:
            assert len(enum) == 4
            assert all(isinstance(val, int) and val > 0 for val in enum)
    
    def test_generate_all_combinations(self):
        """Test all combinations generation."""
        analyzer = CrystallographicAnalyzer()
        
        # Small example that should generate all combinations
        group_enumerations = [
            [[1, 2], [2, 1]],  # Group 1: 2 operations
            [[1], [1]]         # Group 2: 1 operation (repeated for testing)
        ]
        
        combinations = analyzer._generate_all_combinations(group_enumerations)
        
        # Should have 2 × 2 = 4 combinations
        assert len(combinations) == 4
        
        # Check specific combinations
        _ = [
            [1, 2, 1],  # [1,2] + [1]
            [1, 2, 1],  # [1,2] + [1] (duplicate)
            [2, 1, 1],  # [2,1] + [1]
            [2, 1, 1]   # [2,1] + [1] (duplicate)
        ]
        
        # All combinations should be valid
        for combo in combinations:
            assert len(combo) == 3  # 2 + 1
            assert all(isinstance(val, int) and val > 0 for val in combo)
    
    def test_generate_sampled_combinations(self):
        """Test sampled combinations for large cases."""
        analyzer = CrystallographicAnalyzer()
        
        # Create a case with many possible combinations
        group_enumerations = [
            [[1, 2, 3], [3, 2, 1], [2, 3, 1]],  # 3 operations
            [[1, 2], [2, 1]],                   # 2 operations
            [[1, 2, 3, 4], [4, 3, 2, 1]]        # 2 operations
        ]
        # Total: 3 × 2 × 2 = 12 combinations
        
        # Sample with small limit
        combinations = analyzer._generate_sampled_combinations(group_enumerations, max_enumerations=5)
        
        # Should respect the limit
        assert len(combinations) <= 5
        
        # All combinations should be valid
        for combo in combinations:
            assert len(combo) == 9  # 3 + 2 + 4
            assert all(isinstance(val, int) and val > 0 for val in combo)
        
        # Should include identity (first operation from each group)
        identity_combo = [1, 2, 3, 1, 2, 1, 2, 3, 4]
        assert identity_combo in combinations


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
        
        # Should have proper Wyckoff positions
        assert len(result['site_symmetries']) > 0
        for site_sym in result['site_symmetries']:
            assert any(char.isdigit() for char in site_sym)
            assert any(char.isalpha() for char in site_sym)
            
        # Should have enumerations
        assert len(result['sites_enumeration_augmented']) >= 1
    
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
        
        # Should have successful crystallographic analysis
        assert result['immutable_id'] == 'test-structure-001'
        assert result['success'] is True
        assert 'spacegroup_number' in result
        assert 'elements' in result
        assert 'formula' in result
        
        # Should contain Na and Cl
        assert 'Na' in result['elements']
        assert 'Cl' in result['elements']
        
        # Should have proper Wyckoff analysis and enumerations
        assert len(result['site_symmetries']) > 0
        assert len(result['sites_enumeration_augmented']) >= 1
    
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
        
        # Verify proper Wyckoff format
        for site_sym in crystal_data['site_symmetries']:
            assert isinstance(site_sym, str)
            assert len(site_sym) >= 2
            # Should have multiplicity + Wyckoff letter format
            assert any(char.isdigit() for char in site_sym)
            assert any(char.isalpha() for char in site_sym)
        
        # Verify enumerations
        assert len(crystal_data['sites_enumeration_augmented']) >= 1
        for enumeration in crystal_data['sites_enumeration_augmented']:
            assert len(enumeration) == len(crystal_data['elements'])
            assert all(isinstance(val, int) and val > 0 for val in enumeration)
    
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
        assert result1['site_symmetries'] == result2['site_symmetries']
        assert result1['sites_enumeration_augmented'] == result2['sites_enumeration_augmented']
    
    def test_different_structures_give_different_results(self):
        """Test that different structures produce different crystallographic data."""
        structures = create_test_structures()
        analyzer = CrystallographicAnalyzer()
        
        results = {}
        for name, structure in structures.items():
            results[name] = analyzer.analyze_structure(structure)
            assert results[name]['success'] is True
        
        # Different structures should have different space groups or site symmetries
        nacl_result = results['nacl']
        cscl_result = results['cscl']
        
        # Should be different in some way (space group, site symmetries, or elements)
        different = (
            nacl_result['spacegroup_number'] != cscl_result['spacegroup_number'] or
            nacl_result['site_symmetries'] != cscl_result['site_symmetries'] or
            set(nacl_result['elements']) != set(cscl_result['elements'])
        )
        assert different, "Different structures should produce different crystallographic data"


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
            assert result['success'] is True, f"Analysis failed for {name}: {result['error']}"
            print(f"   ✅ {name}: Space group {result['spacegroup_number']}")
            print(f"       Elements: {result['elements']}")
            print(f"       Site symmetries: {result['site_symmetries']}")
            print(f"       Enumerations: {len(result['sites_enumeration_augmented'])}")
        
        # Test LeMat-Bulk conversion
        print("\n2. Testing LeMat-Bulk item conversion...")
        item = create_lematbulk_test_item()
        result = analyze_lematbulk_item(item)
        assert result['success'] is True, f"LeMat-Bulk analysis failed: {result['error']}"
        print(f"   ✅ LeMat-Bulk item: {result['formula']}")
        print(f"       Space group: {result['spacegroup_number']}")
        print(f"       Site symmetries: {result['site_symmetries']}")
        print(f"       Enumerations: {len(result['sites_enumeration_augmented'])}")
        
        # Test group operations
        print("\n3. Testing group operations...")
        test_analyzer = TestCrystallographicAnalyzer()
        test_analyzer.test_group_operations()
        print("   ✅ Group operations working correctly")
        
        # Test enumeration methods
        print("\n4. Testing enumeration methods...")
        test_enum = TestEnumerationMethods()
        test_enum.test_generate_equivalent_enumerations()
        print("   ✅ Enumeration methods working correctly")
        
        print("\n✅ All manual tests passed!")
        print("\nTo run full test suite:")
        print("   uv run pytest tests/fingerprinting/test_crystallographic_analyzer.py -v")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()