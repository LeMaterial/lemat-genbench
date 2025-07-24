#!/usr/bin/env python3
"""
Quick test to verify the MACE monkey patch fix works.
Run this script to test if the MACE calculator can be initialized successfully.
"""

import sys
import traceback
from pymatgen.util.testing import PymatgenTest


def test_mace_import():
    """Test if MACE can be imported without issues."""
    print("🔧 Testing MACE import...")
    try:
        from lematerial_forgebench.models.mace.calculator import MACECalculator, MACE_AVAILABLE
        print(f"✅ MACE import successful. Available: {MACE_AVAILABLE}")
        return True
    except ImportError as e:
        print(f"❌ MACE import failed: {e}")
        return False


def test_mace_calculator_creation():
    """Test if MACE calculator can be created (this is where the monkey patch helps)."""
    print("\n🔧 Testing MACE calculator creation...")
    try:
        from lematerial_forgebench.models.mace.calculator import MACECalculator
        
        # Try to create calculator - this is where the original bug would occur
        calculator = MACECalculator(model_type="mp", device="cpu")
        print("✅ MACE calculator created successfully!")
        return True
        
    except Exception as e:
        print(f"❌ MACE calculator creation failed: {e}")
        print("📋 Full traceback:")
        traceback.print_exc()
        return False


def test_mace_calculation():
    """Test if MACE can perform a simple calculation."""
    print("\n🔧 Testing MACE calculation...")
    try:
        from lematerial_forgebench.models.mace.calculator import MACECalculator
        
        # Create test structure
        test = PymatgenTest()
        structure = test.get_structure("Si")  # Simple silicon structure
        
        # Create calculator
        calculator = MACECalculator(model_type="mp", device="cpu")
        
        # Try calculation
        result = calculator.calculate_energy_forces(structure)
        
        print("✅ MACE calculation successful!")
        print(f"   Energy: {result.energy:.4f} eV")
        print(f"   Forces shape: {result.forces.shape}")
        print(f"   Model type: {result.metadata.get('model_type', 'unknown')}")
        return True
        
    except Exception as e:
        print(f"❌ MACE calculation failed: {e}")
        print("📋 Full traceback:")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🧪 Quick MACE Monkey Patch Test")
    print("=" * 50)
    
    tests = [
        test_mace_import,
        test_mace_calculator_creation,
        test_mace_calculation,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    
    test_names = [
        "MACE Import",
        "Calculator Creation", 
        "MACE Calculation"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {i+1}. {name}: {status}")
    
    total_passed = sum(results)
    print(f"\n🎯 Overall: {total_passed}/{len(results)} tests passed")
    
    if all(results):
        print("🎉 All tests passed! Your monkey patch fix is working!")
        return 0
    else:
        print("⚠️  Some tests failed. The monkey patch may need adjustment.")
        return 1


if __name__ == "__main__":
    sys.exit(main())