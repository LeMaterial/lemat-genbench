#!/usr/bin/env python3
"""
Comprehensive test script for MACE, ORB, and UMA models.
This is the definitive test that checks everything: availability, energies, forces, 
embeddings, benchmarks, and provides detailed diagnostics.
"""

import traceback
import time
import numpy as np
from pathlib import Path
from pymatgen.util.testing import PymatgenTest


class ComprehensiveModelTester:
    """Complete testing suite for all models."""
    
    def __init__(self):
        self.results = {}
        self.test_structures = []
        self.setup_test_structures()
    
    def setup_test_structures(self):
        """Create test structures for evaluation."""
        try:
            test = PymatgenTest()
            self.test_structures = [
                test.get_structure("Si"),
                test.get_structure("LiFePO4"),
            ]
            
            # Add CIF structures if available
            cif_files = ["CsBr.cif", "CsPbBr3.cif", "NiO.cif"]
            for cif_file in cif_files:
                if Path(cif_file).exists():
                    from pymatgen.core import Structure
                    structure = Structure.from_file(cif_file)
                    structure = structure.remove_oxidation_states()
                    self.test_structures.append(structure)
            
            print(f"✅ Setup {len(self.test_structures)} test structures")
            
        except Exception as e:
            print(f"⚠️ Error setting up structures: {e}")
            # Fallback: create simple structures manually
            from pymatgen.core import Structure, Lattice
            si_structure = Structure(
                Lattice.cubic(5.43),
                ["Si", "Si"],
                [[0, 0, 0], [0.25, 0.25, 0.25]]
            )
            self.test_structures = [si_structure]
    
    def print_header(self, title):
        """Print formatted section header."""
        print(f"\n{'='*60}")
        print(f"🧪 {title}")
        print(f"{'='*60}")
    
    def print_subheader(self, title):
        """Print formatted subsection header."""
        print(f"\n{'-'*40}")
        print(f"🔍 {title}")
        print(f"{'-'*40}")
    
    def test_model_availability(self):
        """Test which models are detected and available."""
        self.print_header("MODEL AVAILABILITY")
        
        try:
            from lematerial_forgebench.models.registry import (
                list_available_models,
                get_model_info,
                print_model_info
            )
            
            available_models = list_available_models()
            print(f"📋 Available models: {available_models}")
            
            if available_models:
                print("\n📊 Detailed Model Information:")
                print_model_info()
                
                model_info = get_model_info()
                self.results['availability'] = {
                    'available_models': available_models,
                    'model_info': model_info
                }
            else:
                print("❌ No models detected!")
                self.results['availability'] = {'available_models': [], 'model_info': {}}
                
        except Exception as e:
            print(f"❌ Error checking availability: {e}")
            traceback.print_exc()
            self.results['availability'] = {'error': str(e)}
    
    def test_orb_direct(self):
        """Test ORB model directly."""
        self.print_subheader("ORB Direct Testing")
        
        orb_results = {'available': False, 'calculator_created': False, 'energy_calculated': False, 'embeddings_extracted': False}
        
        try:
            # Test import
            from orb_models.forcefield import pretrained
            from orb_models.forcefield.calculator import ORBCalculator as ORBASECalculator
            print("✅ ORB imports successful")
            orb_results['available'] = True
            
            # Test model creation
            model_func = getattr(pretrained, "orb_v3_conservative_inf_omat")
            model = model_func(device="cpu", precision="float32-high", compile=False)
            ase_calc = ORBASECalculator(model, device="cpu")
            print("✅ ORB model and calculator created")
            orb_results['calculator_created'] = True
            
            # Test energy calculation
            structure = self.test_structures[0]
            from lematerial_forgebench.models.base import BaseMLIPCalculator
            atoms = BaseMLIPCalculator._structure_to_atoms(None, structure)
            atoms.calc = ase_calc
            
            start_time = time.time()
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            calc_time = time.time() - start_time
            
            print(f"✅ ORB energy: {energy:.3f} eV (calculated in {calc_time:.3f}s)")
            print(f"   Forces shape: {forces.shape}, norm: {np.linalg.norm(forces):.3f}")
            orb_results['energy_calculated'] = True
            orb_results['energy'] = energy
            orb_results['forces_norm'] = np.linalg.norm(forces)
            orb_results['calculation_time'] = calc_time
            
            # Test through ForgeBench calculator
            from lematerial_forgebench.models.orb.calculator import ORBCalculator
            forgebench_calc = ORBCalculator(device="cpu")
            result = forgebench_calc.calculate_energy_forces(structure)
            print(f"✅ ORB through ForgeBench: {result.energy:.3f} eV")
            
            # Test embeddings
            try:
                embedding_result = forgebench_calc.extract_embeddings(structure)
                print("✅ ORB embeddings extracted")
                orb_results['embeddings_extracted'] = True
                if hasattr(embedding_result, 'embeddings') and embedding_result.embeddings is not None:
                    if hasattr(embedding_result.embeddings, 'shape'):
                        print(f"   Embeddings shape: {embedding_result.embeddings.shape}")
                    else:
                        print(f"   Embeddings length: {len(embedding_result.embeddings)}")
            except Exception as e:
                print(f"⚠️ ORB embeddings failed: {e}")
            
        except Exception as e:
            print(f"❌ ORB direct test failed: {e}")
            orb_results['error'] = str(e)
        
        self.results['orb_direct'] = orb_results
    
    def test_uma_direct(self):
        """Test UMA model directly."""
        self.print_subheader("UMA Direct Testing")
        
        uma_results = {'available': False, 'calculator_created': False, 'energy_calculated': False, 'embeddings_extracted': False}
        
        try:
            # Test import
            from fairchem.core import pretrained_mlip
            from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
            print("✅ UMA imports successful")
            uma_results['available'] = True
            
            # Test predictor creation
            predictor = pretrained_mlip.get_predict_unit("uma-s-1", device="cpu")
            ase_calc = FAIRChemCalculator(predict_unit=predictor, task_name="omat")
            print("✅ UMA predictor and calculator created")
            uma_results['calculator_created'] = True
            
            # Test energy calculation
            structure = self.test_structures[0]
            from lematerial_forgebench.models.base import BaseMLIPCalculator
            atoms = BaseMLIPCalculator._structure_to_atoms(None, structure)
            
            # Handle UMA's ASE calculator quirks
            from copy import deepcopy
            info_copy = deepcopy(atoms.info)
            atoms.info = {}
            atoms.calc = ase_calc
            
            start_time = time.time()
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            calc_time = time.time() - start_time
            
            atoms.info = {**info_copy, **atoms.info}
            
            print(f"✅ UMA energy: {energy:.3f} eV (calculated in {calc_time:.3f}s)")
            print(f"   Forces shape: {forces.shape}, norm: {np.linalg.norm(forces):.3f}")
            uma_results['energy_calculated'] = True
            uma_results['energy'] = energy
            uma_results['forces_norm'] = np.linalg.norm(forces)
            uma_results['calculation_time'] = calc_time
            
            # Test through ForgeBench calculator
            from lematerial_forgebench.models.uma.calculator import UMACalculator
            forgebench_calc = UMACalculator(model_name="uma-s-1", task="omat", device="cpu")
            result = forgebench_calc.calculate_energy_forces(structure)
            print(f"✅ UMA through ForgeBench: {result.energy:.3f} eV")
            
            # Test embeddings
            try:
                embedding_result = forgebench_calc.extract_embeddings(structure)
                print("✅ UMA embeddings extracted")
                uma_results['embeddings_extracted'] = True
                
                # Handle different embedding result formats
                if hasattr(embedding_result, 'node_embeddings'):
                    embeddings = embedding_result.node_embeddings
                elif hasattr(embedding_result, 'embeddings'):
                    embeddings = embedding_result.embeddings
                else:
                    embeddings = embedding_result
                
                if embeddings is not None:
                    if hasattr(embeddings, 'shape'):
                        print(f"   Embeddings shape: {embeddings.shape}")
                    else:
                        print(f"   Embeddings type: {type(embeddings)}")
                
            except Exception as e:
                print(f"⚠️ UMA embeddings failed: {e}")
            
        except Exception as e:
            print(f"❌ UMA direct test failed: {e}")
            uma_results['error'] = str(e)
        
        self.results['uma_direct'] = uma_results
    
    def test_mace_direct(self):
        """Test MACE model directly with multiple approaches."""
        self.print_subheader("MACE Direct Testing")
        
        mace_results = {'available': False, 'any_approach_worked': False, 'approaches': {}}
        
        # Test import
        try:
            from mace.calculators import mace_mp, mace_off, MACECalculator
            print("✅ MACE imports successful")
            mace_results['available'] = True
        except Exception as e:
            print(f"❌ MACE import failed: {e}")
            mace_results['error'] = str(e)
            self.results['mace_direct'] = mace_results
            return
        
        # Approach 1: MACE-MP
        print("\n   Approach 1: MACE-MP (Materials Project)")
        try:
            calc = mace_mp(device="cpu")
            structure = self.test_structures[0]
            from lematerial_forgebench.models.base import BaseMLIPCalculator
            atoms = BaseMLIPCalculator._structure_to_atoms(None, structure)
            atoms.calc = calc
            
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            
            print(f"✅ MACE-MP energy: {energy:.3f} eV")
            mace_results['approaches']['mp'] = {
                'success': True, 'energy': energy, 'forces_norm': np.linalg.norm(forces)
            }
            mace_results['any_approach_worked'] = True
            
        except Exception as e:
            print(f"❌ MACE-MP failed: {type(e).__name__}: {e}")
            mace_results['approaches']['mp'] = {'success': False, 'error': str(e)}
        
        # Approach 2: MACE-OFF
        print("\n   Approach 2: MACE-OFF (Off-the-shelf)")
        try:
            calc = mace_off(device="cpu")
            structure = self.test_structures[0]
            from lematerial_forgebench.models.base import BaseMLIPCalculator
            atoms = BaseMLIPCalculator._structure_to_atoms(None, structure)
            atoms.calc = calc
            
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            
            print(f"✅ MACE-OFF energy: {energy:.3f} eV")
            mace_results['approaches']['off'] = {
                'success': True, 'energy': energy, 'forces_norm': np.linalg.norm(forces)
            }
            mace_results['any_approach_worked'] = True
            
        except Exception as e:
            print(f"❌ MACE-OFF failed: {type(e).__name__}: {e}")
            mace_results['approaches']['off'] = {'success': False, 'error': str(e)}
        
        # Approach 3: Clear cache and retry
        if not mace_results['any_approach_worked']:
            print("\n   Approach 3: Clear cache and retry")
            import os
            cache_path = os.path.expanduser("~/.cache/mace/")
            if os.path.exists(cache_path):
                try:
                    import shutil
                    shutil.rmtree(cache_path)
                    print("✅ MACE cache cleared")
                    
                    # Retry MACE-MP after clearing cache
                    calc = mace_mp(device="cpu")
                    structure = self.test_structures[0]
                    from lematerial_forgebench.models.base import BaseMLIPCalculator
                    atoms = BaseMLIPCalculator._structure_to_atoms(None, structure)
                    atoms.calc = calc
                    
                    energy = atoms.get_potential_energy()
                    print(f"✅ MACE-MP (post-clear) energy: {energy:.3f} eV")
                    mace_results['approaches']['mp_post_clear'] = {'success': True, 'energy': energy}
                    mace_results['any_approach_worked'] = True
                    
                except Exception as e:
                    print(f"❌ Post-clear retry failed: {e}")
                    mace_results['approaches']['mp_post_clear'] = {'success': False, 'error': str(e)}
        
        # Test through ForgeBench if any approach worked
        if mace_results['any_approach_worked']:
            try:
                from lematerial_forgebench.models.mace.calculator import MACECalculator
                forgebench_calc = MACECalculator(model_type="mp", device="cpu")
                result = forgebench_calc.calculate_energy_forces(self.test_structures[0])
                print(f"✅ MACE through ForgeBench: {result.energy:.3f} eV")
                mace_results['forgebench_success'] = True
            except Exception as e:
                print(f"⚠️ MACE through ForgeBench failed: {e}")
                mace_results['forgebench_success'] = False
        
        self.results['mace_direct'] = mace_results
    
    def test_registry_integration(self):
        """Test models through the registry system."""
        self.print_subheader("Registry Integration Testing")
        
        registry_results = {}
        
        try:
            from lematerial_forgebench.models.registry import get_calculator, list_available_models
            available_models = list_available_models()
            
            for model_name in available_models:
                print(f"\n   Testing {model_name.upper()} through registry...")
                model_result = {'success': False}
                
                try:
                    # Create calculator with appropriate parameters
                    if model_name == "mace":
                        calc = get_calculator("mace", model_type="mp", device="cpu")
                    elif model_name == "orb":
                        calc = get_calculator("orb", model_type="orb_v3_conservative_inf_omat", device="cpu")
                    elif model_name == "uma":
                        # Try to avoid the parameter conflict
                        calc = get_calculator("uma", task="omat", device="cpu")
                    else:
                        calc = get_calculator(model_name, device="cpu")
                    
                    # Test energy calculation
                    structure = self.test_structures[0]
                    result = calc.calculate_energy_forces(structure)
                    
                    print(f"✅ {model_name.upper()} registry: {result.energy:.3f} eV")
                    model_result['success'] = True
                    model_result['energy'] = result.energy
                    
                    # Test embeddings if supported
                    try:
                        embedding_result = calc.extract_embeddings(structure)
                        print(f"   ✅ Embeddings extracted")
                        model_result['embeddings_success'] = True
                    except Exception as e:
                        print(f"   ⚠️ Embeddings failed: {e}")
                        model_result['embeddings_success'] = False
                    
                except Exception as e:
                    print(f"❌ {model_name.upper()} registry failed: {type(e).__name__}: {e}")
                    model_result['error'] = str(e)
                    
                    # Specific error diagnosis
                    if "multiple values for argument" in str(e):
                        print("   💡 Parameter conflict detected - registry issue")
                        model_result['issue_type'] = 'parameter_conflict'
                    elif "too many values to unpack" in str(e):
                        print("   💡 Model loading issue - likely compatibility problem")
                        model_result['issue_type'] = 'compatibility_issue'
                
                registry_results[model_name] = model_result
        
        except Exception as e:
            print(f"❌ Registry testing failed: {e}")
            registry_results['error'] = str(e)
        
        self.results['registry'] = registry_results
    
    def test_benchmarks(self):
        """Test benchmark integration."""
        self.print_subheader("Benchmark Testing")
        
        benchmark_results = {}
        
        # Test Validity Benchmark (model-independent)
        print("\n   Testing Validity Benchmark...")
        try:
            from lematerial_forgebench.benchmarks.validity_benchmark import ValidityBenchmark
            
            validity_benchmark = ValidityBenchmark()
            result = validity_benchmark.evaluate(self.test_structures[:2])  # Use first 2 structures
            
            validity_score = result.final_scores.get('validity_score', 'N/A')
            print(f"✅ Validity benchmark completed - Score: {validity_score}")
            benchmark_results['validity'] = {
                'success': True, 
                'score': validity_score,
                'all_scores': result.final_scores
            }
            
        except Exception as e:
            print(f"❌ Validity benchmark failed: {e}")
            benchmark_results['validity'] = {'success': False, 'error': str(e)}
        
        # Test Stability Benchmark with working models
        print("\n   Testing Stability Benchmark...")
        working_models = []
        
        # Check which models are working from previous tests
        if self.results.get('orb_direct', {}).get('energy_calculated', False):
            working_models.append('orb')
        if self.results.get('uma_direct', {}).get('energy_calculated', False):
            working_models.append('uma')
        if self.results.get('mace_direct', {}).get('any_approach_worked', False):
            working_models.append('mace')
        
        for model_name in working_models[:]:  # Test with first working model only
            try:
                print(f"     Testing with {model_name.upper()}...")
                
                # Use direct calculator creation to avoid registry issues
                if model_name == 'orb':
                    from lematerial_forgebench.models.orb.calculator import ORBCalculator
                    calc = ORBCalculator(device="cpu")
                elif model_name == 'uma':
                    from lematerial_forgebench.models.uma.calculator import UMACalculator
                    calc = UMACalculator(model_name="uma-s-1", task="omat", device="cpu")
                elif model_name == 'mace':
                    from lematerial_forgebench.models.mace.calculator import MACECalculator
                    calc = MACECalculator(model_type="mp", device="cpu")
                
                # Simple stability test - just calculate formation energies
                structure = self.test_structures[0]
                result = calc.calculate_energy_forces(structure)
                
                # Try to calculate formation energy
                try:
                    formation_energy = calc.calculate_formation_energy(structure)
                    print(f"✅ {model_name.upper()} formation energy: {formation_energy:.3f} eV/atom")
                    
                    benchmark_results[f'stability_{model_name}'] = {
                        'success': True,
                        'formation_energy': formation_energy
                    }
                except Exception as e:
                    print(f"⚠️ Formation energy calculation failed: {e}")
                    benchmark_results[f'stability_{model_name}'] = {
                        'success': False, 
                        'error': str(e)
                    }
                
            except Exception as e:
                print(f"❌ Stability test with {model_name.upper()} failed: {e}")
                benchmark_results[f'stability_{model_name}'] = {'success': False, 'error': str(e)}
        
        self.results['benchmarks'] = benchmark_results
    
    def test_performance(self):
        """Test performance characteristics."""
        self.print_subheader("Performance Testing")
        
        performance_results = {}
        
        # Test each working model's performance
        working_models = []
        if self.results.get('orb_direct', {}).get('energy_calculated', False):
            working_models.append(('orb', lambda: self.create_orb_calc()))
        if self.results.get('uma_direct', {}).get('energy_calculated', False):
            working_models.append(('uma', lambda: self.create_uma_calc()))
        
        structure = self.test_structures[0]  # Use smallest structure for performance test
        
        for model_name, calc_factory in working_models:
            print(f"\n   Performance testing {model_name.upper()}...")
            try:
                calc = calc_factory()
                
                # Time multiple calculations
                times = []
                for i in range(3):  # 3 runs for average
                    start_time = time.time()
                    result = calc.calculate_energy_forces(structure)
                    calc_time = time.time() - start_time
                    times.append(calc_time)
                
                avg_time = np.mean(times)
                std_time = np.std(times)
                throughput = len(structure) / avg_time  # atoms per second
                
                print(f"✅ {model_name.upper()} performance:")
                print(f"   Average time: {avg_time:.3f}±{std_time:.3f}s")
                print(f"   Throughput: {throughput:.1f} atoms/sec")
                
                performance_results[model_name] = {
                    'avg_time': avg_time,
                    'std_time': std_time,
                    'throughput': throughput,
                    'num_atoms': len(structure)
                }
                
            except Exception as e:
                print(f"❌ Performance test for {model_name.upper()} failed: {e}")
                performance_results[model_name] = {'error': str(e)}
        
        self.results['performance'] = performance_results
    
    def create_orb_calc(self):
        """Create ORB calculator."""
        from lematerial_forgebench.models.orb.calculator import ORBCalculator
        return ORBCalculator(device="cpu")
    
    def create_uma_calc(self):
        """Create UMA calculator."""
        from lematerial_forgebench.models.uma.calculator import UMACalculator
        return UMACalculator(model_name="uma-s-1", task="omat", device="cpu")
    
    def cross_model_comparison(self):
        """Compare results across different models."""
        self.print_subheader("Cross-Model Comparison")
        
        # Collect energies from all working models
        energies = {}
        
        if 'orb_direct' in self.results and self.results['orb_direct'].get('energy_calculated'):
            energies['ORB'] = self.results['orb_direct']['energy']
        
        if 'uma_direct' in self.results and self.results['uma_direct'].get('energy_calculated'):
            energies['UMA'] = self.results['uma_direct']['energy']
        
        if 'mace_direct' in self.results:
            for approach, result in self.results['mace_direct'].get('approaches', {}).items():
                if result.get('success'):
                    energies[f'MACE-{approach.upper()}'] = result['energy']
                    break  # Use first successful MACE result
        
        if len(energies) >= 2:
            print(f"\n📊 Energy Comparison ({self.test_structures[0].composition}):")
            for model, energy in energies.items():
                print(f"   {model}: {energy:.3f} eV")
            
            # Calculate statistics
            energy_values = list(energies.values())
            energy_range = max(energy_values) - min(energy_values)
            energy_mean = np.mean(energy_values)
            energy_std = np.std(energy_values)
            
            print(f"\n📈 Statistics:")
            print(f"   Mean: {energy_mean:.3f} eV")
            print(f"   Std Dev: {energy_std:.3f} eV")
            print(f"   Range: {energy_range:.3f} eV")
            
            # Assessment
            if energy_range < 1.0:
                print("✅ Models show good agreement (range < 1.0 eV)")
            elif energy_range < 5.0:
                print("⚠️ Models show moderate agreement (range < 5.0 eV)")
            else:
                print("❌ Models show poor agreement (range > 5.0 eV)")
            
            self.results['comparison'] = {
                'energies': energies,
                'mean': energy_mean,
                'std': energy_std,
                'range': energy_range
            }
        else:
            print("⚠️ Need at least 2 working models for comparison")
            self.results['comparison'] = {'insufficient_models': True}
    
    def generate_final_report(self):
        """Generate comprehensive final report."""
        self.print_header("FINAL COMPREHENSIVE REPORT")
        
        # Model Status Summary
        print("📊 MODEL STATUS SUMMARY")
        print("-" * 40)
        
        models_status = {}
        
        # ORB Status
        orb_status = "❌ Not Working"
        if self.results.get('orb_direct', {}).get('energy_calculated'):
            orb_status = "✅ Fully Working"
        elif self.results.get('orb_direct', {}).get('available'):
            orb_status = "⚠️ Partially Working"
        models_status['ORB'] = orb_status
        print(f"ORB: {orb_status}")
        
        # UMA Status
        uma_status = "❌ Not Working"
        if self.results.get('uma_direct', {}).get('energy_calculated'):
            uma_status = "✅ Fully Working"
        elif self.results.get('uma_direct', {}).get('available'):
            uma_status = "⚠️ Partially Working"
        models_status['UMA'] = uma_status
        print(f"UMA: {uma_status}")
        
        # MACE Status
        mace_status = "❌ Not Working"
        if self.results.get('mace_direct', {}).get('any_approach_worked'):
            mace_status = "✅ Working (some approaches)"
        elif self.results.get('mace_direct', {}).get('available'):
            mace_status = "⚠️ Available but not functional"
        models_status['MACE'] = mace_status
        print(f"MACE: {mace_status}")
        
        # Count working models
        fully_working = sum(1 for status in models_status.values() if "✅ Fully Working" in status or "✅ Working" in status)
        
        print(f"\n🎯 OVERALL STATUS: {fully_working}/3 models working")
        
        # Functionality Assessment
        print(f"\n🔧 FUNCTIONALITY ASSESSMENT")
        print("-" * 40)
        
        functionalities = {
            'Energy Calculations': 0,
            'Force Calculations': 0,
            'Embedding Extraction': 0,
            'Registry Integration': 0,
            'Benchmark Integration': 0
        }
        
        # Count functionalities
        for model in ['orb_direct', 'uma_direct', 'mace_direct']:
            if self.results.get(model, {}).get('energy_calculated'):
                functionalities['Energy Calculations'] += 1
                functionalities['Force Calculations'] += 1
            if self.results.get(model, {}).get('embeddings_extracted'):
                functionalities['Embedding Extraction'] += 1
        
        # Registry integration
        registry_working = sum(1 for model_result in self.results.get('registry', {}).values() 
                             if isinstance(model_result, dict) and model_result.get('success'))
        functionalities['Registry Integration'] = registry_working
        
        # Benchmark integration
        if self.results.get('benchmarks', {}).get('validity', {}).get('success'):
            functionalities['Benchmark Integration'] += 1
        
        for func, count in functionalities.items():
            status = "✅" if count > 0 else "❌"
            print(f"{status} {func}: {count}/3 models")
        
        # Performance Summary
        if 'performance' in self.results:
            print(f"\n⚡ PERFORMANCE SUMMARY")
            print("-" * 40)
            for model, perf in self.results['performance'].items():
                if 'avg_time' in perf:
                    print(f"{model.upper()}: {perf['avg_time']:.3f}s ({perf['throughput']:.1f} atoms/s)")
        
        # Issues and Recommendations
        print(f"\n💡 ISSUES AND RECOMMENDATIONS")
        print("-" * 40)
        
        if models_status['ORB'] != "✅ Fully Working":
            print("🔧 ORB Issues:")
            if 'error' in self.results.get('orb_direct', {}):
                print(f"   - {self.results['orb_direct']['error']}")
            print("   - Try: uv sync --extra orb")
        
        if models_status['UMA'] != "✅ Fully Working":
            print("🔧 UMA Issues:")
            if 'error' in self.results.get('uma_direct', {}):
                print(f"   - {self.results['uma_direct']['error']}")
            print("   - Try: uv pip install --upgrade fairchem-core")
        
        if models_status['MACE'] != "✅ Working (some approaches)":
            print("🔧 MACE Issues:")
            print("   - Deep compatibility issue with e3nn library")
            print("   - Try: uv pip install --upgrade mace-torch")
            print("   - Try: rm -rf ~/.cache/mace/")
            print("   - Check PyTorch/e3nn version compatibility")
        
        # Registry Issues
        registry_issues = []
        for model, result in self.results.get('registry', {}).items():
            if isinstance(result, dict) and not result.get('success'):
                if result.get('issue_type') == 'parameter_conflict':
                    registry_issues.append(f"{model.upper()}: Parameter conflict in factory function")
                elif result.get('issue_type') == 'compatibility_issue':
                    registry_issues.append(f"{model.upper()}: Model loading compatibility issue")
        
        if registry_issues:
            print("🔧 Registry Issues:")
            for issue in registry_issues:
                print(f"   - {issue}")
        
        # Success Criteria
        print(f"\n🎯 SUCCESS CRITERIA ASSESSMENT")
        print("-" * 40)
        
        criteria = {
            "At least 2 models working": fully_working >= 2,
            "Energy calculations functional": functionalities['Energy Calculations'] >= 2,
            "At least 1 model with embeddings": functionalities['Embedding Extraction'] >= 1,
            "Benchmarks can run": functionalities['Benchmark Integration'] >= 1,
            "Framework is usable": fully_working >= 1 and functionalities['Energy Calculations'] >= 1
        }
        
        for criterion, met in criteria.items():
            status = "✅" if met else "❌"
            print(f"{status} {criterion}")
        
        overall_success = all(criteria.values())
        partial_success = sum(criteria.values()) >= 3
        
        print(f"\n🏆 FINAL VERDICT")
        print("=" * 40)
        
        if overall_success:
            print("🎉 EXCELLENT! All success criteria met.")
            print("💪 The ForgeBench framework is fully functional!")
            print("🚀 You can proceed with confidence.")
        elif partial_success:
            print("👍 GOOD! Most success criteria met.")
            print("💡 The framework is usable with some limitations.")
            print("🔧 Consider fixing remaining issues for full functionality.")
        else:
            print("⚠️ NEEDS WORK! Major issues detected.")
            print("🛠️ Focus on fixing core model issues before proceeding.")
        
        # Next Steps
        print(f"\n📋 RECOMMENDED NEXT STEPS")
        print("-" * 40)
        
        if fully_working >= 2:
            print("1. ✅ Continue development with working models")
            print("2. 🧪 Run more comprehensive tests on your datasets")
            print("3. 🔧 Fix registry issues for smoother integration")
            print("4. 📊 Benchmark performance on larger structures")
        elif fully_working >= 1:
            print("1. 🔧 Fix the non-working models for full functionality")
            print("2. 🧪 Test the working model(s) thoroughly")
            print("3. 📚 Review installation/dependency documentation")
        else:
            print("1. 🛠️ Focus on dependency/installation issues")
            print("2. 📞 Consider reaching out for support")
            print("3. 🔍 Check system compatibility")
        
        # Save results to file
        try:
            import json
            with open('comprehensive_test_results.json', 'w') as f:
                # Convert numpy types for JSON serialization
                json_results = self.convert_for_json(self.results)
                json.dump(json_results, f, indent=2)
            print(f"\n💾 Detailed results saved to: comprehensive_test_results.json")
        except Exception as e:
            print(f"⚠️ Could not save results to file: {e}")
        
        return overall_success or partial_success
    
    def convert_for_json(self, obj):
        """Convert numpy types and other non-serializable objects for JSON."""
        if isinstance(obj, dict):
            return {k: self.convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj
    
    def run_all_tests(self):
        """Run the complete test suite."""
        print("🚀 STARTING COMPREHENSIVE MODEL TESTING")
        print("=" * 60)
        print("This will test every aspect of MACE, ORB, and UMA integration.")
        print("Testing: availability, direct usage, registry, benchmarks, performance")
        print("=" * 60)
        
        try:
            # Core tests
            self.test_model_availability()
            self.test_orb_direct()
            self.test_uma_direct()
            self.test_mace_direct()
            
            # Integration tests
            self.test_registry_integration()
            self.test_benchmarks()
            
            # Analysis tests
            self.test_performance()
            self.cross_model_comparison()
            
            # Final report
            return self.generate_final_report()
            
        except KeyboardInterrupt:
            print("\n⚠️ Testing interrupted by user")
            return False
        except Exception as e:
            print(f"\n❌ Unexpected error during testing: {e}")
            traceback.print_exc()
            return False


def main():
    """Main function to run comprehensive testing."""
    tester = ComprehensiveModelTester()
    success = tester.run_all_tests()
    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)