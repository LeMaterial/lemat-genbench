"""
Model-focused comprehensive test script focusing on registry-based model 
testing. This tests all models through the ForgeBench registry system and 
focuses on  model-dependent benchmarks.
"""

import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from pymatgen.util.testing import PymatgenTest


class ModelFocusedRegistryTester:
    """Model-focused testing suite for registry-based model testing."""

    def __init__(self):
        """Initialize the tester with empty results and test structures."""
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

            # Add a simple structure for stress testing
            from pymatgen.core import Lattice, Structure

            simple_structure = Structure(
                Lattice.cubic(4.0), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
            )
            self.test_structures.append(simple_structure)

            # Add CIF structures if available
            cif_files = ["CsBr.cif", "CsPbBr3.cif", "NiO.cif"]
            for cif_file in cif_files:
                file_paths = [
                    Path(cif_file),
                    Path(f"notebooks/{cif_file}")
                ]
                
                for file_path in file_paths:
                    if file_path.exists():
                        try:
                            from pymatgen.core import Structure

                            structure = Structure.from_file(str(file_path))
                            structure = structure.remove_oxidation_states()
                            self.test_structures.append(structure)
                            break
                        except Exception as exc:
                            print(f"⚠️ Could not load {cif_file}: {exc}")

            print(f"✅ Setup {len(self.test_structures)} test structures")
            for i, struct in enumerate(self.test_structures):
                print(
                    f"   {i + 1}. {struct.composition} "
                    f"({len(struct)} atoms)"
                )

        except Exception as exc:
            print(f"⚠️ Error setting up structures: {exc}")
            # Fallback: create simple structures manually
            from pymatgen.core import Lattice, Structure

            si_structure = Structure(
                Lattice.cubic(5.43),
                ["Si", "Si"],
                [[0, 0, 0], [0.25, 0.25, 0.25]]
            )
            self.test_structures = [si_structure]

    def print_header(self, title: str):
        """Print formatted section header."""
        print(f"\n{'=' * 70}")
        print(f"🧪 {title}")
        print(f"{'=' * 70}")

    def print_subheader(self, title: str):
        """Print formatted subsection header."""
        print(f"\n{'-' * 50}")
        print(f"🔍 {title}")
        print(f"{'-' * 50}")

    def test_registry_discovery(self):
        """Test registry model discovery and information."""
        self.print_header("REGISTRY MODEL DISCOVERY")

        try:
            from lematerial_forgebench.models.registry import (
                get_model_info,
                is_model_available,
                list_available_models,
                print_model_info,
            )

            # Get available models
            available_models = list_available_models()
            print(
                f"📋 Registry reports {len(available_models)} available MLIPs: "
                f"{available_models}"
            )

            # Test individual model availability
            model_availability = {}
            for model in ["orb", "mace", "uma", "equiformer"]:
                is_available = is_model_available(model)
                model_availability[model] = is_available
                status = "✅" if is_available else "❌"
                print(f"   {status} {model.upper()}: {is_available}")

            # Get detailed model info
            print("\n📊 Detailed Model Information:")
            try:
                print_model_info()
                model_info = get_model_info()
                self.results["registry_discovery"] = {
                    "available_models": available_models,
                    "model_availability": model_availability,
                    "model_info": model_info,
                }
            except Exception as exc:
                print(f"⚠️ Error getting model info: {exc}")
                self.results["registry_discovery"] = {
                    "available_models": available_models,
                    "model_availability": model_availability,
                    "info_error": str(exc),
                }

        except Exception as exc:
            print(f"❌ Error in registry discovery: {exc}")
            traceback.print_exc()
            self.results["registry_discovery"] = {"error": str(exc)}

    def _create_calculator(self, model_name: str):
        """Create calculator for the specified model.
        
        Args:
            model_name: Name of the model to create calculator for
            
        Returns:
            Calculator instance
            
        Raises:
            Exception: If calculator creation fails
        """
        from lematerial_forgebench.models.registry import get_calculator

        if model_name == "mace":
            # Test both approaches for MACE
            print("      Trying MACE with default settings...")
            try:
                calc = get_calculator(
                    "mace",
                    model_type="mp",
                    device="cpu",
                )
                print("      ✅ MACE default settings worked")
                return calc
            except Exception as exc:
                print(f"      ⚠️ MACE default failed: {exc}")
                print("      Trying MACE with small model...")
                calc = get_calculator(
                    "mace",
                    model_type="mp",
                    device="cpu",
                    model_size="small"
                )
                print("      ✅ MACE small model worked")
                return calc, "Required small model for MACE"

        elif model_name == "orb":
            return get_calculator("orb", device="cpu")

        elif model_name == "uma":
            return get_calculator("uma", task="omat", device="cpu")

        elif model_name == "equiformer":
            return get_calculator("equiformer", device="cpu")

        else:
            return get_calculator(model_name, device="cpu")

    def _test_basic_calculation(self, calc, model_results: Dict[str, Any]) -> bool:
        """Test basic energy/force calculation.
        
        Args:
            calc: Calculator instance
            model_results: Dictionary to store results
            
        Returns:
            True if successful, False otherwise
        """
        print("   ⚡ Testing basic energy/force calculation...")

        try:
            structure = self.test_structures[0]

            start_time = time.time()
            result = calc.calculate_energy_forces(structure)
            calc_time = time.time() - start_time

            print(f"      ✅ Energy: {result.energy:.4f} eV")
            print(f"      ✅ Forces shape: {result.forces.shape}")
            print(f"      ✅ Calculation time: {calc_time:.3f}s")

            if result.stress is not None:
                print(f"      ✅ Stress available: {result.stress.shape}")

            model_results.update({
                "basic_calculation": True,
                "basic_energy": result.energy,
                "basic_calc_time": calc_time,
                "forces_shape": result.forces.shape,
            })
            return True

        except Exception as exc:
            print(f"❌ Basic calculation failed: {type(exc).__name__}: {exc}")
            model_results["errors"].append(f"Basic calculation: {str(exc)}")
            return False

    def _test_all_structures(self, calc, model_results: Dict[str, Any]):
        """Test calculations on all structures.
        
        Args:
            calc: Calculator instance
            model_results: Dictionary to store results
        """
        print(f"🧮 Testing all {len(self.test_structures)} structures")

        successful_calcs = 0
        structure_results = []

        for i, structure in enumerate(self.test_structures):
            try:
                print(
                    f"Structure {i + 1}: {structure.composition} "
                    f"({len(structure)} atoms)"
                )

                start_time = time.time()
                result = calc.calculate_energy_forces(structure)
                calc_time = time.time() - start_time

                successful_calcs += 1
                structure_results.append({
                    "composition": str(structure.composition),
                    "n_atoms": len(structure),
                    "energy": result.energy,
                    "calc_time": calc_time,
                    "energy_per_atom": result.energy / len(structure),
                })

                print(
                    f"✅ E = {result.energy:.3f} eV "
                    f"({result.energy / len(structure):.3f} eV/atom)"
                )

            except Exception as exc:
                print(f"         ❌ Failed: {exc}")
                model_results["warnings"].append(
                    f"Structure {i + 1} failed: {str(exc)}"
                )

        model_results.update({
            "all_structures_tested": successful_calcs == len(self.test_structures),
            "successful_structures": successful_calcs,
            "structure_results": structure_results,
        })

        print(
            f"📊 Success rate: {successful_calcs}/"
            f"{len(self.test_structures)} structures"
        )

    def _test_embeddings(self, calc, model_results: Dict[str, Any]):
        """Test embedding extraction.
        
        Args:
            calc: Calculator instance
            model_results: Dictionary to store results
        """
        print("   🧬 Testing embedding extraction...")

        try:
            structure = self.test_structures[0]
            embedding_result = calc.extract_embeddings(structure)

            if (hasattr(embedding_result, "node_embeddings") 
                and embedding_result.node_embeddings is not None):
                embeddings = embedding_result.node_embeddings
                print(f"      ✅ Node embeddings: {embeddings.shape}")
                model_results["node_embeddings_shape"] = embeddings.shape

            if (hasattr(embedding_result, "graph_embeddings") 
                and embedding_result.graph_embeddings is not None):
                graph_emb = embedding_result.graph_embeddings
                print(f"      ✅ Graph embeddings: {graph_emb.shape}")
                model_results["graph_embeddings_shape"] = graph_emb.shape

            if (hasattr(embedding_result, "embeddings") 
                and embedding_result.embeddings is not None):
                embeddings = embedding_result.embeddings
                if hasattr(embeddings, "shape"):
                    print(f"      ✅ Embeddings: {embeddings.shape}")
                    model_results["embeddings_shape"] = embeddings.shape
                else:
                    print(f"Embeddings extracted (type: {type(embeddings)})")

            model_results["embeddings_extracted"] = True

        except Exception as exc:
            print(f"      ❌ Embeddings failed: {type(exc).__name__}: {exc}")
            model_results["errors"].append(f"Embeddings: {str(exc)}")

    def _test_formation_energy(self, calc, model_results: Dict[str, Any]):
        """Test formation energy calculation.
        
        Args:
            calc: Calculator instance
            model_results: Dictionary to store results
        """
        print("   🔥 Testing formation energy calculation...")

        try:
            structure = self.test_structures[0]
            formation_energy = calc.calculate_formation_energy(structure)

            print(f"✅ Formation energy: {formation_energy:.4f} eV/atom")
            model_results.update({
                "formation_energy_calculated": True,
                "formation_energy": formation_energy,
            })

        except Exception as exc:
            print(f"❌ Formation energy failed: {type(exc).__name__}: {exc}")
            model_results["errors"].append(f"Formation energy: {str(exc)}")

    def _test_energy_above_hull(self, calc, model_results: Dict[str, Any]):
        """Test energy above hull calculation.
        
        Args:
            calc: Calculator instance
            model_results: Dictionary to store results
        """
        print("   ⛰️  Testing energy above hull calculation...")

        try:
            structure = self.test_structures[0]
            e_above_hull = calc.calculate_energy_above_hull(structure)

            print(f"      ✅ Energy above hull: {e_above_hull:.4f} eV/atom")
            model_results.update({
                "energy_above_hull_calculated": True,
                "energy_above_hull": e_above_hull,
            })

        except Exception as exc:
            print(f"❌ Energy above hull failed: {type(exc).__name__}: {exc}")
            model_results["errors"].append(f"Energy above hull: {str(exc)}")

    def _test_performance(self, calc, model_results: Dict[str, Any]):
        """Test performance benchmarking.
        
        Args:
            calc: Calculator instance
            model_results: Dictionary to store results
        """
        print("   ⚡ Performance testing...")

        try:
            structure = self.test_structures[0]
            times = []

            # Warmup run
            calc.calculate_energy_forces(structure)

            # Timed runs
            for _ in range(3):
                start_time = time.time()
                calc.calculate_energy_forces(structure)
                times.append(time.time() - start_time)

            avg_time = np.mean(times)
            std_time = np.std(times)
            throughput = len(structure) / avg_time

            print(f"✅ Average time: {avg_time:.3f}±{std_time:.3f}s")
            print(f"✅ Throughput: {throughput:.1f} atoms/sec")

            model_results.update({
                "performance_tested": True,
                "avg_calc_time": avg_time,
                "std_calc_time": std_time,
                "throughput": throughput,
            })

        except Exception as exc:
            print(f"❌ Performance test failed: {exc}")
            model_results["warnings"].append(f"Performance test: {str(exc)}")

    def test_model_through_registry(self, model_name: str) -> Dict[str, Any]:
        """Comprehensive test of a single model through registry.
        
        Args:
            model_name: Name of the model to test
            
        Returns:
            Dictionary containing test results
        """
        self.print_subheader(f"{model_name.upper()} Registry Testing")

        model_results = {
            "model_name": model_name,
            "calculator_created": False,
            "basic_calculation": False,
            "all_structures_tested": False,
            "embeddings_extracted": False,
            "formation_energy_calculated": False,
            "energy_above_hull_calculated": False,
            "performance_tested": False,
            "errors": [],
            "warnings": [],
        }

        try:
            # Step 1: Create calculator through registry
            print(f"   🔧 Creating {model_name.upper()} calculator...")

            try:
                calc_result = self._create_calculator(model_name)
                
                # Handle tuple return from MACE
                if isinstance(calc_result, tuple):
                    calc, warning = calc_result
                    model_results["warnings"].append(warning)
                else:
                    calc = calc_result

                model_results["calculator_created"] = True
                print(f"✅ {model_name.upper()} calculator created successfully")

            except Exception as exc:
                print(
                    f"      ❌ Calculator creation failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                model_results["errors"].append(f"Calculator creation: {str(exc)}")
                return model_results

            # Step 2: Basic energy/force calculation
            basic_success = self._test_basic_calculation(calc, model_results)

            # Step 3: Test all structures
            if basic_success:
                self._test_all_structures(calc, model_results)

            # Step 4: Test embeddings
            self._test_embeddings(calc, model_results)

            # Step 5: Test formation energy
            self._test_formation_energy(calc, model_results)

            # Step 6: Test energy above hull
            self._test_energy_above_hull(calc, model_results)

            # Step 7: Performance testing
            if basic_success:
                self._test_performance(calc, model_results)

        except Exception as exc:
            print(f"❌ Unexpected error in {model_name.upper()} testing: {exc}")
            traceback.print_exc()
            model_results["errors"].append(f"Unexpected error: {str(exc)}")

        # Calculate overall success score
        success_criteria = [
            model_results["calculator_created"],
            model_results["basic_calculation"],
            model_results.get("successful_structures", 0) >= len(self.test_structures) // 2,
            len(model_results["errors"]) <= 2,
        ]

        model_results["success_score"] = sum(success_criteria) / len(success_criteria)

        # Print verdict
        self._print_model_verdict(model_name, model_results["success_score"])

        return model_results

    def _print_model_verdict(self, model_name: str, success_score: float):
        """Print verdict for model test.
        
        Args:
            model_name: Name of the model
            success_score: Success score (0-1)
        """
        if success_score >= 0.75:
            print(
                f" 🎉 {model_name.upper()}: EXCELLENT (score: "
                f"{success_score:.2f})"
            )
        elif success_score >= 0.5:
            print(
                f" 👍 {model_name.upper()}: GOOD (score: "
                f"{success_score:.2f})"
            )
        elif success_score >= 0.25:
            print(
                f" ⚠️ {model_name.upper()}: PARTIAL (score: "
                f"{success_score:.2f})"
            )
        else:
            print(
                f" ❌ {model_name.upper()}: POOR (score: "
                f"{success_score:.2f})"
            )

    def test_all_models_through_registry(self):
        """Test all available models through registry."""
        self.print_header("COMPREHENSIVE REGISTRY-BASED MODEL TESTING")

        # Get available models
        try:
            from lematerial_forgebench.models.registry import list_available_models
            available_models = list_available_models()
        except Exception as exc:
            print(f"❌ Could not get available models: {exc}")
            available_models = ["orb", "mace", "uma"]  # Fallback

        print(f"🎯 Testing {len(available_models)} models: {available_models}")

        all_model_results = {}

        for model_name in available_models:
            try:
                model_result = self.test_model_through_registry(model_name)
                all_model_results[model_name] = model_result
            except Exception as exc:
                print(f"❌ Failed to test {model_name}: {exc}")
                all_model_results[model_name] = {
                    "model_name": model_name,
                    "fatal_error": str(exc),
                    "success_score": 0.0,
                }

        self.results["model_tests"] = all_model_results

    def test_model_dependent_benchmarks(self):
        """Test model-dependent benchmarks with ALL working models."""
        self.print_header("MODEL-DEPENDENT BENCHMARK TESTING")

        benchmark_results = {}

        # Find working models
        working_models = []
        if "model_tests" in self.results:
            for model_name, results in self.results["model_tests"].items():
                if results.get("success_score", 0) >= 0.5:
                    working_models.append(model_name)

        print(
            f"🎯 Testing model-dependent benchmarks with "
            f"{len(working_models)} working models: "
            f"{working_models}"
        )

        if not working_models:
            print("❌ No working models found for benchmark testing")
            self.results["benchmarks"] = {"error": "No working models available"}
            return

        # Test Stability Benchmark with ALL working models
        for model_name in working_models:
            print(f"\nTesting Stability Benchmark with {model_name.upper()}...")
            try:
                from lematerial_forgebench.benchmarks.stability_benchmark import StabilityBenchmark

                # Create benchmark with the specific model
                stability_benchmark = StabilityBenchmark(model_name=model_name)
                result = stability_benchmark.evaluate(self.test_structures[:2])

                print(f"   ✅ Stability benchmark with {model_name.upper()} completed")
                print(f"   📊 Final scores keys: {list(result.final_scores.keys())}")

                # Show key metrics
                key_metrics = ["stable_ratio", "metastable_ratio", "mean_e_above_hull"]
                for metric in key_metrics:
                    if metric in result.final_scores:
                        print(f"      {metric}: {result.final_scores[metric]:.4f}")

                benchmark_results[f"stability_{model_name}"] = {
                    "success": True,
                    "all_scores": result.final_scores,
                    "model_name": model_name,
                }

            except Exception as exc:
                print(
                    f"   ❌ Stability benchmark with {model_name.upper()} failed: {exc}"
                )
                benchmark_results[f"stability_{model_name}"] = {
                    "success": False,
                    "error": str(exc),
                    "model_name": model_name,
                }

        self.results["benchmarks"] = benchmark_results

    def _calculate_energy_statistics(self, model_energies: Dict[str, float]) -> Dict[str, float]:
        """Calculate energy statistics for cross-model analysis.
        
        Args:
            model_energies: Dictionary of model names to energies
            
        Returns:
            Dictionary of statistics
        """
        energies = list(model_energies.values())
        return {
            "mean": np.mean(energies),
            "std": np.std(energies),
            "range": max(energies) - min(energies),
            "min": min(energies),
            "max": max(energies),
        }

    def _assess_model_agreement(self, energy_range: float, n_atoms: int) -> str:
        """Assess agreement between models based on energy range.
        
        Args:
            energy_range: Range of energies across models
            n_atoms: Number of atoms in structure
            
        Returns:
            Agreement assessment string
        """
        per_atom_range = energy_range / n_atoms
        
        if per_atom_range < 0.1:
            return "✅ Excellent model agreement (range < 0.1 eV/atom)"
        elif per_atom_range < 0.5:
            return "👍 Good model agreement (range < 0.5 eV/atom)"
        elif per_atom_range < 1.0:
            return "⚠️ Moderate model agreement (range < 1.0 eV/atom)"
        else:
            return "❌ Poor model agreement (range > 1.0 eV/atom)"

    def cross_model_analysis(self):
        """Perform cross-model analysis."""
        self.print_header("CROSS-MODEL ANALYSIS")

        if "model_tests" not in self.results:
            print("❌ No model test results available for analysis")
            return

        # Collect energies and performance data
        model_data = {
            "total_energies": {},
            "formation_energies": {},
            "energy_above_hull": {},
            "performance": {},
        }

        structure_composition = str(self.test_structures[0].composition)

        for model_name, results in self.results["model_tests"].items():
            if results.get("basic_calculation") and "basic_energy" in results:
                model_data["total_energies"][model_name] = results["basic_energy"]

            if results.get("formation_energy_calculated"):
                model_data["formation_energies"][model_name] = results["formation_energy"]

            if results.get("energy_above_hull_calculated"):
                model_data["energy_above_hull"][model_name] = results["energy_above_hull"]

            if results.get("performance_tested"):
                model_data["performance"][model_name] = {
                    "avg_time": results.get("avg_calc_time", 0),
                    "throughput": results.get("throughput", 0),
                }

        # Energy comparison
        if len(model_data["total_energies"]) >= 2:
            print(f"📊 Total Energy Comparison for {structure_composition}:")

            for model, energy in model_data["total_energies"].items():
                energy_per_atom = energy / len(self.test_structures[0])
                print(
                    f"   {model.upper()}: {energy:.3f} eV "
                    f"({energy_per_atom:.3f} eV/atom)"
                )

            # Statistics
            stats = self._calculate_energy_statistics(model_data["total_energies"])

            print("\n📈 Total Energy Statistics:")
            print(f"   Mean: {stats['mean']:.3f} eV")
            print(f"   Std Dev: {stats['std']:.3f} eV")
            print(f"   Range: {stats['range']:.3f} eV")

            # Agreement assessment
            agreement = self._assess_model_agreement(
                stats['range'], len(self.test_structures[0])
            )
            print(agreement)

        # Formation energy comparison
        if len(model_data["formation_energies"]) >= 2:
            self._print_formation_energy_comparison(
                model_data["formation_energies"], structure_composition
            )

        # Energy above hull comparison
        if len(model_data["energy_above_hull"]) >= 2:
            self._print_energy_above_hull_comparison(
                model_data["energy_above_hull"], structure_composition
            )

        # Performance comparison
        if len(model_data["performance"]) >= 2:
            self._print_performance_comparison(model_data["performance"])

        # Benchmark comparison
        if "benchmarks" in self.results:
            self._print_benchmark_comparison()

        # Store comparison results
        self.results["cross_model_analysis"] = model_data

    def _print_formation_energy_comparison(self, formation_energies: Dict[str, float], 
                                         composition: str):
        """Print formation energy comparison.
        
        Args:
            formation_energies: Dictionary of model to formation energy
            composition: Structure composition string
        """
        print(f"\n🔥 Formation Energy Comparison for {composition}:")

        for model, form_energy in formation_energies.items():
            print(f"   {model.upper()}: {form_energy:.4f} eV/atom")

        # Statistics
        form_energies = list(formation_energies.values())
        form_energy_range = max(form_energies) - min(form_energies)
        form_energy_mean = np.mean(form_energies)

        print("\n📈 Formation Energy Statistics:")
        print(f"   Mean: {form_energy_mean:.4f} eV/atom")
        print(f"   Range: {form_energy_range:.4f} eV/atom")

    def _print_energy_above_hull_comparison(self, energy_above_hull: Dict[str, float], 
                                          composition: str):
        """Print energy above hull comparison.
        
        Args:
            energy_above_hull: Dictionary of model to energy above hull
            composition: Structure composition string
        """
        print(f"\n⛰️  Energy Above Hull Comparison for {composition}:")

        for model, e_hull in energy_above_hull.items():
            print(f"   {model.upper()}: {e_hull:.4f} eV/atom")

        # Statistics
        e_hull_values = list(energy_above_hull.values())
        e_hull_range = max(e_hull_values) - min(e_hull_values)
        e_hull_mean = np.mean(e_hull_values)

        print("\n📈 Energy Above Hull Statistics:")
        print(f"   Mean: {e_hull_mean:.4f} eV/atom")
        print(f"   Range: {e_hull_range:.4f} eV/atom")

    def _print_performance_comparison(self, performance_data: Dict[str, Dict[str, float]]):
        """Print performance comparison.
        
        Args:
            performance_data: Dictionary of model to performance metrics
        """
        print("\n⚡ Performance Comparison:")

        for model, perf in performance_data.items():
            print(
                f"   {model.upper()}: {perf['avg_time']:.3f}s "
                f"({perf['throughput']:.1f} atoms/s)"
            )

        # Find fastest
        fastest_model = min(
            performance_data.items(), key=lambda x: x[1]["avg_time"]
        )
        print(f"🏆 Fastest model: {fastest_model[0].upper()}")

    def _print_benchmark_comparison(self):
        """Print benchmark comparison results."""
        print("\n📊 Model-Dependent Benchmark Comparison:")

        # Group benchmarks by type
        stability_results = {}

        for bench_name, bench_result in self.results["benchmarks"].items():
            if bench_name.startswith("stability_") and bench_result.get("success"):
                model_name = bench_result.get(
                    "model_name", bench_name.split("_")[1]
                )
                stability_results[model_name] = bench_result["all_scores"]

        if stability_results:
            print("\n   ⚖️  Stability Benchmark Results:")
            for model, scores in stability_results.items():
                key_metric = scores.get(
                    "stable_ratio", scores.get("metastable_ratio", "N/A")
                )
                print(f"      {model.upper()}: {key_metric}")

    def _calculate_success_criteria(self) -> Dict[str, bool]:
        """Calculate success criteria for final assessment.
        
        Returns:
            Dictionary of criteria and whether they were met
        """
        if "model_tests" not in self.results:
            return {}

        fully_working = sum(
            1 for r in self.results["model_tests"].values()
            if r.get("success_score", 0) >= 0.75
        )
        
        partially_working = sum(
            1 for r in self.results["model_tests"].values()
            if 0.25 <= r.get("success_score", 0) < 0.75
        )

        stability_success = sum(
            1 for k, v in self.results.get("benchmarks", {}).items()
            if k.startswith("stability_") and v.get("success", False)
        )

        basic_calc_working = sum(
            1 for r in self.results["model_tests"].values()
            if r.get("basic_calculation", False)
        )

        formation_energy_working = sum(
            1 for r in self.results["model_tests"].values()
            if r.get("formation_energy_calculated", False)
        )

        energy_hull_working = sum(
            1 for r in self.results["model_tests"].values()
            if r.get("energy_above_hull_calculated", False)
        )

        return {
            "At least 1 model fully working": fully_working >= 1,
            "At least 2 models working": (fully_working + partially_working) >= 2,
            "Basic calculations functional": basic_calc_working >= 1,
            "Formation energy calculation": formation_energy_working >= 1,
            "Energy above hull calculation": energy_hull_working >= 1,
            "Model-dependent benchmarks working": stability_success > 0,
            "Cross-model comparison possible": fully_working >= 2,
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results.
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if "model_tests" not in self.results:
            return ["🛠️ No model tests available for analysis"]

        # Count working models
        fully_working = sum(
            1 for r in self.results["model_tests"].values()
            if r.get("success_score", 0) >= 0.75
        )

        if fully_working == 0:
            recommendations.extend([
                "🛠️ URGENT: No models are fully working - check dependencies",
                "📞 Consider reaching out for technical support"
            ])
        elif fully_working < 2:
            recommendations.extend([
                "🔧 Focus on fixing remaining models for redundancy",
                "📚 Review installation documentation"
            ])
        else:
            recommendations.extend([
                "✅ Good model coverage - ready for production benchmarks",
                "🚀 Consider testing with larger datasets"
            ])

        # Model-specific recommendations
        model_results = self.results["model_tests"]
        
        if model_results.get("mace", {}).get("success_score", 0) < 0.5:
            recommendations.extend([
                "🔧 MACE: Apply monkey patches and try small model",
                "🔧 MACE: Clear cache: rm -rf ~/.cache/mace/"
            ])

        if model_results.get("uma", {}).get("success_score", 0) < 0.5:
            recommendations.append(
                "🔧 UMA: Update fairchem: uv add --upgrade fairchem-core"
            )

        if model_results.get("orb", {}).get("success_score", 0) < 0.5:
            recommendations.append("🔧 ORB: Install with: uv sync --extra orb")

        return recommendations

    def generate_final_report(self) -> bool:
        """Generate final comprehensive report.
        
        Returns:
            True if test was successful (>= 60% success rate), False otherwise
        """
        self.print_header("FINAL MODEL-FOCUSED REPORT")

        # Model Status Overview
        print("📊 MODEL STATUS OVERVIEW")
        print("-" * 60)

        if "model_tests" not in self.results:
            print("❌ No model test results available")
            return False

        working_models = []
        partially_working = []
        broken_models = []

        for model_name, results in self.results["model_tests"].items():
            success_score = results.get("success_score", 0.0)

            if success_score >= 0.75:
                status = "✅ EXCELLENT"
                working_models.append(model_name)
            elif success_score >= 0.5:
                status = "👍 GOOD"
                working_models.append(model_name)
            elif success_score >= 0.25:
                status = "⚠️ PARTIAL"
                partially_working.append(model_name)
            else:
                status = "❌ BROKEN"
                broken_models.append(model_name)

            # Detailed breakdown
            calc_created = "✅" if results.get("calculator_created") else "❌"
            basic_calc = "✅" if results.get("basic_calculation") else "❌"
            embeddings = "✅" if results.get("embeddings_extracted") else "❌"
            formation_e = "✅" if results.get("formation_energy_calculated") else "❌"
            e_hull = "✅" if results.get("energy_above_hull_calculated") else "❌"

            print(f"{model_name.upper()}: {status} (Score: {success_score:.2f})")
            print(
                f"   Calculator: {calc_created} | Basic Calc: {basic_calc} | "
                f"Embeddings: {embeddings}"
            )
            print(f"   Formation E: {formation_e} | E Above Hull: {e_hull}")

            if results.get("errors"):
                error_preview = results['errors'][0] if results['errors'] else 'None'
                print(f"   Errors: {len(results['errors'])} - {error_preview}")

            if results.get("warnings"):
                print(f"   Warnings: {len(results['warnings'])}")

        # Summary statistics
        total_models = len(self.results["model_tests"])
        fully_working = len(working_models)
        partially_working_count = len(partially_working)
        broken_count = len(broken_models)

        print("\n🎯 SUMMARY STATISTICS")
        print("-" * 60)
        print(f"Total models tested: {total_models}")
        print(
            f"Fully working: {fully_working} "
            f"({fully_working / total_models * 100:.1f}%)"
        )
        print(
            f"Partially working: {partially_working_count} "
            f"({partially_working_count / total_models * 100:.1f}%)"
        )
        print(f"Broken: {broken_count} ({broken_count / total_models * 100:.1f}%)")

        # Model-dependent benchmark results
        self._print_benchmark_summary()

        # Cross-model comparison summary
        self._print_cross_model_summary()

        # Success criteria assessment
        criteria = self._calculate_success_criteria()
        passed_criteria = sum(criteria.values())
        success_percentage = passed_criteria / len(criteria) * 100 if criteria else 0

        print("\n🎯 SUCCESS CRITERIA ASSESSMENT")
        print("-" * 60)

        for criterion, met in criteria.items():
            status = "✅" if met else "❌"
            print(f"{status} {criterion}")

        # Final verdict
        verdict_info = self._generate_verdict(success_percentage)
        print("\n🏆 FINAL VERDICT")
        print("=" * 60)
        print(f"{verdict_info['verdict']}: {success_percentage:.1f}% success rate")
        print(f"📝 {verdict_info['message']}")

        print("\n📋 RECOMMENDED NEXT STEPS:")
        for i, step in enumerate(verdict_info['next_steps'], 1):
            print(f"   {i}. {step}")

        # MACE-specific status
        self._print_mace_status()

        # Generate recommendations
        recommendations = self._generate_recommendations()
        if recommendations:
            print("\n💡 ISSUES AND RECOMMENDATIONS")
            print("-" * 60)
            for rec in recommendations:
                print(f"   {rec}")

        # Save detailed results
        self._save_results(success_percentage, verdict_info['verdict'], working_models, total_models)

        return success_percentage >= 60

    def _print_benchmark_summary(self):
        """Print benchmark summary section."""
        if "benchmarks" not in self.results:
            return

        print("\n🧪 MODEL-DEPENDENT BENCHMARK RESULTS")
        print("-" * 60)

        # Count successful benchmarks by type
        stability_success = 0
        stability_total = 0
        for bench_name, result in self.results["benchmarks"].items():
            if bench_name.startswith("stability_"):
                stability_total += 1
                if result.get("success", False):
                    stability_success += 1

        print(
            f"⚖️ Stability Benchmark: {stability_success}/{stability_total} "
            f"models successful"
        )
        
        # Show detailed results for successful benchmarks
        if stability_success > 0:
            print("\n   Stability Results by Model:")
            for bench_name, result in self.results["benchmarks"].items():
                if bench_name.startswith("stability_") and result.get("success"):
                    model_name = result.get("model_name", bench_name.split("_")[1])
                    scores = result["all_scores"]
                    stable_ratio = scores.get("stable_ratio", "N/A")
                    mean_e_hull = scores.get("mean_e_above_hull", "N/A")
                    print(
                        f"      {model_name.upper()}: Stable ratio: {stable_ratio}, "
                        f"Mean E above hull: {mean_e_hull}"
                    )

    def _print_cross_model_summary(self):
        """Print cross-model comparison summary."""
        if "cross_model_analysis" not in self.results:
            return

        comp_data = self.results["cross_model_analysis"]

        print("\n📊 CROSS-MODEL COMPARISON SUMMARY")
        print("-" * 60)

        if (comp_data.get("total_energies") 
            and len(comp_data["total_energies"]) >= 2):
            energies = list(comp_data["total_energies"].values())
            energy_range = max(energies) - min(energies)
            per_atom_range = energy_range / len(self.test_structures[0])

            agreement = self._assess_model_agreement(energy_range, len(self.test_structures[0]))
            print(f"Energy Agreement: {agreement}")

        if comp_data.get("performance") and len(comp_data["performance"]) >= 2:
            perf_data = comp_data["performance"]
            fastest = min(perf_data.items(), key=lambda x: x[1]["avg_time"])
            slowest = max(perf_data.items(), key=lambda x: x[1]["avg_time"])
            speed_ratio = slowest[1]["avg_time"] / fastest[1]["avg_time"]

            print(f"Performance: {fastest[0].upper()} fastest ({fastest[1]['avg_time']:.3f}s)")
            print(f"           {slowest[0].upper()} slowest ({slowest[1]['avg_time']:.3f}s)")
            print(f"           Speed ratio: {speed_ratio:.1f}x")

    def _generate_verdict(self, success_percentage: float) -> Dict[str, Any]:
        """Generate final verdict based on success percentage.
        
        Args:
            success_percentage: Overall success percentage
            
        Returns:
            Dictionary containing verdict, message, and next steps
        """
        if success_percentage >= 80:
            return {
                "verdict": "🎉 EXCELLENT",
                "message": "All models working perfectly! Ready for production benchmarks!",
                "next_steps": [
                    "Run full benchmarks on your datasets",
                    "Scale to larger structures",
                    "Deploy for real projects",
                ]
            }
        elif success_percentage >= 60:
            return {
                "verdict": "👍 GOOD",
                "message": "Most functionality working. Minor issues remain.",
                "next_steps": [
                    "Fix remaining model issues",
                    "Test with your specific datasets",
                    "Consider production deployment",
                ]
            }
        elif success_percentage >= 40:
            return {
                "verdict": "⚠️ NEEDS WORK",
                "message": "Partial functionality. Some models need fixing.",
                "next_steps": [
                    "Focus on critical model fixes",
                    "Review MACE patches",
                    "Test one model at a time",
                ]
            }
        else:
            return {
                "verdict": "❌ CRITICAL ISSUES",
                "message": "Major problems detected. Significant work needed.",
                "next_steps": [
                    "Review all installations",
                    "Check dependencies",
                    "Contact support",
                ]
            }

    def _print_mace_status(self):
        """Print MACE-specific status information."""
        if "mace" not in self.results.get("model_tests", {}):
            return

        mace_score = self.results["model_tests"]["mace"].get("success_score", 0)
        
        if mace_score >= 0.75:
            print("🎉 MACE is working EXCELLENTLY! Your patches are successful!")
        elif mace_score >= 0.5:
            print("👍 MACE is working WELL! Patches partially successful.")
        elif mace_score >= 0.25:
            print("⚠️ MACE has PARTIAL functionality. Patches need refinement.")
        else:
            print("❌ MACE is NOT WORKING. Patches need major fixes.")

    def _save_results(self, success_percentage: float, verdict: str, 
                     working_models: List[str], total_models: int):
        """Save detailed results to JSON file.
        
        Args:
            success_percentage: Overall success percentage
            verdict: Final verdict string
            working_models: List of working model names
            total_models: Total number of models tested
        """
        try:
            # Prepare results for JSON serialization
            json_results = self.convert_for_json(self.results)
            json_results["test_metadata"] = {
                "timestamp": datetime.now().isoformat(),
                "test_type": "model_focused_registry",
                "num_structures": len(self.test_structures),
                "success_percentage": success_percentage,
                "verdict": verdict,
                "working_models": working_models,
                "total_models_tested": total_models,
            }

            with open("model_focused_test_results.json", "w", encoding="utf-8") as file:
                json.dump(json_results, file, indent=2)

            print("\n💾 Detailed results saved to: model_focused_test_results.json")

        except Exception as exc:
            print(f"⚠️ Could not save results to file: {exc}")

    def convert_for_json(self, obj: Any) -> Any:
        """Convert numpy types and other non-serializable objects for JSON.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON-serializable object
        """
        if isinstance(obj, dict):
            return {k: self.convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_for_json(item) for item in obj]
        elif isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.integer)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.floating)):
            return float(obj)
        elif hasattr(obj, "__dict__"):
            return str(obj)  # Convert complex objects to string
        else:
            return obj

    def run_model_focused_test(self) -> bool:
        """Run the complete model-focused test suite.
        
        Returns:
            True if tests were successful, False otherwise
        """
        print("🚀 STARTING MODEL-FOCUSED REGISTRY TESTING")
        print("=" * 70)
        print("This tests ALL models through the ForgeBench registry system")
        print("and focuses on model-dependent benchmarks with ALL working models.")
        print("=" * 70)

        start_time = time.time()

        try:
            # Core registry tests
            self.test_registry_discovery()
            self.test_all_models_through_registry()

            # Model-dependent benchmark tests
            self.test_model_dependent_benchmarks()

            # Analysis
            self.cross_model_analysis()

            # Final report
            success = self.generate_final_report()

            total_time = time.time() - start_time
            print(f"\n⏱️ Total testing time: {total_time:.1f} seconds")

            return success

        except KeyboardInterrupt:
            print("\n⚠️ Testing interrupted by user")
            return False
        except Exception as exc:
            print(f"\n❌ Unexpected error during testing: {exc}")
            traceback.print_exc()
            return False


def main() -> bool:
    """Main function to run model-focused registry testing.
    
    Returns:
        True if tests were successful, False otherwise
    """
    # Set environment variables for better compatibility
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    print("🧪 ForgeBench Model-Focused Registry Testing")
    print("=" * 70)
    print("This script tests ALL models through the registry system")
    print("and focuses on model-dependent benchmarks with ALL working models.")
    print("No validity/distribution benchmarks - only model-dependent ones!")
    print("=" * 70)

    tester = ModelFocusedRegistryTester()
    success = tester.run_model_focused_test()

    if success:
        print("\n🎉 SUCCESS! Models are working well!")
    else:
        print("\n⚠️ Issues detected. Review the report above.")

    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)