#!/usr/bin/env python3
"""
Model-focused comprehensive test script focusing on registry-based model testing.
This tests all models through the ForgeBench registry system and focuses on model-dependent benchmarks.
"""

import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from pymatgen.util.testing import PymatgenTest


class ModelFocusedRegistryTester:
    """Model-focused testing suite for registry-based model testing."""

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

            # Add a simple structure for stress testing
            from pymatgen.core import Lattice, Structure

            simple_structure = Structure(
                Lattice.cubic(4.0), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
            )
            self.test_structures.append(simple_structure)

            # Add CIF structures if available
            cif_files = ["CsBr.cif", "CsPbBr3.cif", "NiO.cif"]
            for cif_file in cif_files:
                if Path(cif_file).exists() or Path(f"notebooks/{cif_file}").exists():
                    try:
                        from pymatgen.core import Structure

                        file_path = (
                            cif_file
                            if Path(cif_file).exists()
                            else f"notebooks/{cif_file}"
                        )
                        structure = Structure.from_file(file_path)
                        structure = structure.remove_oxidation_states()
                        self.test_structures.append(structure)
                    except Exception as e:
                        print(f"⚠️ Could not load {cif_file}: {e}")

            print(f"✅ Setup {len(self.test_structures)} test structures")
            for i, struct in enumerate(self.test_structures):
                print(f"   {i + 1}. {struct.composition} ({len(struct)} atoms)")

        except Exception as e:
            print(f"⚠️ Error setting up structures: {e}")
            # Fallback: create simple structures manually
            from pymatgen.core import Lattice, Structure

            si_structure = Structure(
                Lattice.cubic(5.43), ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]]
            )
            self.test_structures = [si_structure]

    def print_header(self, title):
        """Print formatted section header."""
        print(f"\n{'=' * 70}")
        print(f"🧪 {title}")
        print(f"{'=' * 70}")

    def print_subheader(self, title):
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
                f"📋 Registry reports {len(available_models)} available models: {available_models}"
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
            except Exception as e:
                print(f"⚠️ Error getting model info: {e}")
                self.results["registry_discovery"] = {
                    "available_models": available_models,
                    "model_availability": model_availability,
                    "info_error": str(e),
                }

        except Exception as e:
            print(f"❌ Error in registry discovery: {e}")
            traceback.print_exc()
            self.results["registry_discovery"] = {"error": str(e)}

    def test_model_through_registry(self, model_name: str) -> Dict[str, Any]:
        """Comprehensive test of a single model through registry."""
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
            from lematerial_forgebench.models.registry import get_calculator

            # Step 1: Create calculator through registry
            print(f"   🔧 Creating {model_name.upper()} calculator...")

            try:
                # Use model-specific parameters
                if model_name == "mace":
                    # Test both approaches for MACE
                    print("      Trying MACE with default settings...")
                    try:
                        calc = get_calculator("mace", model_type="mp", device="cpu")
                        print("      ✅ MACE default settings worked")
                    except Exception as e1:
                        print(f"      ⚠️ MACE default failed: {e1}")
                        print("      Trying MACE with small model...")
                        calc = get_calculator(
                            "mace", model_type="mp", device="cpu", model_size="small"
                        )
                        print("      ✅ MACE small model worked")
                        model_results["warnings"].append(
                            "Required small model for MACE"
                        )

                elif model_name == "orb":
                    calc = get_calculator("orb", device="cpu")

                elif model_name == "uma":
                    calc = get_calculator("uma", task="omat", device="cpu")

                elif model_name == "equiformer":
                    calc = get_calculator("equiformer", device="cpu")

                else:
                    calc = get_calculator(model_name, device="cpu")

                model_results["calculator_created"] = True
                print(f"      ✅ {model_name.upper()} calculator created successfully")

            except Exception as e:
                print(f"      ❌ Calculator creation failed: {type(e).__name__}: {e}")
                model_results["errors"].append(f"Calculator creation: {str(e)}")
                return model_results

            # Step 2: Basic energy/force calculation
            print("   ⚡ Testing basic energy/force calculation...")

            try:
                structure = self.test_structures[0]  # Use simplest structure first

                start_time = time.time()
                result = calc.calculate_energy_forces(structure)
                calc_time = time.time() - start_time

                print(f"      ✅ Energy: {result.energy:.4f} eV")
                print(f"      ✅ Forces shape: {result.forces.shape}")
                print(f"      ✅ Calculation time: {calc_time:.3f}s")

                if result.stress is not None:
                    print(f"      ✅ Stress available: {result.stress.shape}")

                model_results["basic_calculation"] = True
                model_results["basic_energy"] = result.energy
                model_results["basic_calc_time"] = calc_time
                model_results["forces_shape"] = result.forces.shape

            except Exception as e:
                print(f"      ❌ Basic calculation failed: {type(e).__name__}: {e}")
                model_results["errors"].append(f"Basic calculation: {str(e)}")
                # Continue with other tests even if basic calc fails

            # Step 3: Test all structures
            if model_results["basic_calculation"]:
                print(f"   🧮 Testing all {len(self.test_structures)} structures...")

                successful_calcs = 0
                structure_results = []

                for i, structure in enumerate(self.test_structures):
                    try:
                        print(
                            f"      Structure {i + 1}: {structure.composition} ({len(structure)} atoms)"
                        )

                        start_time = time.time()
                        result = calc.calculate_energy_forces(structure)
                        calc_time = time.time() - start_time

                        successful_calcs += 1
                        structure_results.append(
                            {
                                "composition": str(structure.composition),
                                "n_atoms": len(structure),
                                "energy": result.energy,
                                "calc_time": calc_time,
                                "energy_per_atom": result.energy / len(structure),
                            }
                        )

                        print(
                            f"         ✅ E = {result.energy:.3f} eV ({result.energy / len(structure):.3f} eV/atom)"
                        )

                    except Exception as e:
                        print(f"         ❌ Failed: {e}")
                        model_results["warnings"].append(
                            f"Structure {i + 1} failed: {str(e)}"
                        )

                model_results["all_structures_tested"] = successful_calcs == len(
                    self.test_structures
                )
                model_results["successful_structures"] = successful_calcs
                model_results["structure_results"] = structure_results

                print(
                    f"      📊 Success rate: {successful_calcs}/{len(self.test_structures)} structures"
                )

            # Step 4: Test embeddings
            print("   🧬 Testing embedding extraction...")

            try:
                structure = self.test_structures[0]
                embedding_result = calc.extract_embeddings(structure)

                if (
                    hasattr(embedding_result, "node_embeddings")
                    and embedding_result.node_embeddings is not None
                ):
                    embeddings = embedding_result.node_embeddings
                    print(f"      ✅ Node embeddings: {embeddings.shape}")
                    model_results["node_embeddings_shape"] = embeddings.shape

                if (
                    hasattr(embedding_result, "graph_embeddings")
                    and embedding_result.graph_embeddings is not None
                ):
                    graph_emb = embedding_result.graph_embeddings
                    print(f"      ✅ Graph embeddings: {graph_emb.shape}")
                    model_results["graph_embeddings_shape"] = graph_emb.shape

                if (
                    hasattr(embedding_result, "embeddings")
                    and embedding_result.embeddings is not None
                ):
                    embeddings = embedding_result.embeddings
                    if hasattr(embeddings, "shape"):
                        print(f"      ✅ Embeddings: {embeddings.shape}")
                        model_results["embeddings_shape"] = embeddings.shape
                    else:
                        print(
                            f"      ✅ Embeddings extracted (type: {type(embeddings)})"
                        )

                model_results["embeddings_extracted"] = True

            except Exception as e:
                print(f"      ❌ Embeddings failed: {type(e).__name__}: {e}")
                model_results["errors"].append(f"Embeddings: {str(e)}")

            # Step 5: Test formation energy
            print("   🔥 Testing formation energy calculation...")

            try:
                structure = self.test_structures[0]
                formation_energy = calc.calculate_formation_energy(structure)

                print(f"      ✅ Formation energy: {formation_energy:.4f} eV/atom")
                model_results["formation_energy_calculated"] = True
                model_results["formation_energy"] = formation_energy

            except Exception as e:
                print(f"      ❌ Formation energy failed: {type(e).__name__}: {e}")
                model_results["errors"].append(f"Formation energy: {str(e)}")

            # Step 6: Test energy above hull
            print("   ⛰️  Testing energy above hull calculation...")

            try:
                structure = self.test_structures[0]
                e_above_hull = calc.calculate_energy_above_hull(structure)

                print(f"      ✅ Energy above hull: {e_above_hull:.4f} eV/atom")
                model_results["energy_above_hull_calculated"] = True
                model_results["energy_above_hull"] = e_above_hull

            except Exception as e:
                print(f"      ❌ Energy above hull failed: {type(e).__name__}: {e}")
                model_results["errors"].append(f"Energy above hull: {str(e)}")

            # Step 7: Performance testing
            if model_results["basic_calculation"]:
                print("   ⚡ Performance testing...")

                try:
                    structure = self.test_structures[0]  # Use smallest structure
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

                    print(f"      ✅ Average time: {avg_time:.3f}±{std_time:.3f}s")
                    print(f"      ✅ Throughput: {throughput:.1f} atoms/sec")

                    model_results["performance_tested"] = True
                    model_results["avg_calc_time"] = avg_time
                    model_results["std_calc_time"] = std_time
                    model_results["throughput"] = throughput

                except Exception as e:
                    print(f"      ❌ Performance test failed: {e}")
                    model_results["warnings"].append(f"Performance test: {str(e)}")

        except Exception as e:
            print(f"   ❌ Unexpected error in {model_name.upper()} testing: {e}")
            traceback.print_exc()
            model_results["errors"].append(f"Unexpected error: {str(e)}")

        # Calculate overall success score
        success_criteria = [
            model_results["calculator_created"],
            model_results["basic_calculation"],
            model_results.get("successful_structures", 0)
            >= len(self.test_structures) // 2,
            len(model_results["errors"]) <= 2,
        ]

        model_results["success_score"] = sum(success_criteria) / len(success_criteria)

        if model_results["success_score"] >= 0.75:
            print(
                f"   🎉 {model_name.upper()}: EXCELLENT (score: {model_results['success_score']:.2f})"
            )
        elif model_results["success_score"] >= 0.5:
            print(
                f"   👍 {model_name.upper()}: GOOD (score: {model_results['success_score']:.2f})"
            )
        elif model_results["success_score"] >= 0.25:
            print(
                f"   ⚠️ {model_name.upper()}: PARTIAL (score: {model_results['success_score']:.2f})"
            )
        else:
            print(
                f"   ❌ {model_name.upper()}: POOR (score: {model_results['success_score']:.2f})"
            )

        return model_results

    def test_all_models_through_registry(self):
        """Test all available models through registry."""
        self.print_header("COMPREHENSIVE REGISTRY-BASED MODEL TESTING")

        # Get available models
        try:
            from lematerial_forgebench.models.registry import list_available_models

            available_models = list_available_models()
        except Exception as e:
            print(f"❌ Could not get available models: {e}")
            available_models = ["orb", "mace", "uma"]  # Fallback

        print(f"🎯 Testing {len(available_models)} models: {available_models}")

        all_model_results = {}

        for model_name in available_models:
            try:
                model_result = self.test_model_through_registry(model_name)
                all_model_results[model_name] = model_result
            except Exception as e:
                print(f"❌ Failed to test {model_name}: {e}")
                all_model_results[model_name] = {
                    "model_name": model_name,
                    "fatal_error": str(e),
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
            f"🎯 Testing model-dependent benchmarks with {len(working_models)} working models: {working_models}"
        )

        if not working_models:
            print("❌ No working models found for benchmark testing")
            self.results["benchmarks"] = {"error": "No working models available"}
            return

        # Test Stability Benchmark with ALL working models
        for model_name in working_models:  # Test ALL models, not just [:1]
            print(f"\n⚖️  Testing Stability Benchmark with {model_name.upper()}...")
            try:
                from lematerial_forgebench.benchmarks.stability_benchmark import (
                    StabilityBenchmark,
                )

                # Create benchmark with the specific model
                stability_benchmark = StabilityBenchmark(model_name=model_name)
                result = stability_benchmark.evaluate(
                    self.test_structures[:2]
                )  # Use 2 structures

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

            except Exception as e:
                print(
                    f"   ❌ Stability benchmark with {model_name.upper()} failed: {e}"
                )
                benchmark_results[f"stability_{model_name}"] = {
                    "success": False,
                    "error": str(e),
                    "model_name": model_name,
                }

        self.results["benchmarks"] = benchmark_results

    def cross_model_analysis(self):
        """Perform cross-model analysis."""
        self.print_header("CROSS-MODEL ANALYSIS")

        if "model_tests" not in self.results:
            print("❌ No model test results available for analysis")
            return

        # Collect energies and performance data
        model_energies = {}
        model_performance = {}
        model_formation_energies = {}
        model_e_above_hull = {}

        structure_composition = str(self.test_structures[0].composition)

        for model_name, results in self.results["model_tests"].items():
            if results.get("basic_calculation") and "basic_energy" in results:
                model_energies[model_name] = results["basic_energy"]

            if results.get("formation_energy_calculated"):
                model_formation_energies[model_name] = results["formation_energy"]

            if results.get("energy_above_hull_calculated"):
                model_e_above_hull[model_name] = results["energy_above_hull"]

            if results.get("performance_tested"):
                model_performance[model_name] = {
                    "avg_time": results.get("avg_calc_time", 0),
                    "throughput": results.get("throughput", 0),
                }

        # Energy comparison
        if len(model_energies) >= 2:
            print(f"📊 Total Energy Comparison for {structure_composition}:")

            for model, energy in model_energies.items():
                energy_per_atom = energy / len(self.test_structures[0])
                print(
                    f"   {model.upper()}: {energy:.3f} eV ({energy_per_atom:.3f} eV/atom)"
                )

            # Statistics
            energies = list(model_energies.values())
            energy_range = max(energies) - min(energies)
            energy_mean = np.mean(energies)
            energy_std = np.std(energies)

            print("\n📈 Total Energy Statistics:")
            print(f"   Mean: {energy_mean:.3f} eV")
            print(f"   Std Dev: {energy_std:.3f} eV")
            print(f"   Range: {energy_range:.3f} eV")

            # Agreement assessment
            per_atom_range = energy_range / len(self.test_structures[0])
            if per_atom_range < 0.1:
                print("✅ Excellent model agreement (range < 0.1 eV/atom)")
            elif per_atom_range < 0.5:
                print("👍 Good model agreement (range < 0.5 eV/atom)")
            elif per_atom_range < 1.0:
                print("⚠️ Moderate model agreement (range < 1.0 eV/atom)")
            else:
                print("❌ Poor model agreement (range > 1.0 eV/atom)")

        # Formation energy comparison
        if len(model_formation_energies) >= 2:
            print(f"\n🔥 Formation Energy Comparison for {structure_composition}:")

            for model, form_energy in model_formation_energies.items():
                print(f"   {model.upper()}: {form_energy:.4f} eV/atom")

            # Statistics
            form_energies = list(model_formation_energies.values())
            form_energy_range = max(form_energies) - min(form_energies)
            form_energy_mean = np.mean(form_energies)

            print("\n📈 Formation Energy Statistics:")
            print(f"   Mean: {form_energy_mean:.4f} eV/atom")
            print(f"   Range: {form_energy_range:.4f} eV/atom")

        # Energy above hull comparison
        if len(model_e_above_hull) >= 2:
            print(f"\n⛰️  Energy Above Hull Comparison for {structure_composition}:")

            for model, e_hull in model_e_above_hull.items():
                print(f"   {model.upper()}: {e_hull:.4f} eV/atom")

            # Statistics
            e_hull_values = list(model_e_above_hull.values())
            e_hull_range = max(e_hull_values) - min(e_hull_values)
            e_hull_mean = np.mean(e_hull_values)

            print("\n📈 Energy Above Hull Statistics:")
            print(f"   Mean: {e_hull_mean:.4f} eV/atom")
            print(f"   Range: {e_hull_range:.4f} eV/atom")

        # Performance comparison
        if len(model_performance) >= 2:
            print("\n⚡ Performance Comparison:")

            for model, perf in model_performance.items():
                print(
                    f"   {model.upper()}: {perf['avg_time']:.3f}s ({perf['throughput']:.1f} atoms/s)"
                )

            # Find fastest
            fastest_model = min(
                model_performance.items(), key=lambda x: x[1]["avg_time"]
            )
            print(f"🏆 Fastest model: {fastest_model[0].upper()}")

        # Benchmark comparison
        if "benchmarks" in self.results:
            print("\n📊 Model-Dependent Benchmark Comparison:")

            # Group benchmarks by type
            stability_results = {}
            novelty_results = {}

            for bench_name, bench_result in self.results["benchmarks"].items():
                if bench_name.startswith("stability_") and bench_result.get("success"):
                    model_name = bench_result.get(
                        "model_name", bench_name.split("_")[1]
                    )
                    stability_results[model_name] = bench_result["all_scores"]

            if stability_results:
                print(f"\n   ⚖️  Stability Benchmark Results:")
                for model, scores in stability_results.items():
                    key_metric = scores.get(
                        "stable_ratio", scores.get("metastable_ratio", "N/A")
                    )
                    print(f"      {model.upper()}: {key_metric}")

        # Store comparison results
        self.results["cross_model_analysis"] = {
            "total_energies": model_energies,
            "formation_energies": model_formation_energies,
            "energy_above_hull": model_e_above_hull,
            "performance": model_performance,
        }

    def generate_final_report(self):
        """Generate final comprehensive report."""
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
                f"   Calculator: {calc_created} | Basic Calc: {basic_calc} | Embeddings: {embeddings}"
            )
            print(f"   Formation E: {formation_e} | E Above Hull: {e_hull}")

            if results.get("errors"):
                print(
                    f"   Errors: {len(results['errors'])} - {results['errors'][0] if results['errors'] else 'None'}"
                )

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
            f"Fully working: {fully_working} ({fully_working / total_models * 100:.1f}%)"
        )
        print(
            f"Partially working: {partially_working_count} ({partially_working_count / total_models * 100:.1f}%)"
        )
        print(f"Broken: {broken_count} ({broken_count / total_models * 100:.1f}%)")

        # Model-dependent benchmark results
        if "benchmarks" in self.results:
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
                f"⚖️  Stability Benchmark: {stability_success}/{stability_total} models successful"
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
                            f"      {model_name.upper()}: Stable ratio: {stable_ratio}, Mean E above hull: {mean_e_hull}"
                        )

        # Cross-model comparison summary
        if "cross_model_analysis" in self.results:
            comp_data = self.results["cross_model_analysis"]

            print("\n📊 CROSS-MODEL COMPARISON SUMMARY")
            print("-" * 60)

            if (
                comp_data.get("total_energies")
                and len(comp_data["total_energies"]) >= 2
            ):
                energies = list(comp_data["total_energies"].values())
                energy_range = max(energies) - min(energies)
                per_atom_range = energy_range / len(self.test_structures[0])

                if per_atom_range < 0.1:
                    agreement = "✅ EXCELLENT"
                elif per_atom_range < 0.5:
                    agreement = "👍 GOOD"
                elif per_atom_range < 1.0:
                    agreement = "⚠️ MODERATE"
                else:
                    agreement = "❌ POOR"

                print(
                    f"Energy Agreement: {agreement} ({per_atom_range:.3f} eV/atom range)"
                )

            if comp_data.get("performance") and len(comp_data["performance"]) >= 2:
                perf_data = comp_data["performance"]
                fastest = min(perf_data.items(), key=lambda x: x[1]["avg_time"])
                slowest = max(perf_data.items(), key=lambda x: x[1]["avg_time"])
                speed_ratio = slowest[1]["avg_time"] / fastest[1]["avg_time"]

                print(
                    f"Performance: {fastest[0].upper()} fastest ({fastest[1]['avg_time']:.3f}s)"
                )
                print(
                    f"           {slowest[0].upper()} slowest ({slowest[1]['avg_time']:.3f}s)"
                )
                print(f"           Speed ratio: {speed_ratio:.1f}x")

        # Issues and recommendations
        print("\n💡 ISSUES AND RECOMMENDATIONS")
        print("-" * 60)

        critical_issues = []
        recommendations = []

        # Analyze issues from model tests
        for model_name, results in self.results["model_tests"].items():
            if results.get("success_score", 0) < 0.5:
                if not results.get("calculator_created", False):
                    critical_issues.append(
                        f"{model_name.upper()}: Cannot create calculator"
                    )
                elif not results.get("basic_calculation", False):
                    critical_issues.append(
                        f"{model_name.upper()}: Basic calculations fail"
                    )

                if results.get("errors"):
                    for error in results["errors"][:1]:  # Show first error
                        critical_issues.append(f"{model_name.upper()}: {error}")

        # Display issues
        if critical_issues:
            print("🚨 CRITICAL ISSUES:")
            for issue in critical_issues[:5]:  # Show top 5
                print(f"   - {issue}")

        # Generate recommendations
        if fully_working == 0:
            recommendations.append(
                "🛠️ URGENT: No models are fully working - check dependencies"
            )
            recommendations.append("📞 Consider reaching out for technical support")
        elif fully_working < 2:
            recommendations.append("🔧 Focus on fixing remaining models for redundancy")
            recommendations.append("📚 Review installation documentation")
        else:
            recommendations.append(
                "✅ Good model coverage - ready for production benchmarks"
            )
            recommendations.append("🚀 Consider testing with larger datasets")

        # Model-specific recommendations
        mace_results = self.results["model_tests"].get("mace", {})
        if mace_results.get("success_score", 0) < 0.5:
            recommendations.append("🔧 MACE: Apply monkey patches and try small model")
            recommendations.append("🔧 MACE: Clear cache: rm -rf ~/.cache/mace/")

        uma_results = self.results["model_tests"].get("uma", {})
        if uma_results.get("success_score", 0) < 0.5:
            recommendations.append(
                "🔧 UMA: Update fairchem: uv add --upgrade fairchem-core"
            )

        orb_results = self.results["model_tests"].get("orb", {})
        if orb_results.get("success_score", 0) < 0.5:
            recommendations.append("🔧 ORB: Install with: uv sync --extra orb")

        if recommendations:
            print("\n📋 RECOMMENDATIONS:")
            for rec in recommendations:
                print(f"   {rec}")

        # Success criteria assessment
        print("\n🎯 SUCCESS CRITERIA ASSESSMENT")
        print("-" * 60)

        stability_success = sum(
            1
            for k, v in self.results.get("benchmarks", {}).items()
            if k.startswith("stability_") and v.get("success", False)
        )

        criteria = {
            "At least 1 model fully working": fully_working >= 1,
            "At least 2 models working": (fully_working + partially_working_count) >= 2,
            "Basic calculations functional": sum(
                1
                for r in self.results["model_tests"].values()
                if r.get("basic_calculation", False)
            )
            >= 1,
            "Formation energy calculation": sum(
                1
                for r in self.results["model_tests"].values()
                if r.get("formation_energy_calculated", False)
            )
            >= 1,
            "Energy above hull calculation": sum(
                1
                for r in self.results["model_tests"].values()
                if r.get("energy_above_hull_calculated", False)
            )
            >= 1,
            "Model-dependent benchmarks working": (stability_success)
            > 0,
            "Cross-model comparison possible": len(working_models) >= 2,
        }

        passed_criteria = 0
        for criterion, met in criteria.items():
            status = "✅" if met else "❌"
            if met:
                passed_criteria += 1
            print(f"{status} {criterion}")

        success_percentage = passed_criteria / len(criteria) * 100

        # Final verdict
        print("\n🏆 FINAL VERDICT")
        print("=" * 60)

        if success_percentage >= 80:
            verdict = "🎉 EXCELLENT"
            message = "All models working perfectly! Ready for production benchmarks!"
            next_steps = [
                "Run full benchmarks on your datasets",
                "Scale to larger structures",
                "Deploy for real projects",
            ]
        elif success_percentage >= 60:
            verdict = "👍 GOOD"
            message = "Most functionality working. Minor issues remain."
            next_steps = [
                "Fix remaining model issues",
                "Test with your specific datasets",
                "Consider production deployment",
            ]
        elif success_percentage >= 40:
            verdict = "⚠️ NEEDS WORK"
            message = "Partial functionality. Some models need fixing."
            next_steps = [
                "Focus on critical model fixes",
                "Review MACE patches",
                "Test one model at a time",
            ]
        else:
            verdict = "❌ CRITICAL ISSUES"
            message = "Major problems detected. Significant work needed."
            next_steps = [
                "Review all installations",
                "Check dependencies",
                "Contact support",
            ]

        print(f"{verdict}: {success_percentage:.1f}% success rate")
        print(f"📝 {message}")

        print("\n📋 RECOMMENDED NEXT STEPS:")
        for i, step in enumerate(next_steps, 1):
            print(f"   {i}. {step}")

        # Highlight MACE status specifically
        if "mace" in self.results["model_tests"]:
            mace_score = self.results["model_tests"]["mace"].get("success_score", 0)
            print("\n🎯 MACE STATUS:")
            if mace_score >= 0.75:
                print(
                    "   🎉 MACE is working EXCELLENTLY! Your patches are successful!"
                )
            elif mace_score >= 0.5:
                print("   👍 MACE is working WELL! Patches partially successful.")
            elif mace_score >= 0.25:
                print("   ⚠️ MACE has PARTIAL functionality. Patches need refinement.")
            else:
                print("   ❌ MACE is NOT WORKING. Patches need major fixes.")

        # Save detailed results
        try:
            import json
            from datetime import datetime

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

            with open("model_focused_test_results.json", "w") as f:
                json.dump(json_results, f, indent=2)

            print(f"\n💾 Detailed results saved to: model_focused_test_results.json")

        except Exception as e:
            print(f"⚠️ Could not save results to file: {e}")

        return success_percentage >= 60  # Return True if at least 60% success

    def convert_for_json(self, obj):
        """Convert numpy types and other non-serializable objects for JSON."""
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

    def run_model_focused_test(self):
        """Run the complete model-focused test suite."""
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
        except Exception as e:
            print(f"\n❌ Unexpected error during testing: {e}")
            traceback.print_exc()
            return False


def main():
    """Main function to run model-focused registry testing."""

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
