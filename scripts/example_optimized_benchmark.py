#!/usr/bin/env python3
"""Example script demonstrating optimized benchmark execution.

This script shows how to use the new parallelization optimizations
in the run_benchmarks.py script for improved performance.
"""

from pathlib import Path


def run_optimized_benchmark_example():
    """Run an example benchmark with optimized settings."""
    
    print("🚀 MLIP Parallelization Optimization Example")
    print("=" * 50)
    
    # Example 1: Default optimized settings
    print("\n1️⃣ Example 1: Default Optimized Settings")
    print("   (Parallel MLIPs enabled, 3 MLIP workers, 4 structure processes)")
    
    example_command = [
        "python", "scripts/run_benchmarks.py",
        "--cifs", "examples/test_structures.txt",  # You'll need to create this
        "--config", "validity",
        "--name", "optimized_example_1",
        "--monitor-memory"
    ]
    
    print(f"Command: {' '.join(example_command)}")
    print("Note: This uses all optimizations by default")
    
    # Example 2: Custom optimization settings
    print("\n2️⃣ Example 2: Custom Optimization Settings")
    print("   (6 MLIP workers, 8 structure processes for high-performance systems)")
    
    high_performance_command = [
        "python", "scripts/run_benchmarks.py",
        "--cifs", "examples/test_structures.txt",
        "--config", "multi_mlip_stability",
        "--name", "high_performance_example",
        "--max-mlip-workers", "6",
        "--n-jobs", "8",
        "--monitor-memory"
    ]
    
    print(f"Command: {' '.join(high_performance_command)}")
    print("Note: Use this for systems with many CPU cores")
    
    # Example 3: Memory-constrained settings
    print("\n3️⃣ Example 3: Memory-Constrained Settings")
    print("   (Reduced workers to minimize memory usage)")
    
    memory_constrained_command = [
        "python", "scripts/run_benchmarks.py",
        "--cifs", "examples/test_structures.txt",
        "--config", "distribution",
        "--name", "memory_constrained_example",
        "--max-mlip-workers", "2",
        "--n-jobs", "2",
        "--batch-size", "10",
        "--monitor-memory"
    ]
    
    print(f"Command: {' '.join(memory_constrained_command)}")
    print("Note: Use this for systems with limited RAM")
    
    # Example 4: Disable optimizations (for comparison)
    print("\n4️⃣ Example 4: Disable Optimizations (Sequential Processing)")
    print("   (For performance comparison)")
    
    sequential_command = [
        "python", "scripts/run_benchmarks.py",
        "--cifs", "examples/test_structures.txt",
        "--config", "validity",
        "--name", "sequential_example",
        "--no-parallel-mlips",
        "--n-jobs", "1",
        "--monitor-memory"
    ]
    
    print(f"Command: {' '.join(sequential_command)}")
    print("Note: Use this to compare with optimized performance")
    
    # Example 5: Using optimized configuration file
    print("\n5️⃣ Example 5: Using Optimized Configuration File")
    print("   (Using the new optimized config)")
    
    optimized_config_command = [
        "python", "scripts/run_benchmarks.py",
        "--cifs", "examples/test_structures.txt",
        "--config", "multi_mlip_stability_optimized",
        "--name", "optimized_config_example",
        "--monitor-memory"
    ]
    
    print(f"Command: {' '.join(optimized_config_command)}")
    print("Note: Uses the optimized configuration file")
    
    print("\n" + "=" * 50)
    print("📋 Performance Tips:")
    print("   • Use --max-mlip-workers=3 for typical systems")
    print("   • Use --n-jobs=4 for balanced performance")
    print("   • Use --batch-size for large datasets")
    print("   • Use --monitor-memory to track resource usage")
    print("   • Expected speedup: 2-3x for typical configurations")
    
    print("\n🔧 Available Optimization Flags:")
    print("   --parallel-mlips          Enable parallel MLIP processing (default)")
    print("   --no-parallel-mlips       Disable parallel MLIP processing")
    print("   --max-mlip-workers N      Set MLIP worker count (default: 3)")
    print("   --n-jobs N                Set structure process count (default: 4)")
    print("   --batch-size N            Process in batches to reduce memory")
    print("   --monitor-memory          Enable detailed memory monitoring")
    
    print("\n📊 Expected Performance Improvements:")
    print("   • 3 MLIPs, 4 CPU cores: 2.5-3x speedup")
    print("   • 2 MLIPs, 4 CPU cores: 1.8-2.2x speedup")
    print("   • 3 MLIPs, 8 CPU cores: 2.8-3.5x speedup")
    
    print("\n⚠️  Troubleshooting:")
    print("   • Memory issues: Reduce --max-mlip-workers and --n-jobs")
    print("   • Timeout errors: Increase timeout in config files")
    print("   • Slow performance: Ensure --parallel-mlips is enabled")


def create_test_structures_file():
    """Create a simple test structures file for the examples."""
    
    test_file = Path("examples/test_structures.txt")
    test_file.parent.mkdir(exist_ok=True)
    
    # Create some example CIF file paths (you'll need actual CIF files)
    example_cifs = [
        "examples/CoNi.cif",
        "examples/crystal_50.cif", 
        "examples/CsBr.cif",
        "examples/CsPbBr3.cif",
        "examples/NiO.cif"
    ]
    
    with open(test_file, 'w') as f:
        for cif_path in example_cifs:
            f.write(f"{cif_path}\n")
    
    print(f"✅ Created test structures file: {test_file}")
    print("   Note: You'll need to add actual CIF files to the examples/ directory")


if __name__ == "__main__":
    print("MLIP Parallelization Optimization Examples")
    print("=" * 50)
    
    # Create test structures file
    create_test_structures_file()
    
    # Show examples
    run_optimized_benchmark_example()
    
    print("\n🎯 To run an actual benchmark with optimizations:")
    print("   python scripts/run_benchmarks.py --cifs examples/test_structures.txt --config validity --name my_optimized_run")
    
    print("\n📈 To compare performance:")
    print("   1. Run with: --no-parallel-mlips (sequential)")
    print("   2. Run with: --parallel-mlips (optimized)")
    print("   3. Compare execution times")
