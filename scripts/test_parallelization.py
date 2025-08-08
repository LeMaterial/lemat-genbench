#!/usr/bin/env python3
"""Test script to demonstrate MLIP parallelization improvements."""

import logging
import time
from pathlib import Path

import numpy as np
from pymatgen.core import Structure

from lemat_genbench.preprocess.multi_mlip_preprocess import (
    MultiMLIPStabilityPreprocessor,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_structures(n_structures: int = 10) -> list[Structure]:
    """Create test structures for benchmarking."""
    structures = []
    
    # Create simple cubic structures with different compositions
    compositions = [
        ("Fe", "Fe"),
        ("Ni", "Ni"), 
        ("Cu", "Cu"),
        ("Fe", "Ni"),
        ("Ni", "Cu"),
        ("Fe", "Cu"),
        ("Fe", "Ni", "Cu"),
        ("Al", "Al"),
        ("Si", "Si"),
        ("Al", "Si"),
    ]
    
    for i in range(n_structures):
        comp = compositions[i % len(compositions)]
        # Create a simple cubic structure
        lattice = np.array([[3.0, 0, 0], [0, 3.0, 0], [0, 0, 3.0]])
        coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
        
        # Create structure with random small perturbations
        coords = np.array(coords) + np.random.normal(0, 0.1, (len(coords), 3))
        
        structure = Structure(
            lattice=lattice,
            species=comp,
            coords=coords,
            coords_are_cartesian=False
        )
        structures.append(structure)
    
    return structures


def benchmark_parallelization():
    """Benchmark sequential vs parallel MLIP processing."""
    
    # Create test structures
    n_structures = 8
    structures = create_test_structures(n_structures)
    logger.info(f"Created {len(structures)} test structures")
    
    # Test configurations
    configs = [
        {
            "name": "Sequential MLIPs",
            "parallel_mlips": False,
            "n_jobs": 4,
        },
        {
            "name": "Parallel MLIPs (3 workers)",
            "parallel_mlips": True,
            "max_mlip_workers": 3,
            "n_jobs": 4,
        },
        {
            "name": "Parallel MLIPs (2 workers)", 
            "parallel_mlips": True,
            "max_mlip_workers": 2,
            "n_jobs": 4,
        },
    ]
    
    results = {}
    
    for config in configs:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing: {config['name']}")
        logger.info(f"{'='*50}")
        
        # Create preprocessor
        preprocessor = MultiMLIPStabilityPreprocessor(
            mlip_names=["orb", "mace", "uma"],
            relax_structures=False,  # Skip relaxation for faster testing
            calculate_formation_energy=True,
            calculate_energy_above_hull=False,  # Skip for speed
            extract_embeddings=False,  # Skip for speed
            timeout=60,
            parallel_mlips=config.get("parallel_mlips", True),
            max_mlip_workers=config.get("max_mlip_workers", 3),
            n_jobs=config["n_jobs"],
        )
        
        # Time the processing
        start_time = time.time()
        
        try:
            result = preprocessor(structures)
            end_time = time.time()
            
            processing_time = end_time - start_time
            successful_structures = len(result.processed_structures)
            
            logger.info(f"✅ Processing completed in {processing_time:.2f} seconds")
            logger.info(f"   Successful structures: {successful_structures}/{n_structures}")
            logger.info(f"   Failed structures: {len(result.failed_indices)}")
            logger.info(f"   Average time per structure: {processing_time/n_structures:.2f}s")
            
            if result.warnings:
                logger.info(f"   Warnings: {len(result.warnings)}")
                for warning in result.warnings[:3]:  # Show first 3 warnings
                    logger.info(f"     - {warning}")
            
            results[config["name"]] = {
                "time": processing_time,
                "successful": successful_structures,
                "failed": len(result.failed_indices),
                "avg_time_per_structure": processing_time/n_structures,
            }
            
        except Exception as e:
            logger.error(f"❌ Processing failed: {str(e)}")
            results[config["name"]] = {"error": str(e)}
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("PERFORMANCE SUMMARY")
    logger.info(f"{'='*60}")
    
    for name, result in results.items():
        if "error" in result:
            logger.info(f"{name:30s} | ERROR: {result['error']}")
        else:
            logger.info(f"{name:30s} | {result['time']:6.2f}s | {result['successful']:2d}/{n_structures} | {result['avg_time_per_structure']:5.2f}s/structure")
    
    # Calculate speedup
    if "Sequential MLIPs" in results and "Parallel MLIPs (3 workers)" in results:
        if "error" not in results["Sequential MLIPs"] and "error" not in results["Parallel MLIPs (3 workers)"]:
            sequential_time = results["Sequential MLIPs"]["time"]
            parallel_time = results["Parallel MLIPs (3 workers)"]["time"]
            speedup = sequential_time / parallel_time
            logger.info(f"\nSpeedup (Parallel vs Sequential): {speedup:.2f}x")
            
            if speedup > 1.5:
                logger.info("🎉 Significant speedup achieved!")
            elif speedup > 1.1:
                logger.info("✅ Moderate speedup achieved")
            else:
                logger.info("⚠️  Minimal speedup - may need tuning")


def test_memory_usage():
    """Test memory usage patterns."""
    logger.info(f"\n{'='*50}")
    logger.info("MEMORY USAGE TEST")
    logger.info(f"{'='*50}")
    
    import os

    import psutil
    
    process = psutil.Process(os.getpid())
    
    # Create preprocessor
    preprocessor = MultiMLIPStabilityPreprocessor(
        mlip_names=["orb", "mace", "uma"],
        relax_structures=False,
        calculate_formation_energy=True,
        calculate_energy_above_hull=False,
        extract_embeddings=False,
        parallel_mlips=True,
        max_mlip_workers=3,
        n_jobs=2,  # Use fewer processes for memory testing
    )
    
    # Create a few test structures
    structures = create_test_structures(4)
    
    logger.info(f"Initial memory: {process.memory_info().rss / 1024 / 1024:.1f} MB")
    
    # Process structures
    start_time = time.time()
    result = preprocessor(structures)
    end_time = time.time()
    
    logger.info(f"Final memory: {process.memory_info().rss / 1024 / 1024:.1f} MB")
    logger.info(f"Processing time: {end_time - start_time:.2f}s")
    logger.info(f"Successful: {len(result.processed_structures)}/{len(structures)}")


if __name__ == "__main__":
    logger.info("Starting MLIP parallelization benchmark...")
    
    try:
        benchmark_parallelization()
        test_memory_usage()
        
        logger.info("\n✅ Benchmark completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {str(e)}")
        import traceback
        traceback.print_exc()
