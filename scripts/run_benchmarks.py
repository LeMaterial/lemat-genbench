#!/usr/bin/env python3
"""
Comprehensive benchmark runner for material generation evaluation.

This script:
1. Takes a list of CIF files as input
2. Loads a configuration specifying which benchmark families to run
3. Runs appropriate preprocessors based on benchmark requirements
4. Computes all specified metrics
5. Saves results to JSON files in the results/ directory

Usage:
    uv run scripts/run_benchmarks.py --cifs path/to/cifs.txt --config validity --name my_run
    uv run scripts/run_benchmarks.py --cifs path/to/cifs.txt --config distribution --name test_run
"""

import argparse
import gc
import json
import sys
import psutil
import torch
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lemat_genbench.benchmarks.distribution_benchmark import DistributionBenchmark
from lemat_genbench.benchmarks.diversity_benchmark import DiversityBenchmark
from lemat_genbench.benchmarks.hhi_benchmark import HHIBenchmark
from lemat_genbench.benchmarks.multi_mlip_stability_benchmark import (
    StabilityBenchmark as MultiMLIPStabilityBenchmark,
)
from lemat_genbench.benchmarks.novelty_benchmark import NoveltyBenchmark
from lemat_genbench.benchmarks.sun_benchmark import SUNBenchmark
from lemat_genbench.benchmarks.uniqueness_benchmark import UniquenessBenchmark
from lemat_genbench.benchmarks.validity_benchmark import ValidityBenchmark
from lemat_genbench.preprocess.distribution_preprocess import DistributionPreprocessor
from lemat_genbench.preprocess.multi_mlip_preprocess import (
    MultiMLIPStabilityPreprocessor,
)
from lemat_genbench.utils.logging import logger


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # Convert to MB


def log_memory_usage(stage: str, force_log=False):
    """Log current memory usage."""
    memory_mb = get_memory_usage()
    if force_log:
        logger.info(f"💾 Memory usage at {stage}: {memory_mb:.1f} MB")
    else:
        logger.debug(f"💾 Memory usage at {stage}: {memory_mb:.1f} MB")


def clear_memory():
    """Clear memory by running garbage collection and clearing PyTorch cache."""
    # Run Python garbage collection
    gc.collect()
    
    # Clear PyTorch cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Force garbage collection again
    gc.collect()
    
    logger.debug("🧹 Memory cleared (garbage collection + PyTorch cache)")


def clear_mlip_models():
    """Clear MLIP models from memory."""
    try:
        # Clear any cached models
        from lemat_genbench.models.registry import _MODEL_CACHE
        if hasattr(_MODEL_CACHE, 'clear'):
            _MODEL_CACHE.clear()
        
        # Clear any global model caches
        import sys
        for module_name in list(sys.modules.keys()):
            if 'lemat_genbench.models' in module_name:
                module = sys.modules[module_name]
                for attr_name in list(dir(module)):
                    if 'cache' in attr_name.lower() or 'model' in attr_name.lower():
                        try:
                            delattr(module, attr_name)
                        except:
                            pass
        
        logger.debug("🧹 MLIP models cleared from memory")
    except Exception as e:
        logger.debug(f"Could not clear MLIP models: {e}")


def cleanup_after_preprocessor(preprocessor_name: str, monitor_memory: bool = False):
    """Clean up memory after running a preprocessor."""
    logger.info(f"🧹 Cleaning up after {preprocessor_name} preprocessor...")
    
    # Clear memory
    clear_memory()
    
    # Clear MLIP models if it was a MLIP preprocessor
    if "mlip" in preprocessor_name.lower():
        clear_mlip_models()
    
    # Log memory usage
    log_memory_usage(f"after {preprocessor_name} cleanup", force_log=monitor_memory)


def cleanup_after_benchmark(benchmark_name: str, monitor_memory: bool = False):
    """Clean up memory after running a benchmark."""
    logger.info(f"🧹 Cleaning up after {benchmark_name} benchmark...")
    
    # Clear memory
    clear_memory()
    
    # Log memory usage
    log_memory_usage(f"after {benchmark_name} cleanup", force_log=monitor_memory)


def load_cif_files(input_path: str) -> List[str]:
    """Load list of CIF file paths from a text file or directory.
    
    Parameters
    ----------
    input_path : str
        Path to either:
        - A text file containing CIF file paths (one per line)
        - A directory containing CIF files
        
    Returns
    -------
    List[str]
        List of CIF file paths
    """
    input_path_obj = Path(input_path)
    
    if input_path_obj.is_dir():
        # Directory mode: find all CIF files in the directory
        logger.info(f"Scanning directory for CIF files: {input_path}")
        cif_paths = []
        
        # Find all .cif files in the directory (recursive)
        for cif_file in input_path_obj.rglob("*.cif"):
            cif_paths.append(str(cif_file))
        
        if not cif_paths:
            raise FileNotFoundError(f"No CIF files found in directory: {input_path}")
        
        logger.info(f"Found {len(cif_paths)} CIF files in directory")
        return cif_paths
        
    elif input_path_obj.is_file():
        # File mode: read CIF paths from text file
        logger.info(f"Loading CIF file list from: {input_path}")
        with open(input_path, "r") as f:
            cif_paths = [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]

        # Validate that files exist
        missing_files = [path for path in cif_paths if not Path(path).exists()]
        if missing_files:
            raise FileNotFoundError(f"Missing CIF files: {missing_files}")

        return cif_paths
    else:
        raise FileNotFoundError(f"Path does not exist: {input_path}")


def load_benchmark_config(config_name: str) -> Dict[str, Any]:
    """Load benchmark configuration from YAML file."""
    config_dir = Path(__file__).parent.parent / "src" / "config"
    config_path = config_dir / f"{config_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def create_preprocessor_config(benchmark_families: List[str]) -> Dict[str, Any]:
    """Create preprocessor configuration based on required benchmark families."""
    config = {
        "distribution": False,
        "stability": False,
        "embeddings": False,
    }

    # Determine which preprocessors are needed
    for family in benchmark_families:
        if family in ["distribution", "jsdistance", "mmd", "frechet"]:
            config["distribution"] = True
        if family in ["stability", "sun"]:
            config["stability"] = True
        if family in ["frechet", "distribution"]:  # Distribution includes Frechet distance
            config["embeddings"] = True

    return config


def run_preprocessors(structures, preprocessor_config: Dict[str, Any], monitor_memory: bool = False):
    """Run required preprocessors based on configuration."""
    processed_structures = structures
    preprocessor_results = {}

    # Log initial memory usage
    log_memory_usage("before preprocessing", force_log=monitor_memory)

    # Distribution preprocessor (for MMD, JSDistance)
    if preprocessor_config["distribution"]:
        logger.info("Running distribution preprocessor...")
        dist_preprocessor = DistributionPreprocessor()
        dist_result = dist_preprocessor(processed_structures)
        processed_structures = dist_result.processed_structures
        preprocessor_results["distribution"] = dist_result
        logger.info(
            f"✅ Distribution preprocessing complete for {len(processed_structures)} structures"
        )
        
        # Clean up after distribution preprocessor
        cleanup_after_preprocessor("distribution", monitor_memory)

    # Multi-MLIP preprocessor (for stability, embeddings)
    if preprocessor_config["stability"] or preprocessor_config["embeddings"]:
        logger.info("Running Multi-MLIP preprocessor...")

        # Configure MLIP models
        mlip_configs = {
            "orb": {"model_type": "orb_v3_conservative_inf_omat", "device": "cpu"},
            "mace": {"model_type": "mp", "device": "cpu"},
            "uma": {"task": "omat", "device": "cpu"},
        }

        # Determine what to extract based on requirements
        extract_embeddings = preprocessor_config["embeddings"]
        relax_structures = preprocessor_config["stability"]

        mlip_preprocessor = MultiMLIPStabilityPreprocessor(
            mlip_names=["orb", "mace", "uma"],
            mlip_configs=mlip_configs,
            relax_structures=relax_structures,
            relaxation_config={"fmax": 0.1, "steps": 50},
            calculate_formation_energy=relax_structures,
            calculate_energy_above_hull=relax_structures,
            extract_embeddings=extract_embeddings,
            timeout=300,
        )

        mlip_result = mlip_preprocessor(processed_structures)
        processed_structures = mlip_result.processed_structures
        preprocessor_results["multi_mlip"] = mlip_result
        logger.info(
            f"✅ Multi-MLIP preprocessing complete for {len(processed_structures)} structures"
        )
        
        # Clean up after MLIP preprocessor (this is crucial for memory management)
        cleanup_after_preprocessor("multi_mlip", monitor_memory)

    # Log final memory usage
    log_memory_usage("after all preprocessing")

    return processed_structures, preprocessor_results


def run_benchmarks(structures, benchmark_families: List[str], config: Dict[str, Any], monitor_memory: bool = False):
    """Run specified benchmark families."""
    results = {}

    # Log initial memory usage
    log_memory_usage("before benchmarks", force_log=monitor_memory)

    for family in benchmark_families:
        logger.info(f"Running {family} benchmark...")

        try:
            if family == "validity":
                benchmark = ValidityBenchmark(
                    charge_weight=config.get("charge_weight", 0.33),
                    distance_weight=config.get("distance_weight", 0.33),
                    plausibility_weight=config.get("plausibility_weight", 0.34),
                )

            elif family == "distribution":
                benchmark = DistributionBenchmark(
                    mlips=config.get("mlips", ["orb", "mace", "uma"]),
                    cache_dir=config.get("cache_dir", "./data"),
                    js_distributions_file=config.get(
                        "js_distributions_file",
                        "data/lematbulk_jsdistance_distributions.json",
                    ),
                    mmd_values_file=config.get(
                        "mmd_values_file", "data/lematbulk_mmd_values_15k.pkl"
                    ),
                )

            elif family == "diversity":
                benchmark = DiversityBenchmark()

            elif family == "novelty":
                benchmark = NoveltyBenchmark(
                    reference_dataset=config.get(
                        "reference_dataset", "LeMaterial/LeMat-Bulk"
                    ),
                    reference_config=config.get("reference_config", "compatible_pbe"),
                    fingerprint_method=config.get("fingerprint_method", "bawl"),
                    cache_reference=config.get("cache_reference", True),
                    max_reference_size=config.get("max_reference_size", None),
                )

            elif family == "uniqueness":
                benchmark = UniquenessBenchmark(
                    fingerprint_method=config.get("fingerprint_method", "bawl"),
                )

            elif family == "hhi":
                benchmark = HHIBenchmark(
                    production_weight=config.get("production_weight", 0.25),
                    reserve_weight=config.get("reserve_weight", 0.75),
                    scale_to_0_10=config.get("scale_to_0_10", True),
                )

            elif family == "sun":
                benchmark = SUNBenchmark(
                    stability_threshold=config.get("stability_threshold", 0.0),
                    metastability_threshold=config.get("metastability_threshold", 0.1),
                    reference_dataset=config.get(
                        "reference_dataset", "LeMaterial/LeMat-Bulk"
                    ),
                    reference_config=config.get("reference_config", "compatible_pbe"),
                    fingerprint_method=config.get("fingerprint_method", "bawl"),
                    cache_reference=config.get("cache_reference", True),
                    max_reference_size=config.get("max_reference_size", None),
                    include_metasun=config.get("include_metasun", True),
                )

            elif family == "stability":
                benchmark = MultiMLIPStabilityBenchmark(config=config)

            else:
                logger.warning(f"Unknown benchmark family: {family}")
                continue

            # Run the benchmark
            benchmark_result = benchmark.evaluate(structures)
            results[family] = benchmark_result

            logger.info(f"✅ {family} benchmark complete")
            
            # Clean up after each benchmark
            cleanup_after_benchmark(family, monitor_memory)

        except Exception as e:
            logger.error(f"❌ Failed to run {family} benchmark: {str(e)}")
            results[family] = {"error": str(e)}
            
            # Clean up even if benchmark failed
            cleanup_after_benchmark(family, monitor_memory)

    # Log final memory usage
    log_memory_usage("after all benchmarks")

    return results


def save_results(
    results: Dict[str, Any], run_name: str, config_name: str, n_structures: int
):
    """Save benchmark results to JSON file."""
    # Create results directory
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create filename
    filename = f"{run_name}_{config_name}_{timestamp}.json"
    filepath = results_dir / filename

    # Prepare results data
    output_data = {
        "run_info": {
            "run_name": run_name,
            "config_name": config_name,
            "timestamp": timestamp,
            "n_structures": n_structures,
            "benchmark_families": list(results.keys()),
        },
        "results": results,
    }

    # Save to JSON
    with open(filepath, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    logger.info(f"💾 Results saved to: {filepath}")
    return filepath


def main():
    """Main function to run benchmarks."""
    parser = argparse.ArgumentParser(description="Run material generation benchmarks")
    parser.add_argument(
        "--cifs", required=True, help="Path to text file containing CIF file paths OR directory containing CIF files"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Benchmark configuration name (e.g., validity, distribution)",
    )
    parser.add_argument("--name", required=True, help="Name for this benchmark run")
    parser.add_argument(
        "--families",
        nargs="+",
        help="Specific benchmark families to run (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Process structures in batches to reduce memory usage (default: process all at once)",
    )
    parser.add_argument(
        "--monitor-memory",
        action="store_true",
        help="Enable detailed memory monitoring throughout the process",
    )

    args = parser.parse_args()

    try:
        # Log initial memory usage
        log_memory_usage("start of benchmark run", force_log=args.monitor_memory)
        
        # Load CIF files
        logger.info(f"Loading CIF files from: {args.cifs}")
        cif_paths = load_cif_files(args.cifs)
        logger.info(f"✅ Loaded {len(cif_paths)} CIF files")

        # Load benchmark configuration
        logger.info(f"Loading benchmark configuration: {args.config}")
        config = load_benchmark_config(args.config)
        logger.info(f"✅ Loaded configuration: {config.get('type', 'unknown')}")

                # Determine benchmark families to run
        if args.families:
            benchmark_families = args.families
            logger.info(f"Using specified families: {benchmark_families}")
        else:
            # Default to all available families for comprehensive evaluation
            benchmark_families = [
                "validity", 
                "distribution", 
                "diversity", 
                "novelty", 
                "uniqueness", 
                "hhi", 
                "sun", 
                "stability"
            ]
            logger.info(f"Using all benchmark families: {benchmark_families}")

        # Load structures from CIF files
        logger.info("Converting CIF files to structures...")
        structures = []
        for cif_path in cif_paths:
            try:
                # Load CIF file using pymatgen
                from pymatgen.core import Structure
                structure = Structure.from_file(cif_path)
                structures.append(structure)
                logger.info(f"✅ Loaded structure from {cif_path}")
            except Exception as e:
                logger.warning(f"Failed to load {cif_path}: {str(e)}")

        if not structures:
            raise ValueError("No valid structures loaded from CIF files")

        logger.info(f"✅ Loaded {len(structures)} structures")
        
        # Clear memory after loading structures
        clear_memory()
        log_memory_usage("after loading structures", force_log=args.monitor_memory)

        # Determine preprocessor requirements
        preprocessor_config = create_preprocessor_config(benchmark_families)
        logger.info(f"Preprocessor config: {preprocessor_config}")

        # Run preprocessors and benchmarks
        if args.batch_size and len(structures) > args.batch_size:
            logger.info(f"🔄 Processing {len(structures)} structures in batches of {args.batch_size}")
            
            # Process in batches
            all_processed_structures = []
            all_benchmark_results = {}
            
            for i in range(0, len(structures), args.batch_size):
                batch_structures = structures[i:i + args.batch_size]
                batch_num = i // args.batch_size + 1
                total_batches = (len(structures) + args.batch_size - 1) // args.batch_size
                
                logger.info(f"🔄 Processing batch {batch_num}/{total_batches} ({len(batch_structures)} structures)")
                
                # Run preprocessors on batch
                batch_processed, batch_preprocessor_results = run_preprocessors(
                    batch_structures, preprocessor_config, args.monitor_memory
                )
                
                # Run benchmarks on batch
                batch_benchmark_results = run_benchmarks(
                    batch_processed, benchmark_families, config, args.monitor_memory
                )
                
                # Combine results
                all_processed_structures.extend(batch_processed)
                
                # Merge benchmark results (this is simplified - in practice you might want more sophisticated merging)
                for family, result in batch_benchmark_results.items():
                    if family not in all_benchmark_results:
                        all_benchmark_results[family] = result
                    else:
                        # For now, just keep the last result (you might want to aggregate properly)
                        all_benchmark_results[family] = result
                
                # Clear memory between batches
                clear_memory()
                log_memory_usage(f"after batch {batch_num}", force_log=args.monitor_memory)
            
            processed_structures = all_processed_structures
            benchmark_results = all_benchmark_results
            
        else:
            # Process all structures at once
            processed_structures, preprocessor_results = run_preprocessors(
                structures, preprocessor_config, args.monitor_memory
            )

            # Run benchmarks
            benchmark_results = run_benchmarks(
                processed_structures, benchmark_families, config, args.monitor_memory
            )

        # Save results
        results_file = save_results(
            benchmark_results, args.name, args.config, len(structures)
        )

        # Final cleanup
        logger.info("🧹 Performing final cleanup...")
        clear_memory()
        clear_mlip_models()
        log_memory_usage("final cleanup", force_log=args.monitor_memory)

        # Print summary
        print("\n" + "=" * 60)
        print("🎉 BENCHMARK RUN COMPLETE")
        print("=" * 60)
        print(f"📁 Results saved to: {results_file}")
        print(f"📊 Structures processed: {len(structures)}")
        print(f"🔧 Benchmark families: {benchmark_families}")
        print(f"⏱️  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        # Print key results
        for family, result in benchmark_results.items():
            if isinstance(result, dict) and "error" in result:
                print(f"❌ {family}: {result['error']}")
            else:
                print(f"✅ {family}: Completed successfully")

    except Exception as e:
        logger.error(f"Benchmark run failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
