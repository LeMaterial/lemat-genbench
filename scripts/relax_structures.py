#!/usr/bin/env python3
"""Structure relaxation script using MLIPs.

This script reads CIF files from a specified folder, relaxes them using
the chosen MLIP (UMA, ORB, or MACE), and saves the relaxed structures
in an organized folder structure.

Usage:
    uv run scripts/relax_structures.py --input_folder /path/to/cifs --mlip uma
    uv run scripts/relax_structures.py --input_folder /path/to/cifs --mlip orb --data_name my_structures
    uv run scripts/relax_structures.py --input_folder /path/to/cifs --mlip mace --relaxation_config fmax=0.01,steps=100
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter

from lemat_genbench.models.registry import get_calculator
from lemat_genbench.utils.logging import logger


def parse_relaxation_config(config_str: str) -> dict:
    """Parse relaxation configuration string.
    
    Args:
        config_str: String like "fmax=0.01,steps=100"
        
    Returns:
        Dictionary with relaxation parameters
    """
    if not config_str:
        return {"fmax": 0.02, "steps": 500}
    
    config = {}
    for param in config_str.split(','):
        if '=' in param:
            key, value = param.strip().split('=', 1)
            try:
                # Try to convert to float first, then int
                if '.' in value:
                    config[key] = float(value)
                else:
                    config[key] = int(value)
            except ValueError:
                config[key] = value
        else:
            logger.warning(f"Invalid parameter format: {param}")
    
    return config


def get_mlip_config(mlip_name: str) -> dict:
    """Get default configuration for the specified MLIP.
    
    Args:
        mlip_name: Name of the MLIP ("uma", "orb", "mace")
        
    Returns:
        Dictionary with MLIP configuration
    """
    configs = {
        "uma": {"task": "omat", "device": "cpu"},
        "orb": {"model_type": "orb_v3_conservative_inf_omat", "device": "cpu"},
        "mace": {"model_type": "mp", "device": "cpu"},
    }
    
    if mlip_name not in configs:
        raise ValueError(f"Unknown MLIP: {mlip_name}. Available: {list(configs.keys())}")
    
    return configs[mlip_name]


def read_cif_files(input_folder: str) -> List[Structure]:
    """Read all CIF files from the input folder.
    
    Args:
        input_folder: Path to folder containing CIF files
        
    Returns:
        List of pymatgen Structure objects
    """
    input_path = Path(input_folder)
    if not input_path.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")
    
    cif_files = list(input_path.glob("*.cif"))
    if not cif_files:
        raise ValueError(f"No CIF files found in {input_folder}")
    
    logger.info(f"Found {len(cif_files)} CIF files in {input_folder}")
    
    structures = []
    failed_files = []
    
    for cif_file in cif_files:
        try:
            structure = Structure.from_file(str(cif_file))
            structures.append(structure)
            logger.debug(f"Successfully loaded {cif_file.name}")
        except Exception as e:
            logger.warning(f"Failed to load {cif_file.name}: {str(e)}")
            failed_files.append(cif_file.name)
    
    if failed_files:
        logger.warning(f"Failed to load {len(failed_files)} files: {failed_files}")
    
    logger.info(f"Successfully loaded {len(structures)} structures")
    return structures


def relax_structure(structure: Structure, calculator, relaxation_config: dict) -> tuple:
    """Relax a single structure using the specified calculator.
    
    Args:
        structure: Structure to relax
        calculator: MLIP calculator instance
        relaxation_config: Relaxation parameters
        
    Returns:
        Tuple of (relaxed_structure, relaxation_result)
    """
    try:
        relaxed_structure, relaxation_result = calculator.relax_structure(
            structure, **relaxation_config
        )
        
        # Calculate RMSE between original and relaxed positions
        rmse = calculate_rmse(structure, relaxed_structure)
        
        # Add relaxation metadata to the result
        relaxation_result.metadata["relaxation_rmse"] = rmse
        
        return relaxed_structure, relaxation_result
        
    except Exception as e:
        logger.error(f"Failed to relax structure {structure.formula}: {str(e)}")
        raise


def calculate_rmse(original: Structure, relaxed: Structure) -> float:
    """Calculate RMSE between atomic positions of original and relaxed structures.
    
    Args:
        original: Original structure
        relaxed: Relaxed structure
        
    Returns:
        RMSE in Angstroms
    """
    if len(original) != len(relaxed):
        raise ValueError("Structures must have the same number of atoms")
    
    mse = 0.0
    for i in range(len(original)):
        original_coords = original[i].coords
        relaxed_coords = relaxed[i].coords
        mse += np.linalg.norm(original_coords - relaxed_coords) ** 2
    
    mse /= len(original)
    return np.sqrt(mse)


def save_relaxed_structure(structure: Structure, output_path: Path, filename: str) -> None:
    """Save a relaxed structure as a CIF file.
    
    Args:
        structure: Relaxed structure to save
        output_path: Output directory path
        filename: Name for the output file (without extension)
    """
    output_path.mkdir(parents=True, exist_ok=True)
    cif_path = output_path / f"{filename}.cif"
    
    try:
        writer = CifWriter(structure)
        writer.write_file(str(cif_path))
        logger.debug(f"Saved relaxed structure to {cif_path}")
    except Exception as e:
        logger.error(f"Failed to save {filename}: {str(e)}")
        raise


def create_output_folder(data_name: str, mlip_name: str, timestamp: str) -> Path:
    """Create the output folder structure.
    
    Args:
        data_name: Name of the dataset
        mlip_name: Name of the MLIP used
        timestamp: Timestamp string
        
    Returns:
        Path to the created output folder
    """
    # Create submissions/relaxed_structures if it doesn't exist
    base_path = Path("submissions") / "relaxed_structures"
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Create the specific output folder
    folder_name = f"{data_name}_{mlip_name}_{timestamp}"
    output_path = base_path / folder_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created output folder: {output_path}")
    return output_path


def main():
    """Main function for the structure relaxation script."""
    parser = argparse.ArgumentParser(
        description="Relax CIF structures using MLIPs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run scripts/relax_structures.py --input_folder /path/to/cifs --mlip uma
  uv run scripts/relax_structures.py --input_folder /path/to/cifs --mlip orb --data_name my_structures
  uv run scripts/relax_structures.py --input_folder /path/to/cifs --mlip mace --relaxation_config "fmax=0.01,steps=100"
        """
    )
    
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Path to folder containing CIF files to relax"
    )
    
    parser.add_argument(
        "--mlip",
        type=str,
        required=True,
        choices=["uma", "orb", "mace"],
        help="MLIP to use for relaxation (uma, orb, mace)"
    )
    
    parser.add_argument(
        "--data_name",
        type=str,
        default=None,
        help="Name for the dataset (defaults to input folder name)"
    )
    
    parser.add_argument(
        "--relaxation_config",
        type=str,
        default="fmax=0.02,steps=500",
        help="Relaxation configuration as 'key=value,key=value' (default: fmax=0.02,steps=500)"
    )
    
    parser.add_argument(
        "--max_structures",
        type=int,
        default=None,
        help="Maximum number of structures to process (for testing)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logger.set_level("DEBUG")
    else:
        logger.set_level("INFO")
    
    try:
        # Generate data name if not provided
        if args.data_name is None:
            args.data_name = Path(args.input_folder).name
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Parse relaxation configuration
        relaxation_config = parse_relaxation_config(args.relaxation_config)
        logger.info(f"Relaxation config: {relaxation_config}")
        
        # Get MLIP configuration
        mlip_config = get_mlip_config(args.mlip)
        logger.info(f"MLIP config for {args.mlip}: {mlip_config}")
        
        # Create output folder
        output_path = create_output_folder(args.data_name, args.mlip, timestamp)
        
        # Read CIF files
        logger.info(f"Reading CIF files from {args.input_folder}")
        structures = read_cif_files(args.input_folder)
        
        # Limit structures if specified
        if args.max_structures and args.max_structures < len(structures):
            structures = structures[:args.max_structures]
            logger.info(f"Limited to {len(structures)} structures for processing")
        
        # Initialize calculator
        logger.info(f"Initializing {args.mlip.upper()} calculator...")
        calculator = get_calculator(args.mlip, **mlip_config)
        logger.info("Calculator initialized successfully")
        
        # Process structures
        logger.info(f"Starting relaxation of {len(structures)} structures...")
        
        successful_relaxations = 0
        failed_relaxations = 0
        
        for i, structure in enumerate(structures):
            try:
                logger.info(f"Processing structure {i+1}/{len(structures)}: {structure.formula}")
                
                # Relax structure
                relaxed_structure, relaxation_result = relax_structure(
                    structure, calculator, relaxation_config
                )
                
                # Generate filename
                filename = f"relaxed_{i+1:04d}_{structure.formula.replace(' ', '_')}"
                
                # Save relaxed structure
                save_relaxed_structure(relaxed_structure, output_path, filename)
                
                # Log relaxation info
                rmse = relaxation_result.metadata.get("relaxation_rmse", "N/A")
                steps = relaxation_result.metadata.get("relaxation_steps", "N/A")
                logger.info(f"  ✓ Relaxed successfully (RMSE: {rmse:.4f} Å, Steps: {steps})")
                
                successful_relaxations += 1
                
            except Exception as e:
                logger.error(f"  ✗ Failed to relax structure {i+1}: {str(e)}")
                failed_relaxations += 1
                continue
        
        # Summary
        logger.info("=" * 60)
        logger.info("RELAXATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total structures processed: {len(structures)}")
        logger.info(f"Successful relaxations: {successful_relaxations}")
        logger.info(f"Failed relaxations: {failed_relaxations}")
        logger.info(f"Success rate: {successful_relaxations/len(structures)*100:.1f}%")
        logger.info(f"Output folder: {output_path}")
        
        if failed_relaxations > 0:
            logger.warning(f"{failed_relaxations} structures failed to relax. Check logs for details.")
            sys.exit(1)
        else:
            logger.info("All structures relaxed successfully!")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
