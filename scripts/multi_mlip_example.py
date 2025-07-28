"""Example usage of the Multi-MLIP Stability Preprocessor.

This script demonstrates how to use multiple MLIPs simultaneously for
robust stability analysis and uncertainty quantification.
"""

import numpy as np
from pymatgen.core import Structure
from pymatgen.util.testing import PymatgenTest

from lematerial_forgebench.preprocess.multi_mlip_preprocess import (
    MultiMLIPStabilityPreprocessor,
    create_multi_mlip_preprocessor,
    create_orb_mace_uma_preprocessor,
)


def example_basic_usage():
    """Basic usage example with default configuration."""
    print("=== Basic Multi-MLIP Preprocessing ===")
    
    # Create test structures
    test = PymatgenTest()
    structures = [
        test.get_structure("Si"),
        test.get_structure("LiFePO4"),
    ]
    
    # Create preprocessor with ORB, MACE, and UMA
    preprocessor = create_orb_mace_uma_preprocessor()
    
    # Process structures
    result = preprocessor(structures)
    
    # Examine results
    for i, structure in enumerate(result.processed_structures):
        print(f"\nStructure {i+1}: {structure.formula}")
        print(f"Available properties: {len(structure.properties)} properties")
        
        # Show MLIP-specific energies
        mlip_energies = {}
        for prop_name in structure.properties:
            if prop_name.startswith("energy_") and not prop_name.endswith(("_mean", "_std", "_n_mlips")):
                mlip_name = prop_name.replace("energy_", "")
                mlip_energies[mlip_name] = structure.properties[prop_name]
        
        print(f"MLIP-specific energies: {mlip_energies}")
        print(f"Mean energy: {structure.properties.get('energy_mean', 'N/A')}")
        print(f"Energy std: {structure.properties.get('energy_std', 'N/A')}")


def example_custom_configuration():
    """Example with custom MLIP configurations."""
    print("\n=== Custom Configuration Example ===")
    
    # Custom configurations for each MLIP (these will override defaults)
    mlip_configs = {
        "orb": {
            "model_type": "orb_v3_conservative_inf_omat",  # Default
            "device": "cpu"
        },
        "mace": {
            "model_type": "mp",  # Default
            "device": "cpu"
        },
        "uma": {
            "task": "omat",  # Default  
            "device": "CPU"
        }
    }
    
    # Create preprocessor with custom configs
    preprocessor = MultiMLIPStabilityPreprocessor(
        mlip_names=["orb", "mace", "uma"],
        mlip_configs=mlip_configs,
        relax_structures=True,
        relaxation_config={"fmax": 0.01, "steps": 300},  # Tighter convergence
        calculate_formation_energy=True,
        calculate_energy_above_hull=True,
        extract_embeddings=True,
        timeout=120,  # Longer timeout
    )
    
    # Test with a single structure
    test = PymatgenTest()
    structure = test.get_structure("Si")
    
    result = preprocessor([structure])
    processed_structure = result.processed_structures[0]
    
    print(f"Processed structure: {processed_structure.formula}")
    print(f"Total properties: {len(processed_structure.properties)}")


def analyze_results(structure: Structure):
    """Analyze and display multi-MLIP results for a structure."""
    print(f"\n=== Analysis for {structure.formula} ===")
    
    # Scalar metrics analysis
    scalar_metrics = [
        "energy", "formation_energy", "e_above_hull",
        "relaxation_energy", "relaxation_rmse", "relaxation_steps",
        "relaxed_formation_energy", "relaxed_e_above_hull"
    ]
    
    print("\nScalar Metrics Summary:")
    print("-" * 60)
    print(f"{'Metric':<25} {'Mean':<12} {'Std':<12} {'N_MLIPs':<8}")
    print("-" * 60)
    
    for metric in scalar_metrics:
        mean_val = structure.properties.get(f"{metric}_mean")
        std_val = structure.properties.get(f"{metric}_std")
        n_mlips = structure.properties.get(f"{metric}_n_mlips")
        
        if mean_val is not None:
            print(f"{metric:<25} {mean_val:<12.4f} {std_val:<12.4f} {n_mlips:<8}")
        else:
            print(f"{metric:<25} {'N/A':<12} {'N/A':<12} {n_mlips or 0:<8}")
    
    # MLIP-specific energies
    print("\nMLIP-Specific Energies:")
    print("-" * 40)
    for prop_name, value in structure.properties.items():
        if prop_name.startswith("energy_") and not prop_name.endswith(("_mean", "_std", "_n_mlips")):
            mlip_name = prop_name.replace("energy_", "")
            print(f"{mlip_name.upper():<10}: {value:>12.4f} eV")
    
    # Forces analysis
    forces_mean = structure.properties.get("forces_mean")
    forces_std = structure.properties.get("forces_std")
    if forces_mean is not None:
        print("\nForces Analysis:")
        print(f"Mean forces shape: {forces_mean.shape}")
        print(f"Max force magnitude (mean): {np.max(np.linalg.norm(forces_mean, axis=1)):.4f} eV/Å")
        print(f"Max force std: {np.max(forces_std):.4f} eV/Å")
    
    # Embeddings analysis
    print("\nEmbeddings Available:")
    embedding_mlips = []
    for prop_name in structure.properties:
        if prop_name.startswith("graph_embedding_"):
            mlip_name = prop_name.replace("graph_embedding_", "")
            embedding_mlips.append(mlip_name)
            embedding_shape = structure.properties[prop_name].shape
            print(f"{mlip_name.upper():<10}: Graph embedding shape {embedding_shape}")
    
    # Relaxed structures
    print("\nRelaxed Structures Available:")
    for prop_name in structure.properties:
        if prop_name.startswith("relaxed_structure_"):
            mlip_name = prop_name.replace("relaxed_structure_", "")
            relaxed_struct = structure.properties[prop_name]
            print(f"{mlip_name.upper():<10}: {relaxed_struct.formula} ({len(relaxed_struct)} atoms)")


def example_subset_mlips():
    """Example using only a subset of MLIPs."""
    print("\n=== Subset MLIPs Example ===")
    
    # Use only ORB and MACE
    preprocessor = create_multi_mlip_preprocessor(
        mlip_names=["orb", "mace"],
        relax_structures=True,
    )
    
    test = PymatgenTest()
    structure = test.get_structure("LiFePO4")
    
    result = preprocessor([structure])
    processed_structure = result.processed_structures[0]
    
    analyze_results(processed_structure)


def example_error_handling():
    """Example demonstrating error handling."""
    print("\n=== Error Handling Example ===")
    
    # Create preprocessor with short timeout to trigger timeouts
    preprocessor = create_multi_mlip_preprocessor(
        mlip_names=["orb", "mace"],
        timeout=1,  # Very short timeout
    )
    
    test = PymatgenTest()
    structure = test.get_structure("Si")
    
    result = preprocessor([structure])
    processed_structure = result.processed_structures[0]
    
    print("Properties after processing with short timeout:")
    error_props = {k: v for k, v in processed_structure.properties.items() 
                   if "error" in k or "failed" in k}
    
    if error_props:
        print("Error properties found:")
        for prop, value in error_props.items():
            print(f"  {prop}: {value}")
    else:
        print("No error properties found")
    
    # Check which MLIPs succeeded
    successful_mlips = []
    for prop_name in processed_structure.properties:
        if prop_name.startswith("energy_") and not prop_name.endswith(("_mean", "_std", "_n_mlips")):
            mlip_name = prop_name.replace("energy_", "")
            successful_mlips.append(mlip_name)
    
    print(f"Successful MLIPs: {successful_mlips}")


def example_statistical_analysis():
    """Example of statistical analysis across MLIPs."""
    print("\n=== Statistical Analysis Example ===")
    
    # Create preprocessor with ORB, MACE, and UMA (using defaults)
    preprocessor = create_orb_mace_uma_preprocessor()
    
    test = PymatgenTest()
    structures = [
        test.get_structure("Si"),
        test.get_structure("LiFePO4"),
    ]
    
    result = preprocessor(structures)
    
    print("Statistical Analysis Across Structures and MLIPs:")
    print("=" * 70)
    
    for i, structure in enumerate(result.processed_structures):
        print(f"\nStructure {i+1}: {structure.formula}")
        
        # Energy statistics
        energy_mean = structure.properties.get("energy_mean")
        energy_std = structure.properties.get("energy_std")
        energy_n_mlips = structure.properties.get("energy_n_mlips", 0)
        
        if energy_mean is not None and energy_n_mlips > 1:
            cv = (energy_std / abs(energy_mean)) * 100  # Coefficient of variation
            print(f"  Energy: {energy_mean:.4f} ± {energy_std:.4f} eV (CV: {cv:.2f}%, n={energy_n_mlips})")
        
        # Formation energy statistics
        fe_mean = structure.properties.get("formation_energy_mean")
        fe_std = structure.properties.get("formation_energy_std")
        fe_n_mlips = structure.properties.get("formation_energy_n_mlips", 0)
        
        if fe_mean is not None and fe_n_mlips > 1:
            fe_cv = (fe_std / abs(fe_mean)) * 100 if fe_mean != 0 else float('inf')
            print(f"  Formation Energy: {fe_mean:.4f} ± {fe_std:.4f} eV/atom (CV: {fe_cv:.2f}%, n={fe_n_mlips})")
        
        # E_above_hull statistics
        eh_mean = structure.properties.get("e_above_hull_mean")
        eh_std = structure.properties.get("e_above_hull_std")
        eh_n_mlips = structure.properties.get("e_above_hull_n_mlips", 0)
        
        if eh_mean is not None and eh_n_mlips > 1:
            print(f"  E_above_hull: {eh_mean:.4f} ± {eh_std:.4f} eV/atom (n={eh_n_mlips})")
            
            # Stability consensus
            if eh_std < 0.01:
                consensus = "High"
            elif eh_std < 0.05:
                consensus = "Medium"
            else:
                consensus = "Low"
            print(f"  Stability Consensus: {consensus} (std = {eh_std:.4f})")


def compare_relaxed_structures():
    """Compare relaxed structures from different MLIPs."""
    print("\n=== Relaxed Structure Comparison ===")
    
    preprocessor = create_multi_mlip_preprocessor(
        mlip_names=["orb", "mace", "uma"],
        relax_structures=True,
    )
    
    test = PymatgenTest()
    structure = test.get_structure("Si")
    
    result = preprocessor([structure])
    processed_structure = result.processed_structures[0]
    
    print(f"Original structure: {structure.formula}")
    print(f"Original volume: {structure.volume:.4f} Å³")
    
    # Compare relaxed structures
    relaxed_structures = {}
    for prop_name in processed_structure.properties:
        if prop_name.startswith("relaxed_structure_"):
            mlip_name = prop_name.replace("relaxed_structure_", "")
            relaxed_structures[mlip_name] = processed_structure.properties[prop_name]
    
    print("\nRelaxed Structure Comparison:")
    print("-" * 50)
    
    for mlip_name, relaxed_struct in relaxed_structures.items():
        volume_change = ((relaxed_struct.volume - structure.volume) / structure.volume) * 100
        
        # Get RMSE for this MLIP
        rmse = processed_structure.properties.get(f"relaxation_rmse_{mlip_name}")
        steps = processed_structure.properties.get(f"relaxation_steps_{mlip_name}")
        
        print(f"{mlip_name.upper()}:")
        print(f"  Volume: {relaxed_struct.volume:.4f} Å³ (Δ: {volume_change:+.2f}%)")
        print(f"  RMSE: {rmse:.4f} Å")
        print(f"  Steps: {steps}")
        
        # Calculate lattice parameter changes
        orig_params = structure.lattice.abc
        relax_params = relaxed_struct.lattice.abc
        
        param_changes = []
        for orig, relax in zip(orig_params, relax_params):
            change = ((relax - orig) / orig) * 100
            param_changes.append(change)
        
        print(f"  Lattice changes: a={param_changes[0]:+.2f}%, b={param_changes[1]:+.2f}%, c={param_changes[2]:+.2f}%")
        print()


def main():
    """Run all examples."""
    print("Multi-MLIP Stability Preprocessor Examples")
    print("=" * 50)
    
    try:
        example_basic_usage()
        example_custom_configuration()
        example_subset_mlips()
        example_error_handling()
        example_statistical_analysis()
        compare_relaxed_structures()
        
    except Exception as e:
        print(f"Error running examples: {str(e)}")
        print("Make sure the required MLIPs are installed and available.")


if __name__ == "__main__":
    main()