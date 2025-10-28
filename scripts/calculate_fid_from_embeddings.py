#!/usr/bin/env python3
"""
Calculate FID (Fréchet Distance) for models that have saved embeddings.
"""

import pickle
from pathlib import Path

import numpy as np

from lemat_genbench.utils.distribution_utils import (
    compute_frechetdist_with_cache,
    load_reference_stats_cache,
)


def calculate_fid_for_model(embeddings_file, model_name, mlips=['orb', 'mace', 'uma']):
    """
    Calculate FID for a model using saved embeddings.
    
    Parameters
    ----------
    embeddings_file : str or Path
        Path to pickle file containing embeddings
    model_name : str
        Name of the model for display
    mlips : list
        List of MLIP names to compute FID for
        
    Returns
    -------
    dict
        Dictionary with FID results
    """
    print(f"\n{'='*60}")
    print(f"Calculating FID for: {model_name}")
    print(f"{'='*60}")
    
    # Load reference statistics
    cache_dir = "data"
    print(f"\n1. Loading reference statistics from {cache_dir}...")
    reference_stats = load_reference_stats_cache(cache_dir, mlips)
    
    if not reference_stats:
        raise ValueError(f"Could not load reference statistics from {cache_dir}")
    
    print(f"   ✅ Loaded reference stats for: {list(reference_stats.keys())}")
    
    # Load embeddings
    print(f"\n2. Loading embeddings from {embeddings_file}...")
    with open(embeddings_file, 'rb') as f:
        embeddings_data = pickle.load(f)
    
    print(f"   Available keys: {list(embeddings_data.keys())}")
    
    # Calculate FID for each MLIP
    distances = []
    warnings = []
    
    for mlip in mlips:
        print(f"\n3. Computing FID for {mlip.upper()}...")
        
        # Get graph embeddings for this MLIP
        key = f"{mlip}_graph"
        
        if key not in embeddings_data:
            warning = f"No {key} found in embeddings file"
            warnings.append(warning)
            print(f"   ⚠️  {warning}")
            continue
        
        embeddings = embeddings_data[key]
        print(f"   Found {len(embeddings)} embeddings")
        print(f"   Embedding shape: {embeddings[0].shape if len(embeddings) > 0 else 'N/A'}")
        
        # Check for valid embeddings
        valid_embeddings = []
        for i, emb in enumerate(embeddings):
            if emb is not None and isinstance(emb, np.ndarray) and len(emb) > 0:
                # Check for NaN/Inf
                if not (np.isnan(emb).any() or np.isinf(emb).any()):
                    valid_embeddings.append(emb)
        
        print(f"   Valid embeddings: {len(valid_embeddings)}/{len(embeddings)}")
        
        if len(valid_embeddings) == 0:
            warning = f"No valid embeddings for {mlip}"
            warnings.append(warning)
            print(f"   ⚠️  {warning}")
            continue
        
        try:
            # Get cached reference stats
            if mlip not in reference_stats:
                warning = f"No reference stats for {mlip}"
                warnings.append(warning)
                print(f"   ⚠️  {warning}")
                continue
            
            cached_stats = reference_stats[mlip]
            
            # Compute FID
            frechet_dist = compute_frechetdist_with_cache(
                cached_stats["mu"],
                cached_stats["sigma"],
                valid_embeddings
            )
            
            distances.append(frechet_dist)
            print(f"   ✅ FID = {frechet_dist:.4f}")
            
        except Exception as e:
            warning = f"Failed to compute FID for {mlip}: {str(e)}"
            warnings.append(warning)
            print(f"   ❌ {warning}")
    
    # Compute mean FID
    if not distances:
        raise ValueError("No valid FID computed for any MLIP")
    
    mean_fid = np.mean(distances)
    std_fid = np.std(distances) if len(distances) > 1 else 0.0
    
    print(f"\n{'='*60}")
    print(f"RESULTS for {model_name}:")
    print(f"{'='*60}")
    print(f"  Mean FID: {mean_fid:.4f}")
    print(f"  Std FID:  {std_fid:.4f}")
    print(f"  Individual FIDs: {[f'{d:.4f}' for d in distances]}")
    print(f"  MLIPs computed: {len(distances)}/{len(mlips)}")
    
    if warnings:
        print("\n  ⚠️  Warnings:")
        for w in warnings:
            print(f"     • {w}")
    
    return {
        'model_name': model_name,
        'mean_fid': mean_fid,
        'std_fid': std_fid,
        'individual_fids': distances,
        'n_models_computed': len(distances),
        'warnings': warnings
    }


if __name__ == "__main__":
    # Calculate FID for DiffCSP++
    diffcsp_pp_file = Path("results_october/embeddings_diffcsp_pp_21102025/embeddings_orb_mace_uma_20251023_134508.pkl")
    if diffcsp_pp_file.exists():
        try:
            diffcsp_pp_result = calculate_fid_for_model(
                diffcsp_pp_file,
                "DiffCSP++"
            )
        except Exception as e:
            print(f"\n❌ Error calculating FID for DiffCSP++: {e}")
    
    # Calculate FID for SymmCD
    symmcd_file = Path("results_october/embeddings_symmcd_21102025/embeddings_orb_mace_uma_20251024_065704.pkl")
    if symmcd_file.exists():
        try:
            symmcd_result = calculate_fid_for_model(
                symmcd_file,
                "SymmCD"
            )
        except Exception as e:
            print(f"\n❌ Error calculating FID for SymmCD: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    if diffcsp_pp_file.exists() and 'diffcsp_pp_result' in locals():
        print(f"DiffCSP++: Mean FID = {diffcsp_pp_result['mean_fid']:.4f}")
    if symmcd_file.exists() and 'symmcd_result' in locals():
        print(f"SymmCD:    Mean FID = {symmcd_result['mean_fid']:.4f}")

