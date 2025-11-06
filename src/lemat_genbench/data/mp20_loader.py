"""Load and process MP-20 dataset from CSV files.

This module provides utilities for loading the MP-20 dataset,
converting structures, and computing reference data for novelty metrics.
"""

import ast
import json
from io import StringIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
from tqdm import tqdm

from lemat_genbench.utils.logging import logger


def load_mp20_dataset(csv_path: str, split: Optional[str] = None) -> pd.DataFrame:
    """Load MP-20 dataset from CSV file.
    
    Parameters
    ----------
    csv_path : str
        Path to MP-20 CSV file (mp_20.csv or split files)
    split : str, optional
        Filter by split: 'train', 'val', or 'test'. 
        If None, returns all data.
    
    Returns
    -------
    pd.DataFrame
        Loaded dataset with parsed elements column
        
    Examples
    --------
    >>> # Load full dataset
    >>> df = load_mp20_dataset("mp-20-data/mp_20.csv")
    >>> 
    >>> # Load only training data
    >>> df_train = load_mp20_dataset("mp-20-data/train.csv")
    >>> 
    >>> # Load and filter by split
    >>> df = load_mp20_dataset("mp-20-data/mp_20.csv", split="train")
    """
    logger.info(f"Loading MP-20 dataset from {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Parse elements string to list if it's a string
    if isinstance(df['elements'].iloc[0], str):
        df['elements'] = df['elements'].apply(ast.literal_eval)
    
    # Filter by split if requested
    if split:
        if 'split' in df.columns:
            df = df[df['split'] == split]
            logger.info(f"Filtered to {split} split: {len(df)} structures")
        else:
            logger.warning(f"Split column not found, returning all data")
    
    logger.info(f"Loaded {len(df)} structures from MP-20 dataset")
    
    return df


def mp20_item_to_structure(row: pd.Series) -> Structure:
    """Convert MP-20 dataset row to pymatgen Structure.
    
    Parameters
    ----------
    row : pd.Series
        Row from MP-20 dataset containing at minimum:
        - cif: CIF string
        - material_id: MP material ID
        - e_above_hull: Energy above hull
        - formation_energy_per_atom: Formation energy
    
    Returns
    -------
    Structure
        Pymatgen Structure object with metadata in properties
        
    Raises
    ------
    ValueError
        If CIF parsing fails or required fields are missing
        
    Examples
    --------
    >>> df = load_mp20_dataset("mp-20-data/mp_20.csv")
    >>> structure = mp20_item_to_structure(df.iloc[0])
    >>> print(structure.composition)
    >>> print(structure.properties['material_id'])
    """
    try:
        # Parse CIF string
        cif_string = row['cif']
        parser = CifParser(StringIO(cif_string))
        structure = parser.get_structures()[0]
        
        # Add metadata to structure properties
        structure.properties['material_id'] = row.get('material_id', 'unknown')
        structure.properties['e_above_hull'] = float(row.get('e_above_hull', np.nan))
        structure.properties['formation_energy_per_atom'] = float(
            row.get('formation_energy_per_atom', np.nan)
        )
        
        # Add optional fields if present
        if 'band_gap' in row:
            structure.properties['band_gap'] = float(row.get('band_gap', np.nan))
        if 'spacegroup.number' in row:
            structure.properties['spacegroup_number'] = int(row.get('spacegroup.number', 0))
        if 'pretty_formula' in row:
            structure.properties['formula'] = row.get('pretty_formula', '')
        
        return structure
        
    except Exception as e:
        logger.error(f"Failed to convert MP-20 row to Structure: {e}")
        raise ValueError(f"Invalid MP-20 row format: {e}")


def one_hot_encode_composition(elements):
    """One-hot encode a composition into a vector.
    
    Parameters
    ----------
    elements : list
        List of element symbols or Element objects.
        
    Returns
    -------
    np.ndarray
        One-hot encoded vector of length 118 (for all elements).
    """
    from pymatgen.core import Element
    
    one_hot = np.zeros(118)
    for element in elements:
        element_obj = Element(element) if isinstance(element, str) else element
        one_hot[int(element_obj.number) - 1] = 1
    return one_hot


def get_mp20_compositions(csv_path: str = "mp-20-data/mp_20.csv") -> scipy.sparse.csr_matrix:
    """Get composition matrix for MP-20 dataset.
    
    Creates a sparse matrix where each row is a one-hot encoded composition
    vector indicating which elements are present in each structure.
    This is used for fast composition-based filtering during novelty checks.
    
    Parameters
    ----------
    csv_path : str, default="mp-20-data/mp_20.csv"
        Path to MP-20 CSV file
        
    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse matrix of shape (n_structures, 118) containing one-hot
        encoded compositions. Each row corresponds to a structure,
        each column to an element (1-118).
        
    Notes
    -----
    The composition matrix is cached in data/mp20_compositions_{split}.npz.
    If this file exists, it will be loaded instead of recomputed.
    Cache is split-aware (e.g., mp_20, train, test, val).
    
    This enables O(n*m) filtering where n is the number of reference
    structures and m is 118, which is much faster than O(n) structure
    comparisons.
    
    Examples
    --------
    >>> # First call computes and caches
    >>> compositions = get_mp20_compositions()
    >>> print(compositions.shape)  # (n_structures, 118)
    >>> 
    >>> # Subsequent calls load from cache
    >>> compositions = get_mp20_compositions()  # Fast!
    """
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Split-aware cache filename
    dataset_name = Path(csv_path).stem  # e.g., "mp_20", "train", "test"
    compositions_file = data_dir / f"mp20_compositions_{dataset_name}.npz"
    
    try:
        logger.info(f"Loading existing MP-20 compositions from {compositions_file}")
        all_compositions = scipy.sparse.load_npz(compositions_file)
        logger.info(f"✅ Loaded compositions matrix with shape: {all_compositions.shape}")
        return all_compositions
        
    except FileNotFoundError:
        logger.info("🔄 MP-20 compositions file not found, generating from dataset...")
        
        # Load dataset
        df = load_mp20_dataset(csv_path)
        
        logger.info(f"📊 Processing {len(df)} structures...")
        
        # Encode compositions
        all_compositions = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Encoding compositions"):
            elements = row['elements']
            all_compositions.append(one_hot_encode_composition(elements))
        
        # Convert to sparse matrix and save
        all_compositions = np.array(all_compositions)
        all_compositions = scipy.sparse.csr_matrix(all_compositions)
        scipy.sparse.save_npz(compositions_file, all_compositions)
        
        logger.info(f"💾 Saved compositions to {compositions_file}")
        logger.info(f"✅ Generated compositions matrix with shape: {all_compositions.shape}")
        
    return all_compositions


def filter_df_by_composition(
    df: pd.DataFrame, 
    all_compositions: scipy.sparse.csr_matrix, 
    structure: Structure
) -> pd.DataFrame:
    """Filter DataFrame to only structures with same elements as test structure.
    
    This is the critical optimization for structure matcher novelty checks.
    Instead of comparing against all reference structures, we filter to only
    those containing the same elements.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with indexed structures
    all_compositions : scipy.sparse.csr_matrix
        One-hot encoded composition matrix (from get_mp20_compositions)
    structure : Structure
        Test structure to filter against
        
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only structures with same elements
        
    Notes
    -----
    Speedup: Typically reduces from ~45K structures to ~100-1000 structures
    
    Examples
    --------
    >>> df = load_mp20_dataset("mp-20-data/mp_20.csv")
    >>> df = df.set_index('material_id')
    >>> compositions = get_mp20_compositions()
    >>> 
    >>> # Filter to structures with same elements as test_structure
    >>> filtered = filter_df_by_composition(df, compositions, test_structure)
    >>> print(f"Filtered from {len(df)} to {len(filtered)} structures")
    """
    # Encode test structure's elements as one-hot vector
    structure_vector = one_hot_encode_composition(
        structure.composition.elements
    ).reshape(-1, 1)
    
    # Create forbidden elements mask (elements NOT in test structure)
    forbidden_elements = 1 - structure_vector
    
    # Matrix multiplication: for each reference structure, count forbidden elements
    # Keep only structures with 0 forbidden elements (i.e., same element set)
    forbidden_counts = all_compositions @ forbidden_elements
    
    # Handle both sparse and dense arrays
    if scipy.sparse.issparse(forbidden_counts):
        valid_indices = (forbidden_counts == 0).A1  # Convert sparse to dense boolean array
    else:
        valid_indices = (forbidden_counts == 0).flatten()  # Handle dense arrays
    
    # Filter dataframe
    intersection_elements = df.iloc[valid_indices]
    
    return intersection_elements


def compute_mp20_statistics(csv_path: str = "mp-20-data/mp_20.csv") -> dict:
    """Compute statistics about the MP-20 dataset.
    
    Parameters
    ----------
    csv_path : str
        Path to MP-20 CSV file
        
    Returns
    -------
    dict
        Dictionary containing dataset statistics
    """
    df = load_mp20_dataset(csv_path)
    
    # Flatten elements list
    all_elements = [el for elements in df['elements'] for el in elements]
    element_counts = pd.Series(all_elements).value_counts()
    
    stats = {
        "n_structures": len(df),
        "n_unique_materials": df['material_id'].nunique() if 'material_id' in df.columns else len(df),
        "splits": df['split'].value_counts().to_dict() if 'split' in df.columns else {},
        "element_counts": {str(el): int(count) for el, count in element_counts.items()},
        "n_unique_elements": len(element_counts),
        "e_above_hull_stats": {
            "mean": float(df['e_above_hull'].mean()),
            "std": float(df['e_above_hull'].std()),
            "min": float(df['e_above_hull'].min()),
            "max": float(df['e_above_hull'].max()),
            "median": float(df['e_above_hull'].median()),
        } if 'e_above_hull' in df.columns else {},
        "formation_energy_stats": {
            "mean": float(df['formation_energy_per_atom'].mean()),
            "std": float(df['formation_energy_per_atom'].std()),
            "min": float(df['formation_energy_per_atom'].min()),
            "max": float(df['formation_energy_per_atom'].max()),
            "median": float(df['formation_energy_per_atom'].median()),
        } if 'formation_energy_per_atom' in df.columns else {},
    }
    
    return stats


if __name__ == "__main__":
    # Quick test
    print("Testing MP-20 loader...")
    
    # Test loading
    df = load_mp20_dataset("mp-20-data/mp_20.csv")
    print(f"✅ Loaded {len(df)} structures")
    
    # Test structure conversion
    structure = mp20_item_to_structure(df.iloc[0])
    print(f"✅ Converted structure: {structure.composition}")
    print(f"   Properties: {list(structure.properties.keys())}")
    
    # Test composition encoding
    compositions = get_mp20_compositions("mp-20-data/mp_20.csv")
    print(f"✅ Composition matrix: {compositions.shape}")
    
    # Test statistics
    stats = compute_mp20_statistics("mp-20-data/mp_20.csv")
    print(f"✅ Statistics: {stats['n_structures']} structures, {stats['n_unique_elements']} elements")

