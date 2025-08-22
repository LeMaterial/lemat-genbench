"""Encode compositions from LeMat-Bulk dataset.

This module provides utilities for encoding material compositions
and filtering datasets based on composition overlap.
"""

from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy
from datasets import load_dataset
from pymatgen.core import Element, Structure
from scipy.sparse import csr_matrix
from tqdm import tqdm


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
    one_hot = np.zeros(118)
    for element in elements:
        element_obj = Element(element) if isinstance(element, str) else element
        one_hot[int(element_obj.number) - 1] = 1
    return one_hot


def process_chunk(chunk):
    """Process a chunk of elements for multiprocessing.
    
    Parameters
    ----------
    chunk : list
        List of element lists to process.
        
    Returns
    -------
    list
        List of one-hot encoded compositions.
    """
    one_hot_compositions = []
    for elements in tqdm(chunk, desc="Processing chunk"):
        one_hot_compositions.append(one_hot_encode_composition(elements))
    return one_hot_compositions


def lematbulk_item_to_structure(item: dict) -> Structure:
    """Convert a LeMat-Bulk item to a pymatgen Structure object.

    Parameters
    ----------
    item : dict
        The item to convert. Must contain 'species_at_sites', 
        'cartesian_site_positions', and 'lattice_vectors' keys.

    Returns
    -------
    Structure
        The pymatgen Structure object.
    """
    sites = item["species_at_sites"]
    coords = item["cartesian_site_positions"]
    cell = item["lattice_vectors"]

    structure = Structure(
        species=sites, coords=coords, lattice=cell, coords_are_cartesian=True
    )

    return structure


def get_all_compositions(num_processes=1):
    """Get all compositions from LeMat-Bulk dataset.
    
    Parameters
    ----------
    num_processes : int, default=1
        Number of processes to use for computation.
        
    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse matrix of one-hot encoded compositions.
    """
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    compositions_file = data_dir / "all_compositions.npz"
    
    try:
        # Try to load existing file
        print(f"Loading existing compositions from {compositions_file}")
        all_compositions = scipy.sparse.load_npz(compositions_file)
        print(f"✅ Loaded compositions matrix with shape: {all_compositions.shape}")
        return all_compositions
    except FileNotFoundError:
        print("🔄 Compositions file not found, generating from dataset...")
        
        # Load dataset
        print("📥 Loading LeMat-Bulk dataset...")
        dataset = load_dataset(
            "LeMaterial/LeMat-Bulk",
            "compatible_pbe",
            split="train",
            columns=["elements", "immutable_id", "chemical_formula_descriptive", "energy"],
        )
        
        df = dataset.to_pandas()
        df = df.set_index("immutable_id")
        
        print(f"📊 Processing {len(df)} structures...")
        
        elements_list = df["elements"].tolist()
        
        if num_processes == 1:
            # Single process - simpler and more reliable
            all_compositions = []
            for elements in tqdm(elements_list, desc="Encoding compositions"):
                all_compositions.append(one_hot_encode_composition(elements))
            all_compositions = np.array(all_compositions)
        else:
            # Multi-process
            chunk_size = len(elements_list) // num_processes
            chunks = [
                elements_list[i : i + chunk_size]
                for i in range(0, len(elements_list), chunk_size)
            ]

            with Pool(processes=num_processes) as pool:
                results = list(
                    tqdm(
                        pool.imap(process_chunk, chunks),
                        total=len(chunks),
                        desc="Processing chunks",
                    )
                )

            all_compositions = np.concatenate(results)

        # Convert to sparse matrix and save
        all_compositions = csr_matrix(all_compositions)
        scipy.sparse.save_npz(compositions_file, all_compositions)
        print(f"💾 Saved compositions to {compositions_file}")
        print(f"✅ Generated compositions matrix with shape: {all_compositions.shape}")

    return all_compositions


def filter_df(df, all_compositions, structure):
    """Filter DataFrame based on composition overlap with a structure.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to filter.
    all_compositions : scipy.sparse matrix
        One-hot encoded compositions matrix.
    structure : Structure
        pymatgen Structure to compare against.
        
    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame containing only structures with overlapping elements.
    """
    structure_vector = one_hot_encode_composition(
        structure.composition.elements
    ).reshape(-1, 1)
    forbidden_elements = 1 - structure_vector
    intersection_elements = df.loc[(all_compositions @ forbidden_elements) == 0]
    return intersection_elements


# For backward compatibility - create the global variable if needed
if __name__ == "__main__":
    all_compositions = get_all_compositions(num_processes=1)