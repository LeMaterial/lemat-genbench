import os
import pickle
from multiprocessing import Pool

import numpy as np
from datasets import load_dataset
from material_hasher.similarity.pdd import PointwiseDistanceDistributionSimilarity
from pymatgen.core import Structure
from tqdm import tqdm


def process_chunk(chunk):
    hashes = []
    save_every = 300000
    pdd = PointwiseDistanceDistributionSimilarity()
    dataset = load_dataset(
        "LeMaterial/LeMat-Bulk",
        "compatible_pbe",
        split="train",
        columns=[
            "elements",
            "immutable_id",
            "chemical_formula_descriptive",
            "species_at_sites",
            "cartesian_site_positions",
            "lattice_vectors",
        ],
    ).select(chunk)

    j = 0
    for i, item in tqdm(enumerate(dataset)):
        structure = lematbulk_item_to_structure(item)
        hashes.append(pdd.get_material_hash(structure))

        if i % save_every == 0 and i > 0:
            print(f"Saving hashes at index {i}...")
            os.makedirs("/ogre/pdd_hashes", exist_ok=True)
            pickle.dump(
                hashes,
                open(f"/ogre/pdd_hashes/pdd_hashes_chunk_{min(chunk)}_{j}.pkl", "wb"),
            )
            j += 1
            hashes = []

    print(f"Saving hashes at index {i}...")
    os.makedirs("/ogre/pdd_hashes", exist_ok=True)
    pickle.dump(
        hashes, open(f"/ogre/pdd_hashes/pdd_hashes_chunk_{min(chunk)}_{j}.pkl", "wb")
    )
    hashes = []
    return []


dataset = load_dataset(
    "LeMaterial/LeMat-Bulk",
    "compatible_pbe",
    split="train",
    columns=[
        "elements",
        "immutable_id",
        "chemical_formula_descriptive",
        "species_at_sites",
        "cartesian_site_positions",
        "lattice_vectors",
    ],
)


def lematbulk_item_to_structure(item: dict) -> Structure:
    """Convert a LeMat-Bulk item to a pymatgen Structure object.

    Parameters
    ----------
    item : dict
        The item to convert.

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


num_processes = 8
chunks_indices = np.array_split(np.arange(len(dataset)), num_processes)

with Pool(processes=num_processes) as pool:
    results = list(
        tqdm(
            pool.imap(process_chunk, chunks_indices),
            total=len(chunks_indices),
            desc="Processing chunks",
        )
    )
