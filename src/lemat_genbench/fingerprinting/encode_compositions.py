from multiprocessing import Pool

import numpy as np
import scipy
from datasets import load_dataset
from pymatgen.core import Element, Structure
from scipy.sparse import csr_matrix
from tqdm import tqdm


def one_hot_encode_composition(elements):
    one_hot = np.zeros(118)
    for element in elements:
        one_hot[int(Element(element).number) - 1] = 1
    return one_hot


def process_chunk(chunk):
    one_hot_compositions = []
    for elements in tqdm(chunk):
        one_hot_compositions.append(one_hot_encode_composition(elements))
    return one_hot_compositions


dataset = load_dataset(
    "LeMaterial/LeMat-Bulk",
    "compatible_pbe",
    split="train",
    columns=["elements", "immutable_id", "chemical_formula_descriptive", "energy"],
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


def get_all_compositions(num_processes=1):
    df = dataset.to_pandas()
    df = df.set_index("immutable_id")

    try:
        all_compositions = np.load("data/all_compositions.npy")
    except FileNotFoundError:
        elements_list = df["elements"].tolist()
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

        breakpoint()
        all_compositions = np.concatenate(results)
        all_compositions = csr_matrix(all_compositions)
        scipy.sparse.save_npz("data/all_compositions.npz", all_compositions)

    return all_compositions


all_compositions = get_all_compositions()


def filter_df(df, all_compositions, structure):
    structure_vector = one_hot_encode_composition(
        structure.composition.elements
    ).reshape(-1, 1)
    forbidden_elements = 1 - structure_vector
    intersection_elements = df.loc[(all_compositions @ forbidden_elements) == 0]
    return intersection_elements
