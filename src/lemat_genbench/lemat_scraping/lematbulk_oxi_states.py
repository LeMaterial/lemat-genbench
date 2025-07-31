from multiprocessing import Pool, cpu_count

import numpy as np
from datasets import load_dataset
from func_timeout import FunctionTimedOut, func_timeout
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tqdm import tqdm


def get_inequivalent_site_info(structure: Structure) -> dict:
    """Gets the symmetrically inequivalent sites as found by the
    SpacegroupAnalyzer class from Pymatgen.

    Parameters
    ----------
    structure : pymatgen.core.structure.Structure
        The Pymatgen structure of interest.

    Returns
    -------
    dict
        A dictionary containing three lists, one of the inequivalent sites, one
        for the atom types they correspond to and the last for the multiplicity.
    """

    # Get the symmetrically inequivalent indexes
    inequivalent_sites = (
        SpacegroupAnalyzer(structure).get_symmetrized_structure().equivalent_indices
    )

    # Equivalent indexes must all share the same atom type
    multiplicities = [len(xx) for xx in inequivalent_sites]
    inequivalent_sites = [xx[0] for xx in inequivalent_sites]
    species = [str(structure[xx].specie) for xx in inequivalent_sites]

    return {
        "sites": inequivalent_sites,
        "species": species,
        "multiplicities": multiplicities,
    }


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


def map_space_group_to_crystal_system(space_group: int) -> str:
    """Map a space group number to a crystal system.

    Crystal systems are defined as follows:
    - Triclinic: 2 <= space_group <= 15
    - Monoclinic: 15 < space_group <= 74
    - Orthorhombic: 74 < space_group <= 142
    - Tetragonal: 142 < space_group <= 167
    - Trigonal: 167 < space_group <= 194
    - Hexagonal: 194 < space_group <= 230

    Parameters
    ----------
    space_group : int
        The space group number to map.

    Returns
    -------
    str
        The crystal system corresponding to the space group number.

    Raises
    ------
    ValueError
        If the space group number is not in the range of valid space group numbers.
    """
    if space_group <= 2 and space_group > 0:
        return "triclinic"
    elif space_group <= 15 and space_group > 2:
        return "monoclinic"
    elif space_group <= 74 and space_group > 15:
        return "orthorhombic"
    elif space_group <= 142 and space_group > 74:
        return "tetragonal"
    elif space_group <= 167 and space_group > 142:
        return "trigonal"
    elif space_group <= 194 and space_group > 167:
        return "hexagonal"
    elif space_group <= 230 and space_group > 194:
        return "cubic"
    else:
        raise ValueError


def process_item_action(item):
    """Extracts the density of a crystal (in units of atoms/volume)
    and adds it to a list that will eventually include
    all LeMat-Bulk crystals. However, it can be amended to find any associated value
    (e.g. oxidation state) by editing the function and the accumulation list.

    Parameters
    ----------
    item : dict
        The item to process.

    Returns
    -------
    list[Any]
        The result of the processing of the item.
    """
    LeMatID = item["immutable_id"]
    strut = lematbulk_item_to_structure(item)
    bv = BVAnalyzer()
    valences_calculated = False
    sites = None
    species = None

    # try:
    try:
        strut = bv.get_oxi_state_decorated_structure(strut)
        valences_calculated = True
    except ValueError:
        pass

    species = strut.species
    sites = get_inequivalent_site_info(strut)

    return [LeMatID, sites, species, valences_calculated]


def process_item_wrapper(item):
    """Process an item.

    Parameters
    ----------
    item : dict
        The item to process.

    Returns
    -------
    list[Any]
        The result of the processing of the item.
    """
    LeMatID = item["immutable_id"]
    try:
        result = func_timeout(
            15, process_item_action, [item]
        )  # TODO This should not be hardcoded!
        return result
    except FunctionTimedOut:
        print("Function timed out and was skipped")
        return [LeMatID, 1, 1, "TimedOut>15Sec"]


def process_items_parallel(dataset, chunk_size=1000, num_workers=None):
    """Process items in parallel.

    Parameters
    ----------
    dataset : Dataset
        The dataset to process.
    chunk_size : int, optional
        The size of the chunks to process.
    num_workers : int, optional
        The number of workers to use.

    Yields
    ------
    list[list[Any]]
        A list of lists, each containing the results of the processing of a chunk of the dataset.
    """
    if num_workers is None:
        num_workers = cpu_count()

    def chunks():
        for i in range(0, len(dataset), chunk_size):
            chunk = [dataset[j] for j in range(i, min(i + chunk_size, len(dataset)))]
            for item in chunk:
                yield item

    total_items = len(dataset)
    with Pool(processes=num_workers) as pool:
        # Using imap instead of map for memory efficiency
        with tqdm(total=total_items, desc="Processing structures") as pbar:
            for result in pool.imap(process_item_wrapper, chunks(), chunksize=100):
                pbar.update(1)
                yield result


if __name__ == "__main__":
    import pandas as pd

    dataset_name = "Lematerial/LeMat-Bulk"
    name = "compatible_pbe"
    split = "train"
    dataset = load_dataset(dataset_name, name=name, split=split, streaming=False)
    target_columns = ["LeMatID", "Sites", "Species", "ValencesCalculated"]

    for i in np.arange(0, 5_400_000, 100_000):
        dataset_temp = dataset.select(range(i, min(len(dataset), i + 100_000)))

        # Process and handle results as they come
        print(
            f"Processing {len(dataset_temp)} structures using {cpu_count()} workers..."
        )
        results = []
        for result in process_items_parallel(dataset_temp):
            results.append(result)

        print("Creating DataFrame and saving to pkl...")

        df = pd.DataFrame(results, columns=target_columns)
        df.to_pickle("lematbulk_oxi_full_" + str(i) + ".pkl")
