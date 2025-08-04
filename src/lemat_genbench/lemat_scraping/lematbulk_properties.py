from multiprocessing import Pool, cpu_count

from datasets import load_dataset
from pymatgen.core import Structure
from tqdm import tqdm

from lemat_genbench.utils.distribution_utils import (
    map_space_group_to_crystal_system,
    one_hot_encode_composition,
)

# Crystal System Mapping Reference
# ================================
# Integer | Crystal System | Space Group Range
# --------|----------------|------------------
#    1    | Triclinic      | 1-2
#    2    | Monoclinic     | 3-15  
#    3    | Orthorhombic   | 16-74
#    4    | Tetragonal     | 75-142
#    5    | Trigonal       | 143-167
#    6    | Hexagonal      | 168-194
#    7    | Cubic          | 195-230


def lematbulk_item_to_structure(item: dict):
    sites = item["species_at_sites"]
    coords = item["cartesian_site_positions"]
    cell = item["lattice_vectors"]

    structure = Structure(
        species=sites, coords=coords, lattice=cell, coords_are_cartesian=True
    )

    return structure


def process_item(item):
    """
    This function extracts structural and compositional properties of a crystal
    for distribution analysis.
    """

    LeMatID = item["immutable_id"]
    strut = lematbulk_item_to_structure(item)

    # Structural properties
    g_cm3_density = strut.density
    volume = strut.volume
    num_atoms = len(strut)
    atomic_density = num_atoms / volume
    space_group = strut.get_space_group_info()[1]
    crystal_system = map_space_group_to_crystal_system(space_group)
    
    # Compositional properties
    one_hot_vectors = one_hot_encode_composition(strut.composition)
    composition_counts = one_hot_vectors[0]  # Element counts
    composition = one_hot_vectors[1]  # Element presence/absence

    return [
        LeMatID,
        volume,
        round(g_cm3_density, 2),
        round(atomic_density, 2),
        space_group,
        crystal_system,
        composition_counts,
        composition,
    ]


def process_items_parallel(dataset, chunk_size=10000, num_workers=None):
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
            for result in pool.imap(process_item, chunks(), chunksize=1000):
                pbar.update(1)
                yield result


if __name__ == "__main__":
    dataset_name = "Lematerial/LeMat-Bulk"
    name = "compatible_pbe"
    split = "train"
    dataset = load_dataset(dataset_name, name=name, split=split, streaming=False)
    # Process and handle results as they come
    print(f"Processing {len(dataset)} structures using {cpu_count()} workers...")
    results = []
    for result in process_items_parallel(dataset):
        results.append(result)

    print("Creating DataFrame and saving to CSV...")
    import pandas as pd

    df = pd.DataFrame(
        results,
        columns=[
            "LeMatID",
            "Volume",
            "Density(g/cm^3)",
            "Density(atoms/A^3)",
            "SpaceGroup",
            "CrystalSystem",
            "CompositionCounts",
            "Composition",
        ],
    )
    df.to_pickle("data/lematbulk_full_distribution_properties.pkl")
