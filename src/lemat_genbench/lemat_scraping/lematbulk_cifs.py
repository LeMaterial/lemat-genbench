
import numpy as np
from datasets import load_dataset
from pymatgen.core import Structure
from tqdm import tqdm


def lematbulk_item_to_structure(item: dict):
    sites = item["species_at_sites"]
    coords = item["cartesian_site_positions"]
    cell = item["lattice_vectors"]

    structure = Structure(
        species=sites, coords=coords, lattice=cell, coords_are_cartesian=True
    )

    return structure


if __name__ == "__main__":
    dataset_name = "Lematerial/LeMat-Bulk"
    name = "compatible_pbe"
    split = "train"
    dataset = load_dataset(dataset_name, name=name, split=split, streaming=False)
    np.random.seed(32)
    indicies = np.random.randint(0, len(dataset), 600000)
    for i in tqdm(range(len(indicies))):
        # print(index)
        index = int(indicies[i])
        strut = lematbulk_item_to_structure(dataset[index])
        name = dataset[index]["immutable_id"]
        strut.to(filename="data/lematbulk_cifs/"+name+".cif")

 
