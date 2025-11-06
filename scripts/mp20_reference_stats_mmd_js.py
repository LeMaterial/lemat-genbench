from multiprocessing import Pool, cpu_count

from datasets import load_dataset
from pymatgen.core import Structure
from tqdm import tqdm
import pandas as pd 
from lemat_genbench.data.mp20_loader import mp20_item_to_structure
from lemat_genbench.utils.distribution_utils import generate_probabilities, map_space_group_to_crystal_system
from lemat_genbench.fingerprinting import one_hot_encode_composition

import pickle
import numpy as np 
import json 
import warnings 
warnings.filterwarnings("ignore")


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


if __name__ == "__main__":
    calculate_mmd = False 
    calculate_js = True 


    dataset = pd.read_csv("mp-20-data/mp_20.csv")
    if calculate_mmd: 
        results = {
            "Volume": [],
            "Density(g/cm^3)": [],
            "Density(atoms/A^3)": []
        }

        for i in tqdm(range(0, len(dataset))):
            strut = mp20_item_to_structure(dataset.iloc[i])

            # Structural properties
            g_cm3_density = strut.density
            volume = strut.volume
            num_atoms = len(strut)
            atomic_density = num_atoms / volume

            results["Volume"].append(volume)
            results["Density(g/cm^3)"].append(g_cm3_density)
            results["Density(atoms/A^3)"].append(atomic_density)

        with open("data/mp20_mmd_values.pkl", "wb") as f: 
            pickle.dump(results, f)    

    
        indices = np.random.choice(len(dataset), 15000, replace=False)
        indices.sort()  
        np.save('data/mp20_mmd_sample_indices_15k.npy', indices)

    if calculate_js: 
        comp_df = []
        for i in tqdm(range(0, len(dataset))):
            strut = mp20_item_to_structure(dataset.iloc[i])
            space_group = strut.get_space_group_info()[1]
            one_hot_output = one_hot_encode_composition(
                strut.composition
            )
            comp_df.append([one_hot_output[0], one_hot_output[1], space_group, map_space_group_to_crystal_system(space_group)])

        df_composition = pd.DataFrame(comp_df, columns=["CompositionCounts", "Composition", "SpaceGroup", "CrystalSystem"])

        composition_counts_distribution = generate_probabilities(
            df_composition, metric="CompositionCounts", metric_type=np.ndarray
        )
        composition_distribution = generate_probabilities(
            df_composition, metric="Composition", metric_type=np.ndarray
        )
        space_group_distribution = generate_probabilities(
            df_composition, metric="SpaceGroup", metric_type=np.ndarray
        )
        crystal_system_distribution = generate_probabilities(
            df_composition, metric="CrystalSystem", metric_type=np.ndarray
        )

        jsdistance_dict = {
        "CompositionCounts": composition_counts_distribution,
        "Composition": composition_distribution,
        "SpaceGroup": space_group_distribution,
        "CrystalSystem": crystal_system_distribution
        }

        with open('data/mp20_jsdistance_distributions.json', 'w') as f:
            json.dump(jsdistance_dict, f, indent=4)