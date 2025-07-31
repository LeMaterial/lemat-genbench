import json

import numpy as np
import pandas as pd
from datasets import load_dataset
from lemat_genbench.utils.distribution_utils import (
    generate_probabilities,
    one_hot_encode_composition,
)
from pymatgen.core import Composition
from tqdm import tqdm

if __name__ == "__main__":
    dataset_name = "Lematerial/LeMat-Bulk"
    name = "compatible_pbe"
    split = "train"
    dataset = load_dataset(dataset_name, name=name, split=split, streaming=False)

    formulas_df = pd.DataFrame(
        dataset["chemical_formula_reduced"], columns=["chemical_formula_reduced"]
    )

    comp_df = []
    for i in tqdm(range(0, len(formulas_df))):
        row = formulas_df.iloc[i]
        one_hot_output = one_hot_encode_composition(
            Composition(row.chemical_formula_reduced)
        )
        comp_df.append([one_hot_output[0], one_hot_output[1]])

    df_composition = pd.DataFrame(comp_df, columns=["CompositionCounts", "Composition"])

    composition_counts_distribution = generate_probabilities(
        df_composition, metric="CompositionCounts", metric_type=np.ndarray
    )
    composition_distribution = generate_probabilities(
        df_composition, metric="Composition", metric_type=np.ndarray
    )

    with open("data/composition_counts_distribution.json", "w") as json_file:
        json.dump(composition_counts_distribution, json_file, indent=4)

    with open("data/composition_distribution.json", "w") as json_file:
        json.dump(composition_distribution, json_file, indent=4)
