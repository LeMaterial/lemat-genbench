import json

import numpy as np
import pandas as pd
from datasets import load_dataset
from pymatgen.core import Composition
from tqdm import tqdm

from lemat_genbench.utils.distribution_utils import (
    generate_probabilities,
    one_hot_encode_composition,
)

if __name__ == "__main__":
    dataset = pd.read_csv("mp-20-data/mp_20.csv")

    comp_df = []
    for i in tqdm(range(0, len(dataset))):
        row = dataset.iloc[i]
        one_hot_output = one_hot_encode_composition(
            Composition(row.pretty_formula)
        )
        comp_df.append([one_hot_output[0], one_hot_output[1]])

    df_composition = pd.DataFrame(comp_df, columns=["CompositionCounts", "Composition"])

    composition_counts_distribution = generate_probabilities(
        df_composition, metric="CompositionCounts", metric_type=np.ndarray
    )
    composition_distribution = generate_probabilities(
        df_composition, metric="Composition", metric_type=np.ndarray
    )

    with open("data/mp20_composition_counts_distribution.json", "w") as json_file:
        json.dump(composition_counts_distribution, json_file, indent=4)

    with open("data/mp20_composition_distribution.json", "w") as json_file:
        json.dump(composition_distribution, json_file, indent=4)
