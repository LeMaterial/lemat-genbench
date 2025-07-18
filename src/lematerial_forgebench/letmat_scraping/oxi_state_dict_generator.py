import json

import pandas as pd

from lematerial_forgebench.utils.oxidation_state import (
    build_oxi_dict,
    build_oxi_dict_probs,
    build_oxi_state_map,
    build_sorted_oxi_dict,
)

if __name__ == "__main__":
    loaded_df = pd.read_pickle("data\lematbulk_oxi_no_strut.pkl")
    oxi_dict = build_oxi_dict(loaded_df)

    oxi_dict_sorted = dict(sorted(oxi_dict.items()))

    oxi_dict_counts = build_sorted_oxi_dict(oxi_dict_sorted)

    oxi_dict_probs = build_oxi_dict_probs(oxi_dict_sorted, oxi_dict_counts)

    oxi_state_mapping = build_oxi_state_map(oxi_dict_sorted)

    with open("data/oxi_dict_probs.json", "w") as f:
        json.dump(oxi_dict_probs, f, indent=2)

    with open("data/oxi_state_mapping.json", "w") as f:
        json.dump(oxi_state_mapping, f, indent=2)
