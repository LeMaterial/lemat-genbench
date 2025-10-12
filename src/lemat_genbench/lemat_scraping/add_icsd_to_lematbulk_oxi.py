import json
import re
from collections import defaultdict

import numpy as np
from smact.utils.oxidation import ICSD24OxStatesFilter


def build_icsd_oxi_state_mapping(icsd_dict):
    pattern = re.compile(r"([A-Za-z]+)(\d*)([+-])")
    oxi_state_mapping = defaultdict(list)

    for species in icsd_dict.keys():
        match = pattern.fullmatch(species)
        element, number, sign = match.groups()
        try:
            number = int(number)
        except ValueError:
            number = 1 
        if sign == "+":
            charge = number
        else:
            charge = -number
        oxi_state_mapping[element].append(charge)

    return oxi_state_mapping


def update_lemat_oxi_state_mapping(oxi_state_mapping, lemat_oxi_state_mapping, icsd_dict):

    for key in oxi_state_mapping:
        if key in lemat_oxi_state_mapping.keys():
            pass
        else:
            lemat_oxi_state_mapping[key] = oxi_state_mapping[key]
            for charge in oxi_state_mapping[key]:
                new_key = str(key)
                if charge > 0:
                    if charge > 1:
                        new_key += str(charge)
                    new_key += "+"
                if charge < 0:
                    if charge < -1: 
                        new_key += str(np.abs(charge))
                    new_key += "-"
                lemat_oxi_dict_probs[new_key] = icsd_dict[new_key]
    
    return lemat_oxi_dict_probs


def test_matching(oxi_state_mapping, lemat_oxi_dict_probs):
    with open("data/lemat_icsd_oxi_state_mapping.json", "rb") as f:
        reference_oxi_state_mapping = json.load(f) 
    with open("data/lemat_icsd_oxi_dict_probs.json", "rb") as f:
        reference_oxi_dict_probs = json.load(f) 

    if reference_oxi_state_mapping == oxi_state_mapping: 
        pass
    else: 
        raise ValueError 

    if reference_oxi_dict_probs == lemat_oxi_dict_probs:
        pass
    else:
        raise ValueError 

    return "oxidation state dictionaries and probabilities match reference! No need to update"
    

if __name__ == "__main__":
    # Initialise the oxidation state filter
    ox_filter = ICSD24OxStatesFilter()

    # Return the dataframe with non-zero results
    ox_df = ox_filter.get_species_occurrences_df(sort_by_occurrences=False)
    ox_df["species_proportion_fraction"] = ox_df["species_proportion (%)"]/100
    ox_df_subcols = ox_df[["species", "species_proportion_fraction"]]

    icsd_dict = ox_df_subcols.set_index("species")["species_proportion_fraction"].to_dict()

    with open("data/oxi_state_mapping.json", "rb") as f:
        lemat_oxi_state_mapping = json.load(f)

    with open("data/oxi_dict_probs.json", "rb") as f:
        lemat_oxi_dict_probs = json.load(f)

    icsd_oxi_state_mapping = build_icsd_oxi_state_mapping(icsd_dict)
    lemat_oxi_dict_probs = update_lemat_oxi_state_mapping(icsd_oxi_state_mapping, lemat_oxi_state_mapping, icsd_dict)    

    
    print(test_matching(lemat_oxi_state_mapping, lemat_oxi_dict_probs))

    # with open("data/icsd_oxi_dict_probs.json", "w") as f:
    #     json.dump(icsd_dict, f, indent=4)

    # with open("data/icsd_oxi_state_mapping.json", "w") as f:
    #     json.dump(oxi_state_mapping, f, indent=4)

    # with open("data/lemat_icsd_oxi_state_mapping.json", "w") as f:
    #     json.dump(lemat_oxi_state_mapping, f, indent=4)

    # with open("data/lemat_icsd_oxi_state_mapping.json", "w") as f:
    #     json.dump(lemat_oxi_state_mapping, f, indent=4)