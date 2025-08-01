import json
from collections import defaultdict
from itertools import combinations_with_replacement, product
from pathlib import Path

import numpy as np
from pymatgen.core.periodic_table import Element, Species
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def compositional_oxi_state_guesses(
    comp,
    all_oxi_states: bool,
    max_sites: int | None,
    oxi_states_override: dict[str, list] | None,
    target_charge: float,
) -> tuple[tuple, tuple, tuple]:
    """Utility operation for guessing oxidation states. Adapted from the _get_oxi_state_guesses function from Pymatgen.core.Composition

    See `oxi_state_guesses` for full details. This operation does the
    calculation of the most likely oxidation states

    Args:
        comp: A Pymatgen composition object.
        oxi_states_override (dict): dict of str->list to override an element's common oxidation states, e.g.
            {"V": [2,3,4,5]}.
        target_charge (float): the desired total charge on the structure. Default is 0 signifying charge balance.
        all_oxi_states (bool): if True, all oxidation states of an element, even rare ones, are used in the search
            for guesses. However, the full oxidation state list is *very* inclusive and can produce nonsensical
            results. If False, the icsd_oxidation_states list is used when present, or the common_oxidation_states
            is used when icsd_oxidation_states is not present. These oxidation states lists comprise more
            commonly occurring oxidation states and results in more reliable guesses, albeit at the cost of
            missing some uncommon situations. The default is False.
        max_sites (int): if possible, will reduce Compositions to at most
            this many sites to speed up oxidation state guesses. If the
            composition cannot be reduced to this many sites a ValueError
            will be raised. Set to -1 to just reduce fully. If set to a
            number less than -1, the formula will be fully reduced but a
            ValueError will be thrown if the number of atoms in the reduced
            formula is greater than abs(max_sites).

    Returns:
        list[dict]: Each dict maps the element symbol to a list of
            oxidation states for each site of that element. For example, Fe3O4 could
            return a list of [2,2,2,3,3,3] for the oxidation states of the 6 Fe sites.
            If the composition is not charge balanced, an empty list is returned.
    """
    # Reduce Composition if necessary
    if max_sites and max_sites < 0:
        comp = comp.reduced_composition

        if max_sites < -1 and comp.num_atoms > abs(max_sites):
            raise ValueError(
                f"Composition {comp} cannot accommodate max_sites setting!"
            )

    elif max_sites and comp.num_atoms > max_sites:
        reduced_comp, reduced_factor = comp.get_reduced_composition_and_factor()
        if reduced_factor > 1:
            reduced_comp *= max(1, int(max_sites / reduced_comp.num_atoms))
            comp = reduced_comp  # as close to max_sites as possible
        if comp.num_atoms > max_sites:
            raise ValueError(
                f"Composition {comp} cannot accommodate max_sites setting!"
            )

    # Load prior probabilities of oxidation states, used to rank solutions

    here = Path(__file__).resolve().parent
    three_up = here.parents[2]
    with open(three_up / "data" / "oxi_dict_probs.json", "r") as f:
        loaded_dict = json.load(f)

    type(comp).oxi_prob = loaded_dict
    oxi_states_override = oxi_states_override or {}
    # Assert Composition only has integer amounts
    if not all(amt == int(amt) for amt in comp.values()):
        raise ValueError(
            "Charge balance analysis requires integer values in Composition!"
        )

    # For each element, determine all possible sum of oxidations
    # (taking into account nsites for that particular element)
    el_amt = comp.get_el_amt_dict()
    elements = list(el_amt)
    el_sums: list = []  # matrix: dim1= el_idx, dim2=possible sums
    el_sum_scores: defaultdict = defaultdict(set)  # dict of el_idx, sum -> score
    el_best_oxid_combo: dict = {}  # dict of el_idx, sum -> oxid combo with best score
    for idx, el in enumerate(elements):
        el_sum_scores[idx] = {}
        el_best_oxid_combo[idx] = {}
        el_sums.append([])
        if oxi_states_override.get(el):
            oxids: list | tuple = oxi_states_override[el]
        elif all_oxi_states:
            oxids = Element(el).oxidation_states
        else:
            oxids = (
                Element(el).icsd_oxidation_states or Element(el).common_oxidation_states
            )

        # Get all possible combinations of oxidation states
        # and sum each combination
        for oxid_combo in combinations_with_replacement(oxids, int(el_amt[el])):
            # List this sum as a possible option
            oxid_sum = sum(oxid_combo)
            if oxid_sum not in el_sums[idx]:
                el_sums[idx].append(oxid_sum)

            # Determine how probable is this combo?
            scores = []
            for o in oxid_combo:
                scores.append(type(comp).oxi_prob[str(Species(el, o))])
            score = np.mean(scores)
            # If it is the most probable combo for a certain sum,
            # store the combination
            if oxid_sum not in el_sum_scores[idx] or score > el_sum_scores[idx].get(
                oxid_sum, 0
            ):
                if max(oxid_combo) - min(oxid_combo) > 1:
                    pass
                else:
                    el_sum_scores[idx][oxid_sum] = score
                    el_best_oxid_combo[idx][oxid_sum] = oxid_combo

    # Determine which combination of oxidation states for each element
    # is the most probable
    all_sols = []  # will contain all solutions
    all_oxid_combo = []  # will contain the best combination of oxidation states for each site
    all_scores = []  # will contain a score for each solution
    for x in product(*el_sums):
        # Each x is a trial of one possible oxidation sum for each element
        if sum(x) == target_charge:  # charge balance condition
            el_sum_sol = dict(zip(elements, x, strict=True))  # element->oxid_sum
            # Normalize oxid_sum by amount to get avg oxid state
            sol = {el: v / el_amt[el] for el, v in el_sum_sol.items()}
            # Add the solution to the list of solutions
            all_sols.append(sol)

            # Determine the score for this solution
            scores = []
            for idx, v in enumerate(x):
                scores.append(el_sum_scores[idx][v])
            # the score is the minimum of the scores of each of the oxidation states in the composition - the goal is to find a charge
            # balanced oxidation state which limits the occurance of very uncommon oxidation states
            all_scores.append(np.mean(scores))
            # Collect the combination of oxidation states for each site
            all_oxid_combo.append(
                {
                    e: el_best_oxid_combo[idx][v]
                    for idx, (e, v) in enumerate(zip(elements, x, strict=True))
                }
            )
    # Sort the solutions from highest to lowest score
    if all_scores:
        all_sols, all_oxid_combo = zip(
            *(
                (y, x)
                for (z, y, x) in sorted(
                    zip(all_scores, all_sols, all_oxid_combo, strict=True),
                    key=lambda pair: pair[0],
                    reverse=True,
                )
            ),
            strict=True,
        )
    return (
        tuple(all_sols),
        tuple(all_oxid_combo),
        tuple(sorted(all_scores, reverse=True)),
    )


def get_inequivalent_site_info(structure):
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


def build_oxi_dict(df):
    oxi_dict = {}
    for i in range(0, len(df)):
        row = df.iloc[i]
        if row.ValencesCalculated == True:
            for key, value in np.asarray(
                [row.Sites["species"], row.Sites["multiplicities"]]
            ).T:
                if key in oxi_dict:
                    oxi_dict[key] += int(value)
                else:
                    oxi_dict[key] = int(value)
    return oxi_dict


def build_sorted_oxi_dict(oxi_dict_sorted):
    oxi_dict_counts = {}
    for key in oxi_dict_sorted.keys():
        try:
            int(key[1])
            el = key[0]
        except ValueError:
            if key[1] in ["+", "-"]:
                el = key[0]
            else:
                el = key[0:2]

        if el in oxi_dict_counts:
            oxi_dict_counts[el] += int(oxi_dict_sorted[key])
        else:
            oxi_dict_counts[el] = int(oxi_dict_sorted[key])
    return oxi_dict_counts


def build_oxi_dict_probs(oxi_dict_sorted, oxi_dict_counts):
    oxi_dict_probs = oxi_dict_sorted
    for key in oxi_dict_probs.keys():
        try:
            int(key[1])
            el = key[0]
        except ValueError:
            if key[1] in ["+", "-"]:
                el = key[0]
            else:
                el = key[0:2]
        denom = oxi_dict_counts[el]
        oxi_dict_probs[key] = oxi_dict_probs[key] / denom
    return oxi_dict_probs


def oxi_state_map(oxidation_state):
    try:
        int(oxidation_state[1])
        el = oxidation_state[0]
        ox = int(oxidation_state[1:3][::-1])
        return ox, el
    except ValueError:
        if oxidation_state[1] in ["+", "-"]:
            ox = sign_to_int(oxidation_state[1])
            el = oxidation_state[0]
            return ox, el
        elif oxidation_state[2] in ["+", "-"]:
            ox = sign_to_int(oxidation_state[2])
            el = oxidation_state[0:2]
            return ox, el
        else:
            ox = int(oxidation_state[2:4][::-1])
            el = oxidation_state[0:2]
            return ox, el


def build_oxi_state_map(oxi_dict_sorted):
    oxi_state_mapping = {}
    for key in oxi_dict_sorted.keys():
        ox, el = oxi_state_map(key)
        if el in oxi_state_mapping:
            oxi_state_mapping[el].append(ox)
        else:
            oxi_state_mapping[el] = [ox]
    return oxi_state_mapping


def sign_to_int(char):
    return {"+": 1, "-": -1}.get(char, 0)  # default to 0 if unexpected
