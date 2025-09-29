import re
from collections import Counter
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element
from tqdm import tqdm


def parse_species_string(species_str):
    """Parse element symbols from a string, handling both normal and concatenated formats."""
    # Pattern matches element symbols (capital letter optionally followed by lowercase)
    pattern = r"([A-Z][a-z]?)"
    elements = re.findall(pattern, species_str)
    return elements


def get_all_unique_elements(df=None):
    all_elements = [None]
    for z in range(1, 119):
        try:
            element = Element.from_Z(z)
            all_elements.append(str(element))
        except Exception:
            all_elements.append(None)
    return all_elements


def create_composition_vector_single(args):
    idx, species_list, element_order = args
    try:
        if isinstance(species_list, str):
            import ast

            species_list = ast.literal_eval(species_list)

        parsed_species = []
        for item in species_list:
            if isinstance(item, str):
                # Check if this looks like concatenated elements (e.g., 'PP', 'CO', 'SU', 'CF')
                # If it has multiple uppercase letters, it's likely concatenated
                if len(item) >= 2 and sum(1 for c in item if c.isupper()) > 1:
                    # Parse it as potentially concatenated elements
                    parsed_species.extend(parse_species_string(item))
                elif len(item) > 2 and item[0].isupper():
                    # Longer strings that start with uppercase might also be concatenated
                    parsed_species.extend(parse_species_string(item))
                else:
                    # Normal single element
                    parsed_species.append(item)
            else:
                parsed_species.append(item)

        composition = Composition(Counter(parsed_species))
        element_amounts = composition.get_el_amt_dict()
        total_atoms = composition.num_atoms

        comp_vector = np.zeros(len(element_order))
        for i, element in enumerate(element_order):
            if element and element in element_amounts:
                comp_vector[i] = element_amounts[element] / total_atoms

        return {"index": idx, "comp_vec": comp_vector, "success": True, "error": None}

    except Exception as e:
        return {"index": idx, "comp_vec": None, "success": False, "error": str(e)}


def create_composition_vectors_parallel(df, n_workers=None):
    if n_workers is None:
        n_workers = cpu_count()

    unique_elements = get_all_unique_elements()
    print(f"Processing {len(df)} rows using {n_workers} workers...")

    species_data = df["species_at_sites"].values
    args_list = [
        (idx, species, unique_elements) for idx, species in enumerate(species_data)
    ]

    with Pool(processes=n_workers) as pool:
        results = list(
            tqdm(
                pool.imap(create_composition_vector_single, args_list, chunksize=100),
                total=len(args_list),
                desc="Creating composition vectors",
                unit=" rows",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            )
        )

    successful_results = [r for r in results if r["success"]]
    failed_results = [r for r in results if not r["success"]]

    if failed_results:
        print(f"\nWarning: {len(failed_results)} rows failed to process")

    comp_vectors = np.full((len(df), len(unique_elements)), np.nan)

    for result in successful_results:
        comp_vectors[result["index"]] = result["comp_vec"]

    print(f"Successfully processed {len(successful_results)} out of {len(df)} rows")

    return comp_vectors, unique_elements


def create_composition_vectors_simple(df):
    unique_elements = get_all_unique_elements()
    comp_vectors = []
    failed_indices = []
    print(f"Processing {len(df)} rows...")
    for idx, row in tqdm(
        df.iterrows(), total=len(df), desc="Creating composition vectors"
    ):
        try:
            species_list = row["species_at_sites"]

            if isinstance(species_list, str):
                import ast

                species_list = ast.literal_eval(species_list)

            parsed_species = []
            for item in species_list:
                if isinstance(item, str):
                    # Check if this looks like concatenated elements (e.g., 'PP', 'CO', 'SU', 'CF')
                    # If it has multiple uppercase letters, it's likely concatenated
                    if len(item) >= 2 and sum(1 for c in item if c.isupper()) > 1:
                        # Parse it as potentially concatenated elements
                        parsed_species.extend(parse_species_string(item))
                    elif len(item) > 2 and item[0].isupper():
                        # Longer strings that start with uppercase might also be concatenated
                        parsed_species.extend(parse_species_string(item))
                    else:
                        # Normal single element
                        parsed_species.append(item)
                else:
                    parsed_species.append(item)

            composition = Composition(Counter(parsed_species))
            element_amounts = composition.get_el_amt_dict()
            total_atoms = composition.num_atoms

            comp_vector = np.zeros(len(unique_elements))
            for i, element in enumerate(unique_elements):
                if element and element in element_amounts:
                    comp_vector[i] = element_amounts[element] / total_atoms

            comp_vectors.append(comp_vector)

        except Exception:
            comp_vectors.append(np.full(len(unique_elements), np.nan))
            failed_indices.append(idx)

    comp_vectors = np.array(comp_vectors)

    if failed_indices:
        print(f"Warning: {len(failed_indices)} rows failed to process")

    print(
        f"Successfully processed {len(df) - len(failed_indices)} out of {len(df)} rows"
    )

    return comp_vectors, unique_elements


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create composition vectors from materials dataset"
    )
    parser.add_argument(
        "input_csv",
        type=str,
        help="Path to input CSV file with 'species_at_sites' column",
    )
    parser.add_argument(
        "--output_npy",
        type=str,
        default="all_compositions.npy",
        help="Path to save composition vectors array",
    )
    parser.add_argument(
        "--output_elements",
        type=str,
        default="element_order.txt",
        help="Path to save element order",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=None,
        help="Number of parallel workers. Use 1 for simple processing.",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    if "species_at_sites" not in df.columns:
        raise ValueError("Input CSV must contain 'species_at_sites' column")

    # Choose processing method
    if args.n_workers == 1:
        comp_vectors, unique_elements = create_composition_vectors_simple(df)
    else:
        comp_vectors, unique_elements = create_composition_vectors_parallel(
            df, args.n_workers
        )

    np.save(args.output_npy, comp_vectors)

    with open(args.output_elements, "w") as f:
        f.write("index,symbol,atomic_number,name\n")
        for i, element in enumerate(unique_elements):
            if element:
                el = Element(element)
                f.write(f"{i},{element},{el.Z},{el.name}\n")
            else:
                f.write(f"{i},None,0,Unused\n")

    print("Process completed successfully!")
