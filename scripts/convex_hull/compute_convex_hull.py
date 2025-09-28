#!/usr/bin/env python3

import argparse
import pickle
import time
import warnings
from collections import Counter
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from datasets import load_dataset
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition
from tqdm import tqdm

warnings.filterwarnings("ignore")

df_main = None
maximal_sets = None
compositions_binary = None


def parse_species_string(species_str):
    if isinstance(species_str, str):
        species_str = species_str.strip("[]")
        species = species_str.replace("'", "").split()
        return species
    else:
        return list(species_str)


def process_maximal_set(set_data, energy_column, hull_energy_column="true_energy"):
    set_idx, maximal_composition = set_data

    is_subset = ~np.any(
        compositions_binary @ (1 - maximal_composition[:, None]), axis=1
    )
    material_indices = np.where(is_subset)[0]

    if len(material_indices) < 2:
        return {}

    materials_in_set = []
    all_elements = set()

    for mat_idx in material_indices:
        row = df_main.iloc[mat_idx]
        species = parse_species_string(row["species_at_sites"])
        all_elements.update(set(species))

        materials_in_set.append(
            {
                "immutable_id": row["immutable_id"],
                "species": species,
                "true_energy": row["true_energy"],
                energy_column: row[energy_column],
            }
        )

    hull_values = {}
    try:
        pd_entries = []
        material_map = {}
        for mat in materials_in_set:
            composition = Composition(Counter(mat["species"]))
            energy = mat[hull_energy_column]

            if pd.isna(energy):
                continue

            entry = PDEntry(composition, energy)
            pd_entries.append(entry)
            material_map[id(entry)] = mat

        if len(pd_entries) < 2:
            return hull_values

        pd_phase = PhaseDiagram(pd_entries)

        for entry in pd_entries:
            if id(entry) not in material_map:
                continue

            mat = material_map[id(entry)]
            test_energy = mat[energy_column]

            if pd.isna(test_energy):
                continue

            test_entry = PDEntry(entry.composition, test_energy)
            e_above_hull = pd_phase.get_decomp_and_e_above_hull(
                test_entry, allow_negative=True
            )[1]
            hull_values[mat["immutable_id"]] = e_above_hull

    except Exception as e:
        print(f"Error in phase diagram: {e}")
        pass

    return hull_values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--energy-type",
        type=str,
        choices=["dft", "orb", "uma", "mace_mp", "mace_omat", "all"],
        default="dft",
    )
    parser.add_argument(
        "--hull-energy-type",
        type=str,
        choices=["dft", "orb", "uma", "mace_mp", "mace_omat"],
        default=None,
    )
    parser.add_argument("--n-workers", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--input-csv", type=str, default=None)
    parser.add_argument("--maximal-sets", type=str, default="maximal_chemical_sets.npy")
    parser.add_argument("--compositions", type=str, default="all_compositions.npy")
    parser.add_argument(
        "--debug", action="store_true", help="Run in debug mode without multiprocessing"
    )

    args = parser.parse_args()

    global df_main, maximal_sets, compositions_binary

    if args.input_csv is None:
        mlip_energies = load_dataset(
            "LeMaterial/LeMat-Bulk-MLIP-Energies", split="train"
        )
        print("Converting to pandas DataFrame...")
        df_main = mlip_energies.to_pandas()
    else:
        print(f"Loading {args.input_csv}...")
        df_main = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df_main):,} materials.")

    print(f"Loading {args.maximal_sets}...")
    maximal_sets = np.load(args.maximal_sets, allow_pickle=True)
    print(f"Loaded {len(maximal_sets):,} maximal chemical sets")

    print(f"Loading {args.compositions}...")
    compositions = np.load(args.compositions, allow_pickle=True)
    print(f"Loaded compositions matrix with shape {compositions.shape}")

    print("Converting compositions to binary...")
    compositions_binary = (compositions > 0).astype(np.uint8)
    print("Binary conversion complete")

    np.random.seed(42)
    indices = np.arange(len(maximal_sets))
    np.random.shuffle(indices)

    sets_to_process = [(i, maximal_sets[i]) for i in indices]

    if args.batch_size:
        sets_to_process = sets_to_process[: args.batch_size]
        print(f"Processing {args.batch_size} randomly selected maximal sets")

    type_to_column = {
        "dft": "true_energy",
        "orb": "orb_energy",
        "uma": "uma_energy",
        "mace_mp": "mace_mp_energy",
        "mace_omat": "mace_omat_energy",
    }

    if args.energy_type == "all":
        if args.output is None:
            args.output = "all_hull_energies.parquet"

        all_results = {}
        for energy_type, column_name in type_to_column.items():
            print(f"\nProcessing {energy_type.upper()} energies...")
            hull_energy_column = (
                type_to_column[args.hull_energy_type]
                if args.hull_energy_type
                else column_name
            )
            process_func = partial(
                process_maximal_set,
                energy_column=column_name,
                hull_energy_column=hull_energy_column,
            )

            hull_values = {}
            with Pool(args.n_workers) as pool:
                results = list(
                    tqdm(
                        pool.imap_unordered(process_func, sets_to_process),
                        total=len(sets_to_process),
                        desc=f"  {energy_type.upper()}",
                    )
                )

            for result in results:
                hull_values.update(result)

            all_results[f"{energy_type}_hull"] = hull_values

            if hull_values:
                values = list(hull_values.values())
                print(
                    f"  Computed {len(hull_values):,} hull energies - "
                    f"mean={np.mean(values):.4f}, on_hull={sum(1 for v in values if abs(v) < 0.001):,}"
                )

        df_results = pd.DataFrame(all_results)
        df_results.to_parquet(args.output)
        print(f"\nSaved all hull energies to {args.output}")

    else:
        energy_column = type_to_column[args.energy_type]

        if args.output is None:
            args.output = f"{args.energy_type}_hull.pickle"

        print(f"\nComputing hull energies for {args.energy_type.upper()}")
        if args.debug:
            print("DEBUG MODE: Running without multiprocessing")

        hull_energy_column = (
            type_to_column[args.hull_energy_type]
            if args.hull_energy_type
            else energy_column
        )
        process_func = partial(
            process_maximal_set,
            energy_column=energy_column,
            hull_energy_column=hull_energy_column,
        )

        hull_values = {}

        if args.debug:
            for i, set_data in enumerate(
                tqdm(sets_to_process, desc="Processing maximal sets")
            ):
                print(
                    f"\nDebug: Processing set {i + 1}/{len(sets_to_process)} - maximal set index {set_data[0]}"
                )

                t0 = time.time()
                result = process_maximal_set(
                    set_data, energy_column, hull_energy_column
                )
                t1 = time.time()
                hull_values.update(result)
                print(f"  Found {len(result)} hull values in {t1 - t0:.2f}s")
        else:
            with Pool(args.n_workers) as pool:
                results = list(
                    tqdm(
                        pool.imap_unordered(process_func, sets_to_process),
                        total=len(sets_to_process),
                        desc="Processing maximal sets",
                    )
                )

            for result in results:
                hull_values.update(result)

        print(f"\nComputed hull energies for {len(hull_values):,} materials")

        with open(args.output, "wb") as f:
            pickle.dump(hull_values, f)

        if hull_values:
            values = list(hull_values.values())
            print(
                f"E_above_hull: min={min(values):.4f}, max={max(values):.4f}, "
                f"mean={np.mean(values):.4f}, median={np.median(values):.4f} eV/atom"
            )
            print(
                f"On hull (<0.001): {sum(1 for v in values if abs(v) < 0.001):,} "
                f"({100 * sum(1 for v in values if abs(v) < 0.001) / len(values):.1f}%)"
            )


if __name__ == "__main__":
    main()
