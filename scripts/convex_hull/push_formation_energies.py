from collections import Counter
from multiprocessing import Pool, cpu_count

import pandas as pd
from datasets import load_dataset
from pymatgen.core import Composition
from tqdm import tqdm

from lemat_genbench.preprocess.reference_energies import (
    get_formation_energy_from_composition_energy,
    get_formation_energy_per_atom_from_composition_energy,
)


def process_single_row(row_data):
    try:
        composition = Composition(Counter(row_data["species_at_sites"]))
        formation_energy = get_formation_energy_from_composition_energy(
            row_data["true_energy"], composition
        )
        formation_energy_per_atom = (
            get_formation_energy_per_atom_from_composition_energy(
                row_data["true_energy"], composition
            )
        )
        return {
            "index": row_data.name,
            "formation_energy": formation_energy,
            "formation_energy_per_atom": formation_energy_per_atom,
            "success": True,
            "error": None,
        }
    except Exception as e:
        return {
            "index": row_data.name,
            "formation_energy": None,
            "formation_energy_per_atom": None,
            "success": False,
            "error": str(e),
        }


def parallel_process_dataframe(df, n_workers=None):
    if n_workers is None:
        n_workers = cpu_count()

    print(f"\nProcessing {len(df)} rows using {n_workers} workers...")

    # this should be done inside the workers themselves instead
    rows_to_process = [row for _, row in df.iterrows()]

    with Pool(processes=n_workers) as pool:
        results = list(
            tqdm(
                pool.imap(process_single_row, rows_to_process, chunksize=100),
                total=len(rows_to_process),
                desc="Calculating formation energies",
                unit=" rows",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            )
        )

    results_df = pd.DataFrame(results)

    failed_count = results_df[~results_df["success"]].shape[0]
    if failed_count > 0:
        print(f"\nWarning: {failed_count} rows failed to process")

    df_with_formation = df.copy()
    df_with_formation["formation_energy"] = results_df.set_index("index")[
        "formation_energy"
    ]
    df_with_formation["formation_energy_per_atom"] = results_df.set_index("index")[
        "formation_energy_per_atom"
    ]

    print(
        f"\nSuccessfully processed {results_df['success'].sum()} out of {len(df)} rows"
    )

    return df_with_formation


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Update a CSV file with formation energies calculated from total energies."
    )
    parser.add_argument(
        "output_csv",
        type=str,
        help="Path to save the output CSV file with formation energies.",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=None,
        help="Number of parallel workers to use. Defaults to the number of CPU cores.",
    )

    args = parser.parse_args()

    print("Loading dataset from HuggingFace...")
    mlip_energies = load_dataset("LeMaterial/LeMat-Bulk-MLIP-Energies", split="train")
    print("Converting to pandas DataFrame...")
    df = mlip_energies.to_pandas()
    print(f"Dataset loaded: {len(df):,} rows\n")

    output_csv = args.output_csv
    n_workers = args.n_workers

    if "species_at_sites" not in df.columns or "true_energy" not in df.columns:
        raise ValueError(
            "Input CSV must contain 'species_at_sites' and 'true_energy' columns."
        )

    df_with_formation = parallel_process_dataframe(df, n_workers=n_workers)

    print(f"Saving updated data to {output_csv}...")
    df_with_formation.to_csv(output_csv, index=False)

    print(f"Data successfully saved to: {output_csv}")
    print(f"Total rows in output: {len(df_with_formation):,}")
    print("\nProcess completed successfully!")
