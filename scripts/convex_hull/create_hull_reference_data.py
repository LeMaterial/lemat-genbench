#!/usr/bin/env python3
"""Create filtered datasets of materials close to the convex hull."""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, create_repo
from scipy import sparse


def create_composition_matrix(df, element_order_file="element_order.txt"):
    """Create composition matrix from dataframe."""
    element_to_idx = {}

    if os.path.exists(element_order_file):
        with open(element_order_file, "r") as f:
            for line in f.readlines()[1:]:
                parts = line.strip().split(",")
                if len(parts) >= 2 and parts[1] != "None":
                    element_to_idx[parts[1]] = int(parts[0])
    else:
        from pymatgen.core import Element

        for z in range(1, 119):
            try:
                elem = Element.from_Z(z)
                element_to_idx[str(elem)] = z
            except Exception:
                pass

    num_elements = 119
    num_materials = len(df)
    rows, cols, data = [], [], []

    for mat_idx, row in enumerate(df.itertuples()):
        species_str = row.species_at_sites
        if isinstance(species_str, str):
            species = species_str.strip("[]").replace("'", "").split()
        else:
            species = list(species_str)

        from collections import Counter

        element_counts = Counter(species)
        total_atoms = sum(element_counts.values())

        for element, count in element_counts.items():
            if element in element_to_idx:
                rows.append(mat_idx)
                cols.append(element_to_idx[element])
                data.append(count / total_atoms)

    comp_matrix = sparse.csr_matrix(
        (data, (rows, cols)), shape=(num_materials, num_elements)
    )

    return comp_matrix, element_to_idx


def main():
    parser = argparse.ArgumentParser(
        description="Create filtered hull reference datasets"
    )
    parser.add_argument("--csv", default="hello.csv", help="Input CSV file")
    parser.add_argument(
        "--hull", default="all_hull_energies.parquet", help="Hull energies file"
    )
    parser.add_argument(
        "--output-dir", default="data/convex_hulls", help="Output directory"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.01, help="Threshold (eV/atom)"
    )
    parser.add_argument(
        "--energy-types",
        nargs="+",
        default=["dft", "orb", "uma", "mace_mp", "mace_omat"],
        help="Energy types",
    )
    parser.add_argument(
        "--element-order", default="element_order.txt", help="Element order file"
    )
    parser.add_argument("--upload", action="store_true", help="Upload to HuggingFace")
    parser.add_argument(
        "--hf-repo", default="LeMaterial/LeMat-Bulk-MLIP-Hull", help="HF repo name"
    )
    parser.add_argument("--hf-token", default=None, help="HF API token")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    print(f"\nLoading {args.csv}...")
    df_full = pd.read_csv(args.csv)
    print(f"Loaded {len(df_full):,} materials")

    print(f"\nLoading {args.hull}...")

    if args.hull.endswith(".parquet"):
        hull_df = pd.read_parquet(args.hull)
        hull_df = hull_df.reset_index()
        hull_df.rename(columns={"index": "immutable_id"}, inplace=True)

        hull_columns = [col for col in hull_df.columns if "hull" in col.lower()]
        print(f"Found hull columns: {hull_columns}")

        for energy_type in args.energy_types:
            hull_column = f"{energy_type}_hull"

            if hull_column not in hull_columns:
                print(f"\nSkipping {energy_type}: column {hull_column} not found")
                continue

            print(f"\n{'=' * 50}")
            print(f"Processing {energy_type.upper()} hull energies")
            print(f"{'=' * 50}")

            hull_subset = hull_df[["immutable_id", hull_column]].dropna(
                subset=[hull_column]
            )
            hull_subset.rename(columns={hull_column: "e_above_hull"}, inplace=True)

            process_energy_type(
                df_full,
                hull_subset,
                energy_type,
                args.threshold,
                output_dir,
                args.element_order,
            )

    else:
        with open(args.hull, "rb") as f:
            hull_energies = pickle.load(f)

        hull_df = pd.DataFrame.from_dict(
            hull_energies, orient="index", columns=["e_above_hull"]
        )
        hull_df.index.name = "immutable_id"
        hull_df = hull_df.reset_index()

        energy_type = args.energy_types[0] if args.energy_types else "dft"
        print(f"\nProcessing {energy_type.upper()} hull energies from pickle")
        process_energy_type(
            df_full,
            hull_df,
            energy_type,
            args.threshold,
            output_dir,
            args.element_order,
        )

    # Create 'all' split with all hull energies (no threshold)
    df_all = None
    if args.hull.endswith(".parquet"):
        df_all = create_all_split(df_full, hull_df, output_dir, hull_columns)

    print(f"\n{'=' * 50}")
    print("All processing complete!")
    print(f"Files saved in {output_dir}")

    # Upload to Hugging Face if requested
    if args.upload:
        upload_to_huggingface(
            output_dir, args.hf_repo, args.hf_token, args.threshold, df_all
        )


def create_all_split(df_full, hull_df, output_dir, hull_columns):
    """Create 'all' split with all hull energies (no threshold)."""
    print(f"\n{'=' * 50}")
    print("Creating 'all' split with all hull energies")
    print(f"{'=' * 50}")

    df_all = df_full.merge(hull_df, on="immutable_id", how="inner")
    print(f"Merged {len(df_all):,} materials with hull energies")

    columns_to_save = [
        "immutable_id",
        "nsites",
        "species_at_sites",
        "formation_energy",
        "formation_energy_per_atom",
    ]

    columns_to_save.extend(hull_columns)

    energy_columns = [
        "true_energy",
        "orb_energy",
        "uma_energy",
        "mace_mp_energy",
        "mace_omat_energy",
    ]
    for col in energy_columns:
        if col in df_all.columns and col not in columns_to_save:
            columns_to_save.append(col)

    columns_to_save = [col for col in columns_to_save if col in df_all.columns]
    df_save = df_all[columns_to_save].copy()

    if "species_at_sites" in df_save.columns:

        def parse_species(x):
            if isinstance(x, str):
                x = x.strip("[]")
                elements = [
                    elem.strip("'\"") for elem in x.split() if elem.strip("'\"")
                ]
                return elements
            return (
                x.tolist()
                if hasattr(x, "tolist")
                else list(x)
                if not isinstance(x, list)
                else x
            )

        df_save["species_at_sites"] = df_save["species_at_sites"].apply(parse_species)

    all_file = output_dir / "all_hull_energies.parquet"
    df_save.to_parquet(all_file, index=False)
    print(f"Saved {len(df_save):,} materials with all hull energies to {all_file}")

    print("\nStatistics by hull type:")
    for hull_col in hull_columns:
        valid = df_save[hull_col].notna()
        if valid.sum() > 0:
            on_hull = (df_save[hull_col].abs() < 0.001).sum()
            stable = (df_save[hull_col] < 0.025).sum()
            print(f"  {hull_col}:")
            print(f"    Valid: {valid.sum():,}")
            print(
                f"    On hull (<0.001): {on_hull:,} ({100 * on_hull / valid.sum():.1f}%)"
            )
            print(
                f"    Stable (<0.025): {stable:,} ({100 * stable / valid.sum():.1f}%)"
            )

    return df_save


def upload_to_huggingface(output_dir, repo_id, token, threshold, df_all=None):
    """Upload generated files to Hugging Face Hub."""
    print(f"\n{'=' * 50}")
    print(f"Uploading to Hugging Face Hub: {repo_id}")
    print(f"{'=' * 50}")

    try:
        from datasets import Dataset, DatasetDict

        api = HfApi(token=token)

        try:
            create_repo(repo_id, repo_type="dataset", exist_ok=True, token=token)
            print(f"Repository {repo_id} ready")
        except Exception as e:
            print(f"Repository exists or created: {e}")

        all_energy_columns = [
            "true_energy",
            "orb_energy",
            "uma_energy",
            "mace_mp_energy",
            "mace_omat_energy",
        ]

        all_file = Path(output_dir) / "all_hull_energies.parquet"
        if all_file.exists():
            df_all_split = pd.read_parquet(all_file)

            if "species_at_sites" in df_all_split.columns:

                def ensure_list(x):
                    if hasattr(x, "tolist"):
                        return x.tolist()
                    elif isinstance(x, list):
                        return x
                    else:
                        return list(x)

                df_all_split["species_at_sites"] = df_all_split[
                    "species_at_sites"
                ].apply(ensure_list)

            dataset_all = Dataset.from_pandas(df_all_split, preserve_index=False)
            print(
                f"  Prepared 'all' dataset: {len(df_all_split)} materials (no threshold)"
            )
            dataset_all.push_to_hub(f"{repo_id}-All", token=token)

        datasets = {}
        threshold_str = f"{threshold:.3f}".replace(".", "_")

        for parquet_file in Path(output_dir).glob("*_above_hull_dataset.parquet"):
            energy_type = parquet_file.stem.replace("_above_hull_dataset", "")

            df = pd.read_parquet(parquet_file)

            for col in all_energy_columns:
                if col not in df.columns:
                    df[col] = (
                        np.nan
                    )  # Use NaN instead of None to maintain float64 dtype

            if "species_at_sites" in df.columns:

                def ensure_list(x):
                    if hasattr(x, "tolist"):
                        return x.tolist()
                    elif isinstance(x, list):
                        return x
                    else:
                        return list(x)

                df["species_at_sites"] = df["species_at_sites"].apply(ensure_list)

            standard_columns = [
                "immutable_id",
                "energy",
                "nsites",
                "species_at_sites",
                "formation_energy",
                "formation_energy_per_atom",
                "e_above_hull",
            ] + all_energy_columns

            # Keep only columns that exist
            columns_to_keep = [col for col in standard_columns if col in df.columns]
            df = df[columns_to_keep]

            dataset = Dataset.from_pandas(df, preserve_index=False)
            datasets[energy_type] = dataset

            print(f"  Prepared {energy_type} dataset: {len(df)} materials")

        if datasets:
            dataset_dict = DatasetDict(datasets)
            print("\nPushing datasets to hub...")
            dataset_dict.push_to_hub(repo_id, token=token)
            print(f"✓ Successfully pushed {len(datasets)} datasets to {repo_id}")

        print("\nUploading auxiliary files...")
        for file_path in Path(output_dir).glob("*.npz"):
            repo_path = f"threshold_{threshold_str}/{file_path.name}"
            print(f"  Uploading {file_path.name} -> {repo_path}")
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )

        readme_content = f"""# LeMat-Bulk MLIP Hull Reference Datasets

This dataset contains materials close to the convex hull computed using various ML interatomic potentials (MLIPs).

## Dataset Splits

- **`all`**: Contains ALL materials with hull energies for all MLIPs (no threshold filtering)
- **`dft`, `orb`, `uma`, `mace_mp`, `mace_omat`**: Materials within {threshold:.3f} eV/atom of respective hulls

## Energy Types

- `dft`: DFT reference energies
- `orb`: ORB model energies
- `uma`: UMA model energies
- `mace_mp`: MACE-MP model energies
- `mace_omat`: MACE-OMAT model energies

## File Structure

- `all_hull_energies.parquet`: All materials with hull energies for all energy types
- `{{energy_type}}_above_hull_dataset.parquet`: Filtered materials within threshold
- `{{energy_type}}_above_hull_composition_matrix.npz`: Sparse composition matrix
- `{{energy_type}}_above_hull_metadata.npz`: Metadata including element mappings

## Usage

```python
from datasets import load_dataset
import pandas as pd
from scipy import sparse
import numpy as np

# Load dataset
ds = load_dataset("LeMaterial/LeMat-Bulk-MLIP-Hull")

# Or load specific files
df = pd.read_parquet("hf://datasets/{repo_id}/threshold_{threshold_str}/dft_above_hull_dataset.parquet")
comp_matrix = sparse.load_npz("path/to/dft_above_hull_composition_matrix.npz")
metadata = np.load("path/to/dft_above_hull_metadata.npz", allow_pickle=True)
```
"""

        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )

        print(f"\nDataset available at: https://huggingface.co/datasets/{repo_id}")

    except Exception as e:
        print(f"\nError uploading to Hugging Face: {e}")
        print("Files are still saved locally in:", output_dir)


def process_energy_type(
    df_full, hull_df, energy_type, threshold, output_dir, element_order_file
):
    """Process a single energy type."""
    df_merged = df_full.merge(hull_df, on="immutable_id", how="inner")
    print(f"  Merged {len(df_merged):,} materials with hull energies")

    df_filtered = df_merged[df_merged["e_above_hull"] <= threshold].copy()
    print(
        f"  Filtered to {len(df_filtered):,} materials within {threshold} eV/atom of hull"
    )

    if len(df_filtered) == 0:
        print("  WARNING: No materials found within threshold!")
        return

    print("\n  Statistics:")
    print(
        f"    Min/Max: {df_filtered['e_above_hull'].min():.4f} / {df_filtered['e_above_hull'].max():.4f} eV/atom"
    )
    print(
        f"    Mean/Median: {df_filtered['e_above_hull'].mean():.4f} / {df_filtered['e_above_hull'].median():.4f} eV/atom"
    )
    on_hull = (df_filtered["e_above_hull"].abs() < 0.001).sum()
    print(f"    On hull: {on_hull:,} ({100 * on_hull / len(df_filtered):.1f}%)")

    base_name = f"{energy_type}_above_hull"
    parquet_file = output_dir / f"{base_name}_dataset.parquet"
    npz_file = output_dir / f"{base_name}_composition_matrix.npz"

    print(f"\n  Saving dataset to {parquet_file}")
    if energy_type == "dft":
        energy_col = "true_energy"
    else:
        energy_col = f"{energy_type}_energy"

    columns_to_save = [
        "immutable_id",
        "nsites",
        "species_at_sites",
        "formation_energy",
        "formation_energy_per_atom",
        "e_above_hull",
    ]

    if energy_col in df_filtered.columns:
        columns_to_save.append(energy_col)

    energy_columns = [
        "true_energy",
        "orb_energy",
        "uma_energy",
        "mace_mp_energy",
        "mace_omat_energy",
    ]
    for col in energy_columns:
        if col in df_filtered.columns and col not in columns_to_save:
            columns_to_save.append(col)

    columns_to_save = [col for col in columns_to_save if col in df_filtered.columns]

    df_save = df_filtered[columns_to_save].copy()
    if energy_col in df_save.columns:
        df_save["energy"] = df_save[energy_col].copy()
    if "species_at_sites" in df_save.columns:

        def parse_species(x):
            if isinstance(x, str):
                x = x.strip("[]")
                elements = [
                    elem.strip("'\"") for elem in x.split() if elem.strip("'\"")
                ]
                return elements
            return (
                x.tolist()
                if hasattr(x, "tolist")
                else list(x)
                if not isinstance(x, list)
                else x
            )

        df_save["species_at_sites"] = df_save["species_at_sites"].apply(parse_species)

    df_save.to_parquet(parquet_file, index=False)
    print(f"    Saved {len(df_save):,} materials")

    print("\n  Creating composition matrix...")
    comp_matrix, element_to_idx = create_composition_matrix(df_save, element_order_file)
    print(
        f"    Matrix: {comp_matrix.shape}, Sparsity: {1 - comp_matrix.nnz / (comp_matrix.shape[0] * comp_matrix.shape[1]):.2%}"
    )

    sparse.save_npz(npz_file, comp_matrix)

    metadata_file = output_dir / f"{base_name}_metadata.npz"
    np.savez(
        metadata_file,
        num_materials=len(df_filtered),
        num_elements=len(element_to_idx),
        element_names=list(element_to_idx.keys()),
        element_indices=list(element_to_idx.values()),
        threshold_eV_per_atom=threshold,
        energy_type=energy_type,
        immutable_ids=df_filtered["immutable_id"].values,
    )
    print("  Saved metadata")


if __name__ == "__main__":
    main()
