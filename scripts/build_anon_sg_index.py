"""
Build an anonymous formula → space group index from LeMat-Bulk.

For each structure in LeMat-Bulk, computes the space group number and records
it against the structure's anonymous formula (with nsites collapsed, so A2B2
maps to AB).  The result is a JSON mapping each anonymous formula to its list
of known space group numbers.

Uses the same chunked multiprocessing pattern as ``lematbulk_oxi_states.py``.

Usage:
    uv run python scripts/build_anon_sg_index.py
    uv run python scripts/build_anon_sg_index.py --estimate  # print runtime estimate only
"""

import argparse
import json
import time
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path

from datasets import load_dataset
from pymatgen.core import Composition, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tqdm import tqdm

LEMATBULK_DATASET = "Lematerial/LeMat-Bulk"
LEMATBULK_CONFIG = "compatible_pbe"
_SYMPREC = 0.01

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "lematbulk_anon_sg_index.json"


def lematbulk_item_to_structure(item: dict) -> Structure:
    """Convert a LeMat-Bulk item to a pymatgen Structure object.

    Parameters
    ----------
    item : dict
        The item to convert.

    Returns
    -------
    Structure
        The pymatgen Structure object.
    """
    return Structure(
        species=item["species_at_sites"],
        coords=item["cartesian_site_positions"],
        lattice=item["lattice_vectors"],
        coords_are_cartesian=True,
    )


def process_item(item: dict) -> tuple[str | None, int | None]:
    """Extract anonymous formula and space group from a single LeMat-Bulk row.

    Parameters
    ----------
    item : dict
        A single row from the LeMat-Bulk dataset.

    Returns
    -------
    tuple[str | None, int | None]
        ``(anonymous_formula, space_group_number)`` or ``(None, None)`` on failure.
    """
    try:
        struct = lematbulk_item_to_structure(item)
        # Collapse nsites: A2B2 → AB via reduced anonymous formula
        anon = Composition(
            struct.composition.anonymized_formula
        ).reduced_composition.anonymized_formula
        sg = SpacegroupAnalyzer(struct, symprec=_SYMPREC).get_space_group_number()
        return (anon, sg)
    except Exception:
        return (None, None)


def process_item_tuple(args: tuple) -> tuple[str | None, int | None]:
    """Same as process_item but accepts pre-extracted column data.

    Parameters
    ----------
    args : tuple
        ``(species_at_sites, cartesian_site_positions, lattice_vectors)``.

    Returns
    -------
    tuple[str | None, int | None]
        ``(anonymous_formula, space_group_number)`` or ``(None, None)`` on failure.
    """
    species, coords, lattice = args
    try:
        struct = Structure(
            species=species, coords=coords, lattice=lattice,
            coords_are_cartesian=True,
        )
        anon = Composition(
            struct.composition.anonymized_formula
        ).reduced_composition.anonymized_formula
        sg = SpacegroupAnalyzer(struct, symprec=_SYMPREC).get_space_group_number()
        return (anon, sg)
    except Exception:
        return (None, None)


def estimate_runtime(dataset, output_path: Path, n_sample: int = 500) -> None:
    """Time a sample of structures, estimate total runtime, and save sample results.

    Parameters
    ----------
    dataset : Dataset
        The full LeMat-Bulk dataset.
    output_path : Path
        Where to save the sample index JSON for format inspection.
    n_sample : int
        Number of structures to time.
    """
    import random

    indices = random.sample(range(len(dataset)), n_sample)
    items = [dataset[i] for i in indices]

    # Single-core timing, collecting results
    index: dict[str, set[int]] = defaultdict(set)
    n_failed = 0
    start = time.time()
    for item in tqdm(items, desc="Timing sample (single-core)"):
        anon, sg = process_item(item)
        if anon is not None and sg is not None:
            index[anon].add(sg)
        else:
            n_failed += 1
    elapsed = time.time() - start
    per_item = elapsed / n_sample

    n_workers = max(1, cpu_count() - 1)
    total_serial = per_item * len(dataset)
    total_parallel = total_serial / n_workers

    print(f"\nSample: {n_sample} structures in {elapsed:.1f}s ({per_item*1000:.1f}ms each)")
    print(f"  {len(index)} anonymous formulas, {n_failed} failures")
    print(f"Total dataset: {len(dataset):,} structures")
    print(f"Estimated serial runtime: {total_serial/3600:.1f} hours")
    print(f"Estimated parallel runtime ({n_workers} workers): {total_parallel/3600:.1f} hours")

    # Save sample results so user can inspect formatting
    sample_path = output_path.with_name(output_path.stem + "_sample.json")
    serialisable = {anon: sorted(sgs) for anon, sgs in index.items()}
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sample_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"Sample index saved to {sample_path}")


def build_index(dataset) -> dict[str, list[int]]:
    """Build the anonymous formula → space group index using multiprocessing.

    Pre-extracts columns as lists so workers receive lightweight tuples
    instead of full dataset dicts, avoiding a main-process bottleneck.

    Parameters
    ----------
    dataset : Dataset
        The full LeMat-Bulk dataset.

    Returns
    -------
    dict[str, list[int]]
        ``{anonymous_formula: [sorted space group numbers]}``.
    """
    n_workers = max(1, cpu_count() - 1)
    total = len(dataset)

    # Pre-extract columns to avoid per-item dataset access in the main process
    print("Extracting columns...")
    species_col = dataset["species_at_sites"]
    coords_col = dataset["cartesian_site_positions"]
    lattice_col = dataset["lattice_vectors"]

    print(f"Processing {total:,} structures with {n_workers} workers...")

    work = zip(species_col, coords_col, lattice_col)

    index: dict[str, set[int]] = defaultdict(set)
    n_failed = 0

    with Pool(processes=n_workers) as pool:
        with tqdm(total=total, desc="Computing space groups") as pbar:
            for anon, sg in pool.imap(process_item_tuple, work, chunksize=100):
                pbar.update(1)
                if anon is not None and sg is not None:
                    index[anon].add(sg)
                else:
                    n_failed += 1

    print(f"Completed: {len(index)} anonymous formulas, {n_failed} failures")

    # Convert sets to sorted lists for JSON serialisation
    return {anon: sorted(sgs) for anon, sgs in index.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build anonymous formula → space group index from LeMat-Bulk."
    )
    parser.add_argument(
        "--estimate", action="store_true",
        help="Only estimate runtime, don't build the full index",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=OUTPUT_PATH,
        help=f"Output JSON path (default: {OUTPUT_PATH})",
    )
    args = parser.parse_args()

    print("Loading LeMat-Bulk dataset...")
    dataset = load_dataset(
        LEMATBULK_DATASET,
        name=LEMATBULK_CONFIG,
        split="train",
        streaming=False,
    )
    print(f"Loaded {len(dataset):,} structures")

    if args.estimate:
        estimate_runtime(dataset, args.output)
        return

    index = build_index(dataset)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(index, f)
    print(f"Saved index to {args.output}")


if __name__ == "__main__":
    main()
