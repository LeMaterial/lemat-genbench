"""
Run validity, uniqueness, novelty, and diversity benchmarks across a range of symprec values.

Accepts the same --csv / --cifs input as run_benchmarks.py. Results are saved to
results_final/ with one JSON per symprec value.

Usage:
    uv run python scripts/run_symprec_sweep.py --csv scripts/diffcsp_2500.csv --name diffcsp_sweep
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from pymatgen.core import Structure
from tqdm import tqdm

from lemat_genbench.utils.logging import logger

SYMPREC_VALUES = [1e-5, 1e-3, 0.01, 0.1, 0.5]
RESULTS_DIR = Path(__file__).parent.parent / "results_final"


def load_structures_from_csv(csv_path: str) -> List[Structure]:
    """Load structures from a CSV file.

    Looks for a column named ``structure``, ``LeMatStructs``, or ``cif_string``
    and parses each row into a pymatgen ``Structure``.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.

    Returns
    -------
    list[Structure]
        Parsed structures.
    """
    import json as _json

    import pandas as pd

    df = pd.read_csv(csv_path)

    structure_column = None
    for col_name in ["structure", "LeMatStructs", "cif_string"]:
        if col_name in df.columns:
            structure_column = col_name
            break
    if structure_column is None:
        raise ValueError(
            "CSV must contain a 'structure', 'LeMatStructs', or 'cif_string' column"
        )

    structures: List[Structure] = []
    for idx, row in df.iterrows():
        try:
            data = row[structure_column]
            if isinstance(data, str) and data.strip().startswith("{"):
                structures.append(Structure.from_dict(_json.loads(data)))
            else:
                structures.append(Structure.from_str(data, fmt="cif"))
        except Exception as e:
            logger.warning(f"Skipping row {idx}: {e}")

    logger.info(f"Loaded {len(structures)} structures from {csv_path}")
    return structures


def load_structures_from_cifs(cif_path: str) -> List[Structure]:
    """Load structures from a directory of CIF files or a text file listing paths.

    Parameters
    ----------
    cif_path : str
        Directory or text file.

    Returns
    -------
    list[Structure]
        Parsed structures.
    """
    p = Path(cif_path)
    if p.is_dir():
        cif_files = sorted(p.glob("*.cif"))
    elif p.is_file():
        cif_files = [Path(line.strip()) for line in p.read_text().splitlines() if line.strip()]
    else:
        raise FileNotFoundError(f"Path does not exist: {cif_path}")

    structures: List[Structure] = []
    for f in tqdm(cif_files, desc="Loading CIF files"):
        try:
            structures.append(Structure.from_file(str(f)))
        except Exception as e:
            logger.warning(f"Skipping {f.name}: {e}")

    logger.info(f"Loaded {len(structures)} structures from {cif_path}")
    return structures


def run_validity(structures: List[Structure], symprec: float) -> Dict[str, Any]:
    """Run the validity benchmark with a specific symprec.

    Parameters
    ----------
    structures : list[Structure]
        Structures to evaluate.
    symprec : float
        Symmetry precision passed through to ``SpacegroupAnalyzer``.

    Returns
    -------
    dict[str, Any]
        Validity benchmark result.
    """
    from lemat_genbench.benchmarks.validity_benchmark import ValidityBenchmark
    from lemat_genbench.metrics import validity_metrics as _vm

    original = getattr(_vm, "_SYMPREC", None)
    _vm._SYMPREC = symprec
    try:
        benchmark = ValidityBenchmark()
        return benchmark.evaluate(structures)
    finally:
        if original is None and hasattr(_vm, "_SYMPREC"):
            del _vm._SYMPREC
        else:
            _vm._SYMPREC = original


def run_novelty(
    structures: List[Structure],
    symprec: float,
    fingerprint_method: str = "short-bawl",
) -> Dict[str, Any]:
    """Run the novelty benchmark with a specific symprec.

    Parameters
    ----------
    structures : list[Structure]
        Structures to evaluate.
    symprec : float
        Symmetry precision passed through to ``SpacegroupAnalyzer``.
    fingerprint_method : str
        Fingerprint method for the underlying novelty metric.

    Returns
    -------
    dict[str, Any]
        Novelty benchmark result.
    """
    from lemat_genbench.benchmarks.novelty_benchmark import NoveltyBenchmark
    from lemat_genbench.metrics import novelty_metric as _nm

    original = getattr(_nm, "_SYMPREC", None)
    _nm._SYMPREC = symprec
    try:
        benchmark = NoveltyBenchmark(fingerprint_method=fingerprint_method)
        return benchmark.evaluate(structures)
    finally:
        if original is None and hasattr(_nm, "_SYMPREC"):
            del _nm._SYMPREC
        else:
            _nm._SYMPREC = original


def run_uniqueness(
    structures: List[Structure],
    symprec: float,
    fingerprint_method: str = "short-bawl",
) -> Dict[str, Any]:
    """Run the uniqueness benchmark with a specific symprec.

    Parameters
    ----------
    structures : list[Structure]
        Structures to evaluate.
    symprec : float
        Symmetry precision passed through to ``SpacegroupAnalyzer``.
    fingerprint_method : str
        Fingerprint method for the underlying uniqueness metric.

    Returns
    -------
    dict[str, Any]
        Uniqueness benchmark result.
    """
    from lemat_genbench.benchmarks.uniqueness_benchmark import UniquenessBenchmark
    from lemat_genbench.metrics import uniqueness_metric as _um

    original = getattr(_um, "_SYMPREC", None)
    _um._SYMPREC = symprec
    try:
        benchmark = UniquenessBenchmark(fingerprint_method=fingerprint_method)
        return benchmark.evaluate(structures)
    finally:
        if original is None and hasattr(_um, "_SYMPREC"):
            del _um._SYMPREC
        else:
            _um._SYMPREC = original


def run_diversity(structures: List[Structure], symprec: float) -> Dict[str, Any]:
    """Run the diversity benchmark with a specific symprec.

    Parameters
    ----------
    structures : list[Structure]
        Structures to evaluate.
    symprec : float
        Symmetry precision passed through to ``SpacegroupAnalyzer``.

    Returns
    -------
    dict[str, Any]
        Diversity benchmark result.
    """
    from lemat_genbench.benchmarks.diversity_benchmark import DiversityBenchmark
    from lemat_genbench.metrics import diversity_metric as _dm

    original = getattr(_dm, "_SYMPREC", None)
    _dm._SYMPREC = symprec
    try:
        benchmark = DiversityBenchmark()
        return benchmark.evaluate(structures)
    finally:
        if original is None and hasattr(_dm, "_SYMPREC"):
            del _dm._SYMPREC
        else:
            _dm._SYMPREC = original


def save_sweep_results(
    all_results: Dict[float, Dict[str, Any]],
    run_name: str,
    n_structures: int,
) -> Path:
    """Save the full sweep to a single JSON file.

    Parameters
    ----------
    all_results : dict[float, dict[str, Any]]
        Mapping of symprec → benchmark results.
    run_name : str
        Human-readable run name.
    n_structures : int
        Total structures loaded.

    Returns
    -------
    Path
        Path to the saved JSON file.
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = RESULTS_DIR / f"{run_name}_symprec_sweep_{timestamp}.json"

    output = {
        "run_info": {
            "run_name": run_name,
            "timestamp": timestamp,
            "n_structures": n_structures,
            "symprec_values": SYMPREC_VALUES,
        },
        "results_by_symprec": {
            str(sp): res for sp, res in all_results.items()
        },
    }

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"Results saved to {filepath}")
    return filepath


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run validity/uniqueness/novelty/diversity benchmarks across symprec values."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", help="CSV file with a structure column")
    source.add_argument(
        "--cifs", help="Directory of .cif files or text file listing CIF paths"
    )
    parser.add_argument("--name", required=True, help="Name for this sweep run")
    parser.add_argument(
        "--fingerprint-method",
        default="short-bawl",
        choices=["bawl", "short-bawl", "structure-matcher"],
        help="Fingerprinting method (default: short-bawl)",
    )
    args = parser.parse_args()

    # --- Load structures once ---
    if args.csv:
        structures = load_structures_from_csv(args.csv)
    else:
        structures = load_structures_from_cifs(args.cifs)

    if not structures:
        logger.error("No structures loaded. Exiting.")
        return

    # --- Sweep symprec for validity, uniqueness, and novelty ---
    all_results: Dict[float, Dict[str, Any]] = {}

    for symprec in SYMPREC_VALUES:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"symprec = {symprec}")
        logger.info(f"{'=' * 60}")

        sweep_entry: Dict[str, Any] = {}

        # Validity
        logger.info(f"  Running validity (symprec={symprec})...")
        t0 = time.time()
        validity_result = run_validity(structures, symprec)
        logger.info(f"  Validity complete in {time.time() - t0:.1f}s")
        sweep_entry["validity"] = validity_result

        # Novelty
        logger.info(f"  Running novelty (symprec={symprec})...")
        t0 = time.time()
        novelty_result = run_novelty(
            structures, symprec, fingerprint_method=args.fingerprint_method
        )
        logger.info(f"  Novelty complete in {time.time() - t0:.1f}s")
        sweep_entry["novelty"] = novelty_result

        # Uniqueness
        logger.info(f"  Running uniqueness (symprec={symprec})...")
        t0 = time.time()
        uniqueness_result = run_uniqueness(
            structures, symprec, fingerprint_method=args.fingerprint_method
        )
        logger.info(f"  Uniqueness complete in {time.time() - t0:.1f}s")
        sweep_entry["uniqueness"] = uniqueness_result

        # Diversity
        logger.info(f"  Running diversity (symprec={symprec})...")
        t0 = time.time()
        diversity_result = run_diversity(structures, symprec)
        logger.info(f"  Diversity complete in {time.time() - t0:.1f}s")
        sweep_entry["diversity"] = diversity_result

        all_results[symprec] = sweep_entry

    # --- Save ---
    filepath = save_sweep_results(all_results, args.name, len(structures))
    print(f"\nSweep complete. Results: {filepath}")


if __name__ == "__main__":
    main()
