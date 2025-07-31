import json
from pathlib import Path

import pandas as pd
from pymatgen.core import Structure

from lemat_genbench.utils.logging import logger


def format_structures(
    structures: list[Structure] | list[dict] | pd.DataFrame | str | Path,
) -> list[Structure]:
    """Convert various input formats to a list of pymatgen Structure objects.

    Parameters
    ----------
    structures : list[Structure] | list[dict] | pd.DataFrame | str | Path
        Structures in various formats:
        - List of pymatgen Structure objects
        - List of dictionaries convertible to Structure
        - Pandas DataFrame with structure information
        - String or Path to a file or directory containing structures

    Returns
    -------
    list[Structure]
        List of pymatgen Structure objects.

    Raises
    ------
    ValueError
        If the input format is not supported.
    """
    if isinstance(structures, list):
        if all(isinstance(s, Structure) for s in structures):
            return structures
        elif all(isinstance(s, dict) for s in structures):
            return [Structure.from_dict(s) for s in structures]

    elif isinstance(structures, pd.DataFrame):
        # Convert dataframe to structures (implementation depends on the expected format)
        pass

    elif isinstance(structures, (str, Path)):
        path = Path(structures)
        if path.is_file():
            files = [path]
        elif path.is_dir():
            files = list(path.glob("*.jsonl"))

        logger.info(f"Found {len(files)} files in {path}")

        structures = []
        for file in files:
            if file.suffix == ".jsonl":
                structures.extend(
                    [Structure.from_dict(json.loads(line)) for line in open(file)]
                )
            else:
                raise ValueError(f"Unsupported file extension: {file.suffix}")

    return structures
