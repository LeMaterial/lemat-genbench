"""Util functions for e_above_hull calculation."""

import gzip
import pickle
import tempfile
from pathlib import Path

from pymatgen.analysis.phase_diagram import PatchedPhaseDiagram
from pymatgen.core import Structure
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.vasp.inputs import Incar, Poscar
from pymatgen.io.vasp.sets import MPRelaxSet


def get_patched_phase_diagram_mp(path: Path) -> PatchedPhaseDiagram:
    # Check if the file has a .gz extension
    if path.suffix == ".gz":
        # Open as gzip file
        with gzip.open(path, "rb") as f:
            ppd_mp = pickle.load(f)
    else:
        # Open as regular file
        with open(path, "rb") as f:
            ppd_mp = pickle.load(f)
    return ppd_mp


def generate_CSE(structure, energy):
    # Write VASP inputs files as if we were going to do a standard MP run
    # this is mainly necessary to get the right U values / etc
    b = MPRelaxSet(structure)
    with tempfile.TemporaryDirectory() as tmpdirname:
        b.write_input(f"{tmpdirname}/", potcar_spec=True)
        poscar = Poscar.from_file(f"{tmpdirname}/POSCAR")
        incar = Incar.from_file(f"{tmpdirname}/INCAR")
        clean_structure = Structure.from_file(f"{tmpdirname}/POSCAR")

    # Get the U values and figure out if we should have run a GGA+U calc
    param = {"hubbards": {}}
    if "LDAUU" in incar:
        param["hubbards"] = dict(zip(poscar.site_symbols, incar["LDAUU"]))
    param["is_hubbard"] = (
        incar.get("LDAU", True) and sum(param["hubbards"].values()) > 0
    )
    if param["is_hubbard"]:
        param["run_type"] = "GGA+U"

    # Make a ComputedStructureEntry without the correction
    cse_d = {
        "structure": clean_structure,
        "energy": energy,
        "correction": 0.0,
        "parameters": param,
    }

    # Apply the MP 2020 correction scheme (anion/+U/etc)
    cse = ComputedStructureEntry.from_dict(cse_d)
    _ = MaterialsProject2020Compatibility(check_potcar=False).process_entries(
        cse,
        clean=True,
    )

    # Return the final CSE (notice that the composition/etc is also clean, not things like Fe3+)!
    return cse
