import os

import numpy as np
import pandas as pd
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure


def generate_random_structure(elements=["Li", "Fe", "O"], n_atoms=None):
    """Generate a random crystal structure.

    Parameters
    ----------
    elements : list
        List of possible elements to use
    n_atoms : int
        Number of atoms. If None, will use random number between 2-10

    Returns
    -------
    Structure
        Pymatgen Structure object
    """
    if n_atoms is None:
        n_atoms = np.random.randint(2, 11)

    # Random Lattices
    a = np.random.uniform(3, 10)
    b = np.random.uniform(3, 10)
    c = np.random.uniform(3, 10)
    alpha = np.random.uniform(60, 120)
    beta = np.random.uniform(60, 120)
    gamma = np.random.uniform(60, 120)

    lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma).matrix

    # Random Atomic Species and Positions
    species = np.random.choice(elements, size=n_atoms)
    coords = np.random.random((n_atoms, 3))

    structure = Structure(lattice, species, coords)
    return structure


def structure_to_dict(structure):
    """Convert a structure to a flat dictionary for CSV storage."""
    return structure.as_dict()


def main():
    n_structures = 100
    elements = ["Li", "Fe", "O", "Na", "Mg", "Al", "Si", "P", "S"]

    structures = []
    for _ in range(n_structures):
        structure = generate_random_structure(elements=elements)
        structures.append(structure_to_dict(structure))

    # Convert to DataFrame and save as JSONL
    df = pd.DataFrame(structures)

    os.makedirs("data", exist_ok=True)

    output_file = "data/random_structures.jsonl"
    df.to_json(output_file, orient="records", lines=True)
    print(f"Generated {n_structures} structures and saved to {output_file}")


if __name__ == "__main__":
    main()
