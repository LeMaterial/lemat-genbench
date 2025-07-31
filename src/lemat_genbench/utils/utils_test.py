from pymatgen.core.structure import Lattice, Structure


def create_test_structure(a=4.0, elements=["Si"]):
    """Create a simple test structure."""
    lattice = Lattice.cubic(a)
    coords = [[0, 0, 0]]
    return Structure(lattice, elements, coords)
