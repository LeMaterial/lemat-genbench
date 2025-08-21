from typing import Any, Optional

from material_hasher.hasher.bawl import BAWLHasher
from material_hasher.similarity.pdd import PointwiseDistanceDistributionSimilarity
from material_hasher.similarity.structure_matchers import PymatgenStructureSimilarity
from pymatgen.analysis.local_env import EconNN
from pymatgen.core import Structure


def get_fingerprinter(fingerprint_method: str) -> Any:
    """Get a fingerprinter instance based on the specified method.

    Parameters
    ----------
    fingerprint_method : str
        Method to use for structure fingerprinting.
        Supported methods:
        - "bawl", "short-bawl": BAWL fingerprinting
        - "structure-matcher": Pymatgen structure similarity
        - "pdd": Pointwise Distance Distribution similarity

    Returns
    -------
    Any
        Fingerprinter instance.

    Raises
    ------
    ValueError
        If the specified fingerprint method is not supported.
    """
    if "bawl" in fingerprint_method.lower():
        return BAWLHasher(
            graphing_algorithm="WL",
            bonding_algorithm=EconNN,
            bonding_kwargs={
                "tol": 0.2,
                "cutoff": 10,
                "use_fictive_radius": True,
            },
            include_composition=True,
            symmetry_labeling="SPGLib",
            shorten_hash="short" in fingerprint_method.lower(),
        )
    elif fingerprint_method.lower() == "structure-matcher":
        return PymatgenStructureSimilarity(tolerance=0.1)
    elif fingerprint_method.lower() == "pdd":
        return PointwiseDistanceDistributionSimilarity()
    else:
        raise ValueError(
            f"Unknown fingerprint method: {fingerprint_method}. "
            "Currently supported: 'bawl', 'short-bawl', 'structure-matcher', 'pdd'"
        )


def get_fingerprint(structure: Structure, fingerprinter: Any) -> Optional[str]:
    """Get the fingerprint for a structure, using cached value if available.

    Parameters
    ----------
    structure : Structure
        The pymatgen Structure to get the fingerprint for.
    fingerprinter : Any
        Fingerprinter instance.

    Returns
    -------
    Optional[str]
        Structure fingerprint if successful, None if failed.
    """
    # Check if the fingerprint is already cached in the structure properties
    if "fingerprint" in structure.properties:
        return structure.properties["fingerprint"]

    # Otherwise compute the fingerprint
    try:
        return fingerprinter.get_material_hash(structure)
    except Exception:
        return None
