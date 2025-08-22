from typing import Any, Optional

from material_hasher.hasher.bawl import BAWLHasher
from pymatgen.analysis.local_env import EconNN
from pymatgen.core import Structure

# Handle optional imports gracefully
try:
    from material_hasher.similarity.pdd import PointwiseDistanceDistributionSimilarity
    PDD_AVAILABLE = True
except ImportError:
    PDD_AVAILABLE = False

try:
    from material_hasher.similarity.structure_matchers import (
        PymatgenStructureSimilarity,
    )
    STRUCTURE_MATCHER_AVAILABLE = True
except ImportError:
    STRUCTURE_MATCHER_AVAILABLE = False


def get_fingerprinter(fingerprint_method: str) -> Any:
    """Get a fingerprinter instance based on the specified method.

    Parameters
    ----------
    fingerprint_method : str
        Method to use for structure fingerprinting.
        Supported methods:
        - "bawl", "short-bawl": BAWL fingerprinting
        - "structure-matcher": Pymatgen structure similarity (if available)
        - "pdd": Pointwise Distance Distribution similarity (if available)

    Returns
    -------
    Any
        Fingerprinter instance.

    Raises
    ------
    ValueError
        If the specified fingerprint method is not supported.
    ImportError
        If the required dependencies for the fingerprint method are not available.
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
        if not STRUCTURE_MATCHER_AVAILABLE:
            raise ImportError(
                "PymatgenStructureSimilarity is not available. "
                "Please check your material-hasher installation."
            )
        return PymatgenStructureSimilarity(tolerance=0.1)
    elif fingerprint_method.lower() == "pdd":
        if not PDD_AVAILABLE:
            raise ImportError(
                "PointwiseDistanceDistributionSimilarity is not available. "
                "Please check your material-hasher installation or use an alternative fingerprint method."
            )
        return PointwiseDistanceDistributionSimilarity()
    else:
        available_methods = ["bawl", "short-bawl"]
        if STRUCTURE_MATCHER_AVAILABLE:
            available_methods.append("structure-matcher")
        if PDD_AVAILABLE:
            available_methods.append("pdd")
        
        raise ValueError(
            f"Unknown fingerprint method: {fingerprint_method}. "
            f"Currently available: {', '.join(available_methods)}"
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