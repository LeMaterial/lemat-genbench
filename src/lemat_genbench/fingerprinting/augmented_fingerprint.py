"""Augmented fingerprinting for crystal structures.

This module implements the augmented fingerprinting approach that combines
space group, element, and Wyckoff position information to create highly
specific fingerprints for crystal structure comparison.
"""

import warnings
from collections import Counter, defaultdict
from math import gcd
from typing import Any, Dict, Tuple, Union

from pymatgen.core.structure import Structure

from lemat_genbench.fingerprinting.crystallographic_analyzer import (
    CrystallographicAnalyzer,
    structure_to_crystallographic_dict,
)
from lemat_genbench.utils.logging import logger

# Suppress common warnings
warnings.filterwarnings("ignore", message="No oxidation states specified on sites!")
warnings.filterwarnings("ignore", category=FutureWarning)


class AugmentedFingerprinter:
    """Fingerprinter using augmented crystallographic approach.

    This class implements the augmented fingerprinting method that considers
    space group, elements, site symmetries, and equivalent Wyckoff position
    enumerations to create highly specific structure fingerprints.
    """

    def __init__(self, symprec: float = 0.01, angle_tolerance: float = 5.0):
        """Initialize the augmented fingerprinter.

        Parameters
        ----------
        symprec : float, default=0.01
            Symmetry precision for crystallographic analysis.
        angle_tolerance : float, default=5.0
            Angle tolerance in degrees for crystallographic analysis.
        """
        self.analyzer = CrystallographicAnalyzer(
            symprec=symprec, angle_tolerance=angle_tolerance
        )

    def get_structure_fingerprint(self, structure: Structure) -> Union[str, None]:
        """Get augmented fingerprint for a pymatgen Structure.

        Parameters
        ----------
        structure : Structure
            The pymatgen Structure object to fingerprint.

        Returns
        -------
        str or None
            The augmented fingerprint string, or None if analysis failed.
        """
        try:
            # Get crystallographic metadata
            crystal_data = structure_to_crystallographic_dict(structure)

            if not crystal_data["success"]:
                logger.warning(
                    f"Crystallographic analysis failed: {crystal_data['error']}"
                )
                return None

            # Convert to fingerprint using augmented method
            fingerprint = record_to_augmented_fingerprint(crystal_data)

            # Convert to string representation
            return self._fingerprint_to_string(fingerprint)

        except Exception as e:
            logger.warning(f"Augmented fingerprinting failed: {e}")
            return None

    def _fingerprint_to_string(self, fingerprint: Tuple) -> str:
        """Convert fingerprint tuple to string representation.

        Parameters
        ----------
        fingerprint : Tuple
            The fingerprint tuple from record_to_augmented_fingerprint.

        Returns
        -------
        str
            String representation of the fingerprint.
        """
        try:
            # Convert complex nested structures to string
            spacegroup, wyckoff_variants = fingerprint

            # Sort the variants for consistent string representation
            variant_strs = []
            for variant in wyckoff_variants:
                # Convert frozenset of Counter items to sorted string
                variant_items = sorted(list(variant))
                variant_str = "_".join(
                    [f"{item[0]}:{item[1]}" for item in variant_items]
                )
                variant_strs.append(variant_str)

            variant_strs.sort()  # Ensure consistent ordering
            variants_combined = "|".join(variant_strs)

            return f"AUG_{spacegroup}_{variants_combined}"

        except Exception as e:
            logger.warning(f"Failed to convert fingerprint to string: {e}")
            return f"AUG_FAILED_{hash(str(fingerprint))}"


def record_to_augmented_fingerprint(row: Dict[str, Any]) -> Tuple:
    """Compute augmented fingerprint from crystallographic metadata.

    This is adapted from the original pasted code to work with our
    crystallographic analyzer output format.

    Parameters
    ----------
    row : Dict[str, Any]
        Dictionary containing crystallographic metadata:
        - spacegroup_number: int
        - elements: List[str]
        - site_symmetries: List[str]
        - sites_enumeration_augmented: List[List[int]]

    Returns
    -------
    Tuple
        Fingerprint tuple (spacegroup_number, frozenset of Wyckoff representations)
    """
    try:
        spacegroup_number = row["spacegroup_number"]
        elements = row["elements"]
        site_symmetries = row["site_symmetries"]
        sites_enumeration_augmented = row["sites_enumeration_augmented"]

        # Generate all possible Wyckoff representations
        wyckoff_variants = frozenset(
            map(
                lambda enumeration: frozenset(
                    Counter(
                        map(tuple, zip(elements, site_symmetries, enumeration))
                    ).items()
                ),
                sites_enumeration_augmented,
            )
        )

        return (spacegroup_number, wyckoff_variants)

    except Exception as e:
        logger.warning(f"Failed to compute augmented fingerprint: {e}")
        # Return a fallback fingerprint
        return (0, frozenset())


def record_to_anonymous_fingerprint(row: Dict[str, Any]) -> Tuple:
    """Compute anonymous fingerprint (structure without chemistry).

    Parameters
    ----------
    row : Dict[str, Any]
        Dictionary containing crystallographic metadata.

    Returns
    -------
    Tuple
        Anonymous fingerprint tuple (spacegroup_number, structural variants)
    """
    try:
        spacegroup_number = row["spacegroup_number"]
        site_symmetries = row["site_symmetries"]
        sites_enumeration_augmented = row["sites_enumeration_augmented"]

        # Generate structural variants without element information
        structural_variants = frozenset(
            map(
                lambda enumeration: frozenset(
                    Counter(map(tuple, zip(site_symmetries, enumeration))).items()
                ),
                sites_enumeration_augmented,
            )
        )

        return (spacegroup_number, structural_variants)

    except Exception as e:
        logger.warning(f"Failed to compute anonymous fingerprint: {e}")
        return (0, frozenset())


def record_to_relaxed_AFLOW_fingerprint(row: Dict[str, Any]) -> Tuple:
    """Compute relaxed AFLOW-style fingerprint.

    Parameters
    ----------
    row : Dict[str, Any]
        Dictionary containing crystallographic metadata.

    Returns
    -------
    Tuple
        AFLOW-style fingerprint tuple.
    """
    try:
        spacegroup_number = row["spacegroup_number"]
        elements = row["elements"]
        multiplicity = row["multiplicity"]
        site_symmetries = row["site_symmetries"]
        sites_enumeration_augmented = row["sites_enumeration_augmented"]

        # Calculate simplified stoichiometry
        element_counts = defaultdict(int)
        for element, mult in zip(elements, multiplicity):
            element_counts[element] += mult

        stochio_gcd = gcd(*element_counts.values()) if element_counts.values() else 1
        simplified_stochio = [mult // stochio_gcd for mult in element_counts.values()]

        # Generate Wyckoff site information
        sites = frozenset(
            map(
                lambda enumeration: frozenset(
                    Counter(map(tuple, zip(site_symmetries, enumeration))).items()
                ),
                sites_enumeration_augmented,
            )
        )

        return (
            spacegroup_number,
            frozenset(Counter(simplified_stochio).items()),
            sites,
        )

    except Exception as e:
        logger.warning(f"Failed to compute relaxed AFLOW fingerprint: {e}")
        return (0, frozenset(), frozenset())


def record_to_strict_AFLOW_fingerprint(row: Dict[str, Any]) -> Tuple:
    """Compute strict AFLOW-style fingerprint.

    Parameters
    ----------
    row : Dict[str, Any]
        Dictionary containing crystallographic metadata.

    Returns
    -------
    Tuple
        Strict AFLOW fingerprint tuple.
    """
    try:
        spacegroup_number = row["spacegroup_number"]
        elements = row["elements"]
        site_symmetries = row["site_symmetries"]
        sites_enumeration_augmented = row["sites_enumeration_augmented"]

        def count_and_freeze(data):
            return frozenset(Counter(data).items())

        all_variants = []
        for enumeration in sites_enumeration_augmented:
            per_element_wyckoffs = defaultdict(list)
            for element, site_symmetry, site_enumeration in zip(
                elements, site_symmetries, enumeration
            ):
                per_element_wyckoffs[element].append((site_symmetry, site_enumeration))
            all_variants.append(
                count_and_freeze(map(count_and_freeze, per_element_wyckoffs.values()))
            )

        return (spacegroup_number, frozenset(all_variants))

    except Exception as e:
        logger.warning(f"Failed to compute strict AFLOW fingerprint: {e}")
        return (0, frozenset())


# Convenience function for direct structure fingerprinting
def get_augmented_fingerprint(
    structure: Structure, symprec: float = 0.01, angle_tolerance: float = 5.0
) -> Union[str, None]:
    """Get augmented fingerprint for a structure (convenience function).

    Parameters
    ----------
    structure : Structure
        The pymatgen Structure object to fingerprint.
    symprec : float, default=0.01
        Symmetry precision for crystallographic analysis.
    angle_tolerance : float, default=5.0
        Angle tolerance in degrees for crystallographic analysis.

    Returns
    -------
    str or None
        The augmented fingerprint string, or None if analysis failed.
    """
    fingerprinter = AugmentedFingerprinter(
        symprec=symprec, angle_tolerance=angle_tolerance
    )
    return fingerprinter.get_structure_fingerprint(structure)
