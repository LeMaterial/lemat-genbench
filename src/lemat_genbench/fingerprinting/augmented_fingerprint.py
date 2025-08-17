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
            crystal_data = structure_to_crystallographic_dict(
                structure, 
                symprec=self.analyzer.symprec, 
                angle_tolerance=self.analyzer.angle_tolerance
            )

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
            for variant in sorted(wyckoff_variants, key=str):
                # Convert frozenset of Counter items to sorted string
                variant_items = sorted(list(variant), key=str)
                variant_str = "_".join([f"{item[0]}:{item[1]}" for item in variant_items])
                variant_strs.append(variant_str)

            variant_strs.sort()  # Ensure consistent ordering
            variants_combined = "|".join(variant_strs)

            return f"AUG_{spacegroup}_{variants_combined}"

        except Exception as e:
            logger.warning(f"Failed to convert fingerprint to string: {e}")
            return f"AUG_FAILED_{hash(str(fingerprint))}"


def record_to_augmented_fingerprint(row: Dict[str, Any]) -> Tuple:
    """Compute augmented fingerprint from crystallographic metadata.

    This implementation properly handles the improved crystallographic data
    from spglib analysis with accurate Wyckoff positions and enumerations.

    Parameters
    ----------
    row : Dict[str, Any]
        Dictionary containing crystallographic metadata:
        - spacegroup_number: int
        - elements: List[str]
        - site_symmetries: List[str] (Wyckoff positions in format "4a", "8c", etc.)
        - sites_enumeration_augmented: List[List[int]]

    Returns
    -------
    Tuple
        Fingerprint tuple (spacegroup_number, frozenset of Wyckoff representations)
    """
    try:
        # Handle spacegroup number with type conversion
        spacegroup_raw = row.get("spacegroup_number", 0)
        try:
            spacegroup_number = int(spacegroup_raw) if spacegroup_raw is not None else 0
        except (ValueError, TypeError):
            spacegroup_number = 0
            
        elements = row.get("elements", [])
        site_symmetries = row.get("site_symmetries", [])
        sites_enumeration_augmented = row.get("sites_enumeration_augmented", [[]])

        # Validate input data
        if not all([elements, site_symmetries]):
            logger.warning("Missing required crystallographic data for fingerprinting")
            return (spacegroup_number, frozenset())

        # Ensure all lists have compatible lengths
        min_length = min(len(elements), len(site_symmetries))
        if min_length == 0:
            logger.warning("Empty elements or site_symmetries list")
            return (spacegroup_number, frozenset())

        # Truncate to minimum length for consistency
        elements = elements[:min_length]
        site_symmetries = site_symmetries[:min_length]

        # Handle empty or invalid enumerations
        if not sites_enumeration_augmented or all(not enum for enum in sites_enumeration_augmented):
            logger.warning("Empty or invalid enumerations, using default")
            sites_enumeration_augmented = [[1] * min_length]

        # Generate all possible Wyckoff representations from enumerations
        wyckoff_variants = set()
        for enumeration in sites_enumeration_augmented:
            # Ensure enumeration matches element count
            if len(enumeration) < min_length:
                # Pad enumeration if needed
                enumeration = enumeration + [1] * (min_length - len(enumeration))
            elif len(enumeration) > min_length:
                # Truncate if too long
                enumeration = enumeration[:min_length]
            
            # Create triplets of (element, site_symmetry, enumeration_value)
            triplets = []
            for elem, site_sym, enum_val in zip(elements, site_symmetries, enumeration):
                # Ensure proper data types
                elem_str = str(elem)
                site_sym_str = str(site_sym)
                
                # Handle enumeration value conversion
                try:
                    enum_int = int(float(enum_val)) if enum_val is not None else 1
                    enum_int = max(1, enum_int)  # Ensure positive
                except (ValueError, TypeError):
                    enum_int = 1
                
                triplets.append((elem_str, site_sym_str, enum_int))
            
            # Count occurrences of each unique triplet
            triplet_counts = Counter(triplets)
            
            # Convert to frozenset for hashing
            variant = frozenset(triplet_counts.items())
            wyckoff_variants.add(variant)

        return (spacegroup_number, frozenset(wyckoff_variants))

    except Exception as e:
        logger.warning(f"Failed to compute augmented fingerprint: {e}")
        # Return a fallback fingerprint with proper type conversion
        spacegroup_raw = row.get("spacegroup_number", 0)
        try:
            spacegroup_number = int(spacegroup_raw) if spacegroup_raw is not None else 0
        except (ValueError, TypeError):
            spacegroup_number = 0
        return (spacegroup_number, frozenset())


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
        # Handle spacegroup number with type conversion
        spacegroup_raw = row.get("spacegroup_number", 0)
        try:
            spacegroup_number = int(spacegroup_raw) if spacegroup_raw is not None else 0
        except (ValueError, TypeError):
            spacegroup_number = 0
            
        site_symmetries = row.get("site_symmetries", [])
        sites_enumeration_augmented = row.get("sites_enumeration_augmented", [[]])

        if not site_symmetries:
            logger.warning("Empty site_symmetries for anonymous fingerprint")
            return (spacegroup_number, frozenset())

        # Handle empty enumerations
        if not sites_enumeration_augmented or all(not enum for enum in sites_enumeration_augmented):
            sites_enumeration_augmented = [[1] * len(site_symmetries)]

        # Generate structural variants without element information
        structural_variants = set()
        for enumeration in sites_enumeration_augmented:
            # Ensure enumeration matches site_symmetries count
            if len(enumeration) < len(site_symmetries):
                enumeration = enumeration + [1] * (len(site_symmetries) - len(enumeration))
            elif len(enumeration) > len(site_symmetries):
                enumeration = enumeration[:len(site_symmetries)]
            
            # Create pairs of (site_symmetry, enumeration_value)
            pairs = []
            for site_sym, enum_val in zip(site_symmetries, enumeration):
                site_sym_str = str(site_sym)
                try:
                    enum_int = int(float(enum_val)) if enum_val is not None else 1
                    enum_int = max(1, enum_int)  # Ensure positive
                except (ValueError, TypeError):
                    enum_int = 1
                pairs.append((site_sym_str, enum_int))
            
            pair_counts = Counter(pairs)
            variant = frozenset(pair_counts.items())
            structural_variants.add(variant)

        return (spacegroup_number, frozenset(structural_variants))

    except Exception as e:
        logger.warning(f"Failed to compute anonymous fingerprint: {e}")
        spacegroup_raw = row.get("spacegroup_number", 0)
        try:
            spacegroup_number = int(spacegroup_raw) if spacegroup_raw is not None else 0
        except (ValueError, TypeError):
            spacegroup_number = 0
        return (spacegroup_number, frozenset())


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
        # Handle spacegroup number with type conversion
        spacegroup_raw = row.get("spacegroup_number", 0)
        try:
            spacegroup_number = int(spacegroup_raw) if spacegroup_raw is not None else 0
        except (ValueError, TypeError):
            spacegroup_number = 0
            
        elements = row.get("elements", [])
        multiplicity = row.get("multiplicity", [])
        site_symmetries = row.get("site_symmetries", [])
        sites_enumeration_augmented = row.get("sites_enumeration_augmented", [[]])

        # Validate inputs
        if not elements:
            logger.warning("Empty elements list for AFLOW fingerprint")
            return (spacegroup_number, frozenset(), frozenset())

        # Calculate simplified stoichiometry
        element_counts = defaultdict(int)
        mult_list = multiplicity if multiplicity else [1] * len(elements)
        
        for element, mult in zip(elements, mult_list):
            try:
                mult_int = int(float(mult)) if mult is not None else 1
            except (ValueError, TypeError):
                mult_int = 1
            element_counts[str(element)] += mult_int

        if element_counts:
            stochio_gcd = gcd(*element_counts.values()) if len(element_counts.values()) > 1 else list(element_counts.values())[0]
            simplified_stochio = [mult // stochio_gcd for mult in element_counts.values()]
        else:
            simplified_stochio = []

        # Handle empty enumerations
        if not sites_enumeration_augmented or all(not enum for enum in sites_enumeration_augmented):
            sites_enumeration_augmented = [[1] * len(site_symmetries)] if site_symmetries else [[]]

        # Generate Wyckoff site information
        sites = set()
        for enumeration in sites_enumeration_augmented:
            if not site_symmetries:
                continue
                
            if len(enumeration) < len(site_symmetries):
                enumeration = enumeration + [1] * (len(site_symmetries) - len(enumeration))
            elif len(enumeration) > len(site_symmetries):
                enumeration = enumeration[:len(site_symmetries)]
            
            pairs = []
            for site_sym, enum_val in zip(site_symmetries, enumeration):
                site_sym_str = str(site_sym)
                try:
                    enum_int = int(float(enum_val)) if enum_val is not None else 1
                    enum_int = max(1, enum_int)  # Ensure positive
                except (ValueError, TypeError):
                    enum_int = 1
                pairs.append((site_sym_str, enum_int))
            
            pair_counts = Counter(pairs)
            site_variant = frozenset(pair_counts.items())
            sites.add(site_variant)

        return (
            spacegroup_number,
            frozenset(Counter(simplified_stochio).items()) if simplified_stochio else frozenset(),
            frozenset(sites),
        )

    except Exception as e:
        logger.warning(f"Failed to compute relaxed AFLOW fingerprint: {e}")
        spacegroup_raw = row.get("spacegroup_number", 0)
        try:
            spacegroup_number = int(spacegroup_raw) if spacegroup_raw is not None else 0
        except (ValueError, TypeError):
            spacegroup_number = 0
        return (spacegroup_number, frozenset(), frozenset())


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
        # Handle spacegroup number with type conversion
        spacegroup_raw = row.get("spacegroup_number", 0)
        try:
            spacegroup_number = int(spacegroup_raw) if spacegroup_raw is not None else 0
        except (ValueError, TypeError):
            spacegroup_number = 0
            
        elements = row.get("elements", [])
        site_symmetries = row.get("site_symmetries", [])
        sites_enumeration_augmented = row.get("sites_enumeration_augmented", [[]])

        if not all([elements, site_symmetries]):
            logger.warning("Missing elements or site_symmetries for strict AFLOW fingerprint")
            return (spacegroup_number, frozenset())

        # Ensure consistent lengths
        min_length = min(len(elements), len(site_symmetries))
        elements = elements[:min_length]
        site_symmetries = site_symmetries[:min_length]

        # Handle empty enumerations
        if not sites_enumeration_augmented or all(not enum for enum in sites_enumeration_augmented):
            sites_enumeration_augmented = [[1] * min_length]

        def count_and_freeze(data):
            return frozenset(Counter(data).items())

        all_variants = []
        for enumeration in sites_enumeration_augmented:
            # Ensure enumeration matches element count
            if len(enumeration) < min_length:
                enumeration = enumeration + [1] * (min_length - len(enumeration))
            elif len(enumeration) > min_length:
                enumeration = enumeration[:min_length]
            
            per_element_wyckoffs = defaultdict(list)
            for element, site_symmetry, site_enumeration in zip(
                elements, site_symmetries, enumeration
            ):
                elem_str = str(element)
                site_sym_str = str(site_symmetry)
                try:
                    enum_int = int(float(site_enumeration)) if site_enumeration is not None else 1
                    enum_int = max(1, enum_int)  # Ensure positive
                except (ValueError, TypeError):
                    enum_int = 1
                per_element_wyckoffs[elem_str].append((site_sym_str, enum_int))
            
            # Convert each element's Wyckoff list to frozen counter
            element_wyckoff_counters = []
            for element, wyckoff_list in per_element_wyckoffs.items():
                element_wyckoff_counters.append(count_and_freeze(wyckoff_list))
            
            variant = count_and_freeze(element_wyckoff_counters)
            all_variants.append(variant)

        return (spacegroup_number, frozenset(all_variants))

    except Exception as e:
        logger.warning(f"Failed to compute strict AFLOW fingerprint: {e}")
        spacegroup_raw = row.get("spacegroup_number", 0)
        try:
            spacegroup_number = int(spacegroup_raw) if spacegroup_raw is not None else 0
        except (ValueError, TypeError):
            spacegroup_number = 0
        return (spacegroup_number, frozenset())


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