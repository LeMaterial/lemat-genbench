"""Crystallographic analysis utilities for structure fingerprinting.

This module provides functions to extract crystallographic metadata from
pymatgen Structure objects, including space group information, Wyckoff positions,
and equivalent site enumerations needed for augmented fingerprinting.
"""

import warnings
from typing import Any, Dict, List

from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from lemat_genbench.utils.logging import logger

# Suppress common warnings from pymatgen
warnings.filterwarnings("ignore", message="No oxidation states specified on sites!")
warnings.filterwarnings("ignore", category=FutureWarning)

class CrystallographicAnalyzer:
    """Analyzer for extracting crystallographic metadata from structures.

    This class provides methods to extract space group information, Wyckoff positions,
    and site multiplicities from pymatgen Structure objects, which are required for
    augmented fingerprinting approaches.
    """

    def __init__(self, symprec: float = 0.01, angle_tolerance: float = 5.0):
        """Initialize the crystallographic analyzer.

        Parameters
        ----------
        symprec : float, default=0.01
            Symmetry precision for spacegroup analysis.
        angle_tolerance : float, default=5.0
            Angle tolerance in degrees for spacegroup analysis.
        """
        self.symprec = symprec
        self.angle_tolerance = angle_tolerance

    def analyze_structure(self, structure: Structure) -> Dict[str, Any]:
        """Extract complete crystallographic metadata from a structure.

        Parameters
        ----------
        structure : Structure
            The pymatgen Structure object to analyze.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - spacegroup_number: int
            - elements: List[str]
            - site_symmetries: List[str] (Wyckoff positions)
            - sites_enumeration_augmented: List[List[int]]
            - multiplicity: List[int]
            - success: bool (whether analysis succeeded)
            - error: str (error message if failed)
        """
        try:
            # Get spacegroup analyzer
            sga = SpacegroupAnalyzer(
                structure, symprec=self.symprec, angle_tolerance=self.angle_tolerance
            )

            # Extract space group number
            spacegroup_number = sga.get_space_group_number()

            # Get symmetrized structure for Wyckoff analysis
            sym_structure = sga.get_symmetrized_structure()

            # Extract elements and site information
            elements = []
            site_symmetries = []
            multiplicity = []

            # Process each equivalent site group
            for i, equiv_sites in enumerate(sym_structure.equivalent_sites):
                # Get element for this site
                element = equiv_sites[0].specie.symbol
                elements.append(element)

                # Get multiplicity
                mult_val = len(equiv_sites)
                multiplicity.append(mult_val)

                # Generate site symmetry label (simplified approach)
                # In a full implementation, this would use actual Wyckoff letters from spglib
                wyckoff_letter = chr(ord("a") + i)
                site_symmetry = f"{mult_val}{wyckoff_letter}"
                site_symmetries.append(site_symmetry)

            # Generate equivalent site enumerations
            # Use the equivalent_indices from the symmetrized structure
            equivalent_indices = sym_structure.equivalent_indices
            sites_enumeration_augmented = self._generate_equivalent_enumerations(
                equivalent_indices
            )

            return {
                "spacegroup_number": spacegroup_number,
                "elements": elements,
                "site_symmetries": site_symmetries,
                "sites_enumeration_augmented": sites_enumeration_augmented,
                "multiplicity": multiplicity,
                "success": True,
                "error": None,
            }

        except Exception as e:
            logger.warning(f"Crystallographic analysis failed: {e}")
            return {
                "spacegroup_number": None,
                "elements": [],
                "site_symmetries": [],
                "sites_enumeration_augmented": [],
                "multiplicity": [],
                "success": False,
                "error": str(e),
            }

    def _generate_equivalent_enumerations(
        self, equivalent_indices: List[List[int]], max_enumerations: int = 1000
    ) -> List[List[int]]:
        """Generate simple equivalent enumerations for Wyckoff positions.

        This generates a small number of key equivalent enumerations rather than
        the full combinatorial explosion. The original pasted code expects these
        to be pre-computed, but we need to generate them from pymatgen analysis.

        Parameters
        ----------
        equivalent_indices : List[List[int]]
            List of lists, where each inner list contains indices of equivalent sites.
        max_enumerations : int, default=1000
            Maximum number of enumerations to generate (not used in simple approach).

        Returns
        -------
        List[List[int]]
            List of simple enumerations for fingerprinting.
        """
        try:
            if not equivalent_indices:
                return [[]]

            total_atoms = sum(len(group) for group in equivalent_indices)

            # Generate simple consecutive enumeration (main approach)
            enumeration = []
            for group in equivalent_indices:
                for i, atom_idx in enumerate(group):
                    enumeration.append(i + 1)  # 1-indexed Wyckoff position within group

            # For now, just return the main enumeration
            # The original code seems to expect simple enumerations, not complex permutations
            return [enumeration]

        except Exception as e:
            logger.warning(f"Failed to generate equivalent enumerations: {e}")
            total_atoms = sum(len(group) for group in equivalent_indices)
            return [[1] * total_atoms]

    # Removed complex _generate_equivalent_enumerations_simplified method
    # Using simple approach directly in main method

    def extract_composition_info(self, structure: Structure) -> Dict[str, Any]:
        """Extract composition-related information from structure.

        Parameters
        ----------
        structure : Structure
            The pymatgen Structure object.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing composition information.
        """
        try:
            composition = structure.composition

            # Convert element objects to strings
            elements = [str(el) for el in composition.element_composition.keys()]
            element_counts = list(composition.element_composition.values())

            return {
                "formula": composition.reduced_formula,
                "elements": elements,
                "element_counts": element_counts,
                "reduced_composition": composition.reduced_composition.as_dict(),
                "num_sites": len(structure),
                "num_elements": len(composition.elements),
            }

        except Exception as e:
            logger.warning(f"Composition analysis failed: {e}")
            return {
                "formula": "Unknown",
                "elements": [],
                "element_counts": [],
                "reduced_composition": {},
                "num_sites": 0,
                "num_elements": 0,
            }


def structure_to_crystallographic_dict(
    structure: Structure, symprec: float = 0.01, angle_tolerance: float = 5.0
) -> Dict[str, Any]:
    """Convert a pymatgen Structure to crystallographic metadata dictionary.

    This is a convenience function that creates an analyzer and extracts
    crystallographic metadata in one call.

    Parameters
    ----------
    structure : Structure
        The pymatgen Structure object to analyze.
    symprec : float, default=0.01
        Symmetry precision for spacegroup analysis.
    angle_tolerance : float, default=5.0
        Angle tolerance in degrees for spacegroup analysis.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing crystallographic metadata.
    """
    analyzer = CrystallographicAnalyzer(
        symprec=symprec, angle_tolerance=angle_tolerance
    )
    crystal_data = analyzer.analyze_structure(structure)
    composition_data = analyzer.extract_composition_info(structure)

    # Combine both datasets
    result = {**crystal_data, **composition_data}
    return result


def lematbulk_item_to_structure(item: Dict[str, Any]) -> Structure:
    """Convert a LeMat-Bulk dataset item to a pymatgen Structure.

    Parameters
    ----------
    item : Dict[str, Any]
        LeMat-Bulk dataset item containing structure information.

    Returns
    -------
    Structure
        The pymatgen Structure object.
    """
    try:
        # Extract structure data from LeMat-Bulk format
        lattice_vectors = item["lattice_vectors"]
        species_at_sites = item["species_at_sites"]
        cartesian_positions = item["cartesian_site_positions"]

        # Create Structure object
        structure = Structure(
            lattice=lattice_vectors,
            species=species_at_sites,
            coords=cartesian_positions,
            coords_are_cartesian=True,
        )

        return structure

    except Exception as e:
        logger.error(f"Failed to convert LeMat-Bulk item to Structure: {e}")
        raise ValueError(f"Invalid LeMat-Bulk item format: {e}")


def analyze_lematbulk_item(
    item: Dict[str, Any], symprec: float = 0.01, angle_tolerance: float = 5.0
) -> Dict[str, Any]:
    """Extract crystallographic metadata directly from LeMat-Bulk item.

    This function combines structure conversion and crystallographic analysis
    for efficient processing of LeMat-Bulk dataset items.

    Parameters
    ----------
    item : Dict[str, Any]
        LeMat-Bulk dataset item.
    symprec : float, default=0.01
        Symmetry precision for spacegroup analysis.
    angle_tolerance : float, default=5.0
        Angle tolerance in degrees for spacegroup analysis.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing crystallographic metadata plus original immutable_id.
    """
    try:
        # Convert to structure
        structure = lematbulk_item_to_structure(item)

        # Analyze crystallography
        result = structure_to_crystallographic_dict(
            structure, symprec=symprec, angle_tolerance=angle_tolerance
        )

        # Add original identifier
        result["immutable_id"] = item.get("immutable_id", "unknown")
        result["original_item"] = item  # Keep reference for debugging

        return result

    except Exception as e:
        logger.error(f"Failed to analyze LeMat-Bulk item: {e}")
        return {
            "immutable_id": item.get("immutable_id", "unknown"),
            "success": False,
            "error": str(e),
            "spacegroup_number": None,
            "elements": [],
            "site_symmetries": [],
            "sites_enumeration_augmented": [],
            "multiplicity": [],
        }
