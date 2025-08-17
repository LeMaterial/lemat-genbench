"""Crystallographic analysis utilities for structure fingerprinting.

This module provides functions to extract crystallographic metadata from
pymatgen Structure objects, including space group information, Wyckoff positions,
and equivalent site enumerations needed for augmented fingerprinting.
"""

import warnings
from itertools import product
from typing import Any, Dict, List

import spglib
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
            # Get spacegroup analyzer for fallback
            sga = SpacegroupAnalyzer(
                structure, symprec=self.symprec, angle_tolerance=self.angle_tolerance
            )
            spacegroup_number = sga.get_space_group_number()

            # Prepare structure for spglib analysis
            lattice = structure.lattice.matrix
            positions = structure.frac_coords
            numbers = [site.specie.Z for site in structure]

            # Get spglib symmetry dataset
            cell = (lattice, positions, numbers)
            dataset = spglib.get_symmetry_dataset(cell, symprec=self.symprec)

            if dataset is None:
                raise ValueError("spglib failed to analyze structure")

            # Extract Wyckoff information
            wyckoffs = dataset["wyckoffs"]
            equivalent_atoms = dataset["equivalent_atoms"]

            # Group atoms by equivalent positions
            equiv_groups = {}
            for i, equiv_idx in enumerate(equivalent_atoms):
                if equiv_idx not in equiv_groups:
                    equiv_groups[equiv_idx] = []
                equiv_groups[equiv_idx].append(i)

            # Extract information for each unique Wyckoff position
            elements = []
            site_symmetries = []
            multiplicity = []
            equivalent_indices = []

            # Process each unique equivalent atom group
            processed_groups = set()
            for equiv_idx in sorted(equiv_groups.keys()):
                if equiv_idx in processed_groups:
                    continue

                atom_indices = equiv_groups[equiv_idx]
                equivalent_indices.append(atom_indices)

                # Get element and Wyckoff letter for this group
                atom_idx = atom_indices[0]  # Representative atom
                element = structure[atom_idx].specie.symbol
                wyckoff_letter = wyckoffs[atom_idx]
                mult_val = len(atom_indices)

                elements.append(element)
                site_symmetries.append(f"{mult_val}{wyckoff_letter}")
                multiplicity.append(mult_val)

                processed_groups.add(equiv_idx)

            # Generate equivalent site enumerations
            sites_enumeration_augmented = self._generate_equivalent_enumerations(
                equivalent_indices, max_enumerations=50
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
        self, equivalent_indices: List[List[int]], max_enumerations: int = 50
    ) -> List[List[int]]:
        """Generate equivalent enumerations for Wyckoff positions.

        This generates multiple valid enumerations that respect the symmetry
        relationships between equivalent sites. For simple cases with few possibilities,
        generates all of them. For complex cases, generates a representative sample.

        Parameters
        ----------
        equivalent_indices : List[List[int]]
            List of lists, where each inner list contains indices of equivalent sites.
        max_enumerations : int, default=50
            Maximum number of enumerations to generate.

        Returns
        -------
        List[List[int]]
            List of enumerations for fingerprinting.
        """
        try:
            if not equivalent_indices:
                return [[]]

            total_atoms = sum(len(group) for group in equivalent_indices)
            if total_atoms == 0:
                return [[]]

            # Generate enumerations for each group
            group_enumerations = []
            for group in equivalent_indices:
                group_size = len(group)
                group_ops = self._generate_group_operations(group_size)
                group_enumerations.append(group_ops)

            # Calculate total possible combinations
            total_combinations = 1
            for group_ops in group_enumerations:
                total_combinations *= len(group_ops)

            # If total combinations is small, generate them all
            if total_combinations <= max_enumerations:
                return self._generate_all_combinations(group_enumerations)
            else:
                # For large numbers, sample strategically
                return self._generate_sampled_combinations(
                    group_enumerations, max_enumerations
                )

        except Exception as e:
            logger.warning(f"Failed to generate equivalent enumerations: {e}")
            total_atoms = sum(len(group) for group in equivalent_indices)
            # Simple fallback: just base enumeration
            if total_atoms > 0:
                base_enum = []
                for group in equivalent_indices:
                    for i in range(len(group)):
                        base_enum.append(i + 1)
                return [base_enum]
            else:
                return [[]]

    def _generate_group_operations(self, group_size: int) -> List[List[int]]:
        """Generate symmetry operations for a single equivalent group.

        Parameters
        ----------
        group_size : int
            Size of the equivalent group.

        Returns
        -------
        List[List[int]]
            List of enumerations for this group.
        """
        if group_size == 0:
            return [[]]

        if group_size == 1:
            return [[1]]

        base = list(range(1, group_size + 1))
        operations = []

        # 1. Identity
        operations.append(base[:])

        # 2. Reverse
        operations.append(base[::-1])

        # 3. Rotations (cyclic shifts)
        for shift in range(1, group_size):
            rotated = base[shift:] + base[:shift]
            if rotated not in operations:
                operations.append(rotated)

        # 4. For even groups, add pairwise swaps
        if group_size >= 2 and group_size % 2 == 0:
            swapped = []
            for i in range(0, group_size, 2):
                swapped.extend([base[i + 1], base[i]])
            if swapped not in operations:
                operations.append(swapped)

        # 5. For groups of 4+, add some reflections
        if group_size == 4:
            # Diagonal reflections for 2x2 arrangement
            operations.append([base[0], base[2], base[1], base[3]])  # transpose
            operations.append([base[3], base[1], base[2], base[0]])  # anti-diagonal

        # Remove duplicates
        unique_operations = []
        seen = set()
        for op in operations:
            op_tuple = tuple(op)
            if op_tuple not in seen:
                unique_operations.append(op)
                seen.add(op_tuple)

        return unique_operations

    def _generate_all_combinations(
        self, group_enumerations: List[List[List[int]]]
    ) -> List[List[int]]:
        """Generate all possible combinations when the total is manageable.

        Parameters
        ----------
        group_enumerations : List[List[List[int]]]
            Enumerations for each group.

        Returns
        -------
        List[List[int]]
            All possible full structure enumerations.
        """
        all_combinations = []

        for combination in product(*group_enumerations):
            full_enum = []
            for group_enum in combination:
                full_enum.extend(group_enum)
            all_combinations.append(full_enum)

        return all_combinations

    def _generate_sampled_combinations(
        self, group_enumerations: List[List[List[int]]], max_enumerations: int
    ) -> List[List[int]]:
        """Generate a strategic sample when there are too many combinations.

        Parameters
        ----------
        group_enumerations : List[List[List[int]]]
            Enumerations for each group.
        max_enumerations : int
            Maximum number of enumerations to generate.

        Returns
        -------
        List[List[int]]
            Sampled full structure enumerations.
        """
        sampled_combinations = []

        # Strategy 1: Always include identity (first operation from each group)
        identity_combination = [group_ops[0] for group_ops in group_enumerations]
        full_enum = []
        for group_enum in identity_combination:
            full_enum.extend(group_enum)
        sampled_combinations.append(full_enum)

        # Strategy 2: Include combinations with single group variations
        for group_idx, group_ops in enumerate(group_enumerations):
            for op_idx in range(1, min(len(group_ops), 4)):  # Take first few operations
                combination = [
                    group_ops[0] for group_ops in group_enumerations
                ]  # Start with identity
                combination[group_idx] = group_ops[op_idx]  # Vary one group

                full_enum = []
                for group_enum in combination:
                    full_enum.extend(group_enum)

                if full_enum not in sampled_combinations:
                    sampled_combinations.append(full_enum)
                    if len(sampled_combinations) >= max_enumerations:
                        return sampled_combinations

        # Strategy 3: Include some combinations with multiple group variations
        import random

        random.seed(42)  # For reproducibility

        attempts = 0
        while (
            len(sampled_combinations) < max_enumerations
            and attempts < max_enumerations * 2
        ):
            combination = []
            for group_ops in group_enumerations:
                combination.append(random.choice(group_ops))

            full_enum = []
            for group_enum in combination:
                full_enum.extend(group_enum)

            if full_enum not in sampled_combinations:
                sampled_combinations.append(full_enum)

            attempts += 1

        return sampled_combinations

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