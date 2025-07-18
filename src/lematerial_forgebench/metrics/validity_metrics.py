"""Validity metrics for evaluating material structures.

This module implements fundamental validity metrics that ensure generated
structures are physically meaningful and chemically plausible.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from pymatgen.analysis.bond_valence import BVAnalyzer, calculate_bv_sum
from pymatgen.analysis.local_env import (
    CrystalNN,
    VoronoiNN,
    get_neighbors_of_site_with_index,
)
from pymatgen.core import Composition
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifParser, CifWriter

from lematerial_forgebench.metrics.base import BaseMetric, MetricConfig, MetricResult
from lematerial_forgebench.utils.logging import logger
from lematerial_forgebench.utils.oxidation_state import (
    compositional_oxi_state_guesses,
    get_inequivalent_site_info,
)


@dataclass
class ChargeNeutralityConfig(MetricConfig):
    """Configuration for the ChargeNeutrality metric.

    Parameters
    ----------
    tolerance : float, default=0.1
        Tolerance for deviations from charge neutrality.
    strict : bool, default=False
        If True, oxidation states must be determinable for all atoms.
        If False, will attempt to calculate oxidation states but pass
        the structure if calculation fails.
    """

    tolerance: float = 0.1
    strict: bool = False


class ChargeNeutralityMetric(BaseMetric):
    """Evaluate charge neutrality of a crystal structure.

    This metric checks if the sum of oxidation states of atoms in the
    structure is close enough to zero, ensuring charge balance.

    Parameters
    ----------
    tolerance : float, default=0.1
        Tolerance for deviations from charge neutrality.
    strict : bool, default=False
        If True, oxidation states must be determinable for all atoms.
    name : str, optional
        Custom name for the metric.
    description : str, optional
        Description of what the metric measures.
    lower_is_better : bool, default=True
        Lower values indicate better charge neutrality.
    n_jobs : int, default=1
        Number of parallel jobs to run.
    """

    def __init__(
        self,
        tolerance: float = 0.1,
        strict: bool = False,
        name: str | None = None,
        description: str | None = None,
        lower_is_better: bool = True,
        n_jobs: int = 1,
    ):
        super().__init__(
            name=name or "ChargeNeutrality",
            description=description
            or "Measures how close a structure is to charge neutrality",
            lower_is_better=lower_is_better,
            n_jobs=n_jobs,
        )
        self.config = ChargeNeutralityConfig(
            name=self.config.name,
            description=self.config.description,
            lower_is_better=self.config.lower_is_better,
            n_jobs=self.config.n_jobs,
            tolerance=tolerance,
            strict=strict,
        )
        self.bv_analyzer = BVAnalyzer()

    def _get_compute_attributes(self) -> dict[str, Any]:
        """Get the attributes for the compute_structure method."""
        return {
            "tolerance": self.config.tolerance,
            "strict": self.config.strict,
            "bv_analyzer": self.bv_analyzer,
        }

    @staticmethod
    def compute_structure(
        structure: Structure, tolerance: float, strict: bool, bv_analyzer: BVAnalyzer
    ) -> float:
        """Compute the deviation from charge neutrality for a structure.

        Parameters
        ----------
        structure : Structure
            A pymatgen Structure object to evaluate.
        tolerance : float
            Tolerance for charge neutrality.
        strict : bool
            Whether to require determinable oxidation states.
        bv_analyzer : BVAnalyzer
            Bond valence analyzer for determining oxidation states.

        Returns
        -------
        float
            The absolute deviation from charge neutrality.
            0.0 means perfectly neutral, larger values indicate charge imbalance.
        """

        sites = get_inequivalent_site_info(structure)
        bvs = []
        count = 0
        for site_index in sites["sites"]:
            nn_list = get_neighbors_of_site_with_index(structure, site_index)
            bvs.append(
                [
                    sites["species"][count],
                    calculate_bv_sum(structure[site_index], nn_list),
                    sites["multiplicities"][count],
                ]
            )
            count += 1

        try:
            for bv in bvs:
                if np.abs(bv[1]) < 10**-15:
                    pass
                else:
                    raise ValueError
            print(
                "Valid structure - Metallic structure with a bond valence equal to zero for all atoms"
            )
            return 1.0
        except ValueError:
            # this means the bv_sum calculation has predicted this structure is NOT metallic. Therefore, we can try and assign oxidation states using PMG's
            # oxidation state functions, which do not return oxidation states for metallic structures.
            logger.warning(
                "the bond valence sum calculation yielded values that were not zero meaning this is not predicted to be a metallic structure"
            )

            try:
                # Try to determine oxidation states - good first pass, if this can be done within pymatgen, it will almost certainly be a structure that is charge balanced
                structure_with_oxi = bv_analyzer.get_oxi_state_decorated_structure(
                    structure
                )
                charge_sum = sum(
                    site.specie.oxi_state for site in structure_with_oxi.sites
                )
                print(
                    "Valid structure - charge balanced based on Pymatgen's get_oxi_state_decorated_structure function, which almost always returns "
                    "reasonable oxidation states"
                )
                if charge_sum == 0.0:
                    return 1.0
                else:
                    return 0.0
            except ValueError:
                # get_oxi_state_decorated_structure fails when structures and compositions are outside the distribution of the Materials Project.
                # We will now need to determine if this composition has the ability to be charged balanced using a reasonable combination of oxidation states.
                logger.warning(
                    "Could not determine oxidation states using get_oxi_state_decorated_structure"
                )

                comp = Composition(structure.composition)
                here = Path(__file__).resolve().parent
                three_up = here.parents[2]
                with open(three_up / "data" / "oxi_state_mapping.json", "r") as f:
                    oxi_state_mapping = json.load(f)
                oxi_states_override = {}
                for e in comp.elements:
                    oxi_states_override[str(e)] = oxi_state_mapping[str(e)]
                output = compositional_oxi_state_guesses(
                    comp,
                    all_oxi_states=False,
                    max_sites=-1,
                    target_charge=0,
                    oxi_states_override=oxi_states_override,
                )
                print(
                    "Most valid oxidation state and score based on composition",
                    output[1][0],
                    output[2][0],
                )
                try:
                    score = output[2][0]
                    if score > 0.001:
                        return 1.0
                    else:
                        return float(
                            0.0
                        )  # TODO decide on a function to make this continuous based on LeMatBulk statistics (and scale with other metrics!)
                except IndexError:
                    return float(0.0)

    def aggregate_results(self, values: list[float]) -> Dict[str, Any]:
        """Aggregate results into final metric values.

        Parameters
        ----------
        values : list[float]
            Absolute deviations from charge neutrality for each structure.

        Returns
        -------
        dict
            Dictionary with aggregated metrics.
        """
        # Filter out NaN values
        valid_values = [v for v in values if not np.isnan(v)]

        if not valid_values:
            return {
                "metrics": {
                    "charge_neutrality_error": float("nan"),
                    "charge_neutral_ratio": 0.0,
                },
                "primary_metric": "charge_neutrality_error",
                "uncertainties": {},
            }

        # Count how many structures are within tolerance
        within_tolerance = sum(1 for v in valid_values if v <= self.config.tolerance)
        charge_neutral_ratio = within_tolerance / len(valid_values)

        # Calculate mean absolute deviation
        mean_abs_deviation = np.mean(valid_values)

        return {
            "metrics": {
                "charge_neutrality_error": mean_abs_deviation,
                "charge_neutral_ratio": charge_neutral_ratio,
            },
            "primary_metric": "charge_neutrality_error",
            "uncertainties": {
                "charge_neutrality_error": {
                    "std": np.std(valid_values) if len(valid_values) > 1 else 0.0
                }
            },
        }


@dataclass
class MinimumInteratomicDistanceConfig(MetricConfig):
    """Configuration for the MinimumInteratomicDistance metric.

    Parameters
    ----------
    element_radii : dict[str, float] | None, default=None
        Custom atomic radii for elements in Angstroms. If None,
        default values from pymatgen will be used.
    scaling_factor : float, default=0.5
        Factor to scale the minimum distance (sum of atomic radii).
        Values below 1.0 allow atoms to be closer than the sum of
        their atomic radii.
    """

    element_radii: Dict[str, float] | None = None
    scaling_factor: float = 0.5


class MinimumInteratomicDistanceMetric(BaseMetric):
    """Evaluate if structures have reasonable interatomic distances.

    This metric checks if the minimum distance between any pair of
    atoms in the structure exceeds a threshold based on atomic radii.

    Parameters
    ----------
    element_radii : dict[str, float] | None, default=None
        Custom atomic radii for elements.
    scaling_factor : float, default=0.5
        Factor to scale the minimum distance threshold.
    name : str, optional
        Custom name for the metric.
    description : str, optional
        Description of what the metric measures.
    lower_is_better : bool, default=False
        Higher values indicate better adherence to minimum distance constraints.
    n_jobs : int, default=1
        Number of parallel jobs to run.
    """

    def __init__(
        self,
        element_radii: Dict[str, float] | None = None,
        scaling_factor: float = 0.5,
        name: str | None = None,
        description: str | None = None,
        lower_is_better: bool = False,
        n_jobs: int = 1,
    ):
        super().__init__(
            name=name or "MinimumInteratomicDistance",
            description=description
            or "Evaluates if interatomic distances are physically reasonable",
            lower_is_better=lower_is_better,
            n_jobs=n_jobs,
        )
        self.config = MinimumInteratomicDistanceConfig(
            name=self.config.name,
            description=self.config.description,
            lower_is_better=self.config.lower_is_better,
            n_jobs=self.config.n_jobs,
            element_radii=element_radii,
            scaling_factor=scaling_factor,
        )

        # Initialize default element radii if not provided
        if element_radii is None:
            self.element_radii = {
                str(el): el.atomic_radius
                for el in Element
                if el.atomic_radius is not None
            }
        else:
            self.element_radii = element_radii

    def _get_compute_attributes(self) -> dict[str, Any]:
        """Get the attributes for the compute_structure method."""
        return {
            "element_radii": self.element_radii,
            "scaling_factor": self.config.scaling_factor,
        }

    @staticmethod
    def compute_structure(
        structure: Structure, element_radii: Dict[str, float], scaling_factor: float
    ) -> float:
        """Check if a structure has reasonable interatomic distances.

        Parameters
        ----------
        structure : Structure
            A pymatgen Structure object to evaluate.
        element_radii : dict[str, float]
            Dictionary mapping element symbols to atomic radii in Angstroms.
        scaling_factor : float
            Factor to scale the minimum allowed distance.

        Returns
        -------
        float
            1.0 if all interatomic distances are reasonable, 0.0 otherwise.
            In case of missing radii data, returns 0.5.
        """
        # Get all pairs of sites
        all_distances = structure.distance_matrix
        n_sites = len(structure)

        # Create a mask to identify which pairs to check
        # (we only check upper triangle to avoid duplicate pairs)
        mask = np.triu(np.ones((n_sites, n_sites), dtype=bool), k=1)

        # Extract only the distances we want to check
        distances_to_check = all_distances[mask]

        # No distances to check (single atom structure)
        if len(distances_to_check) == 0:
            return 0.0

        # Check if we have radius data for all elements in the structure
        elements_in_structure = {str(site.specie) for site in structure}
        missing_elements = elements_in_structure - set(element_radii.keys())

        if missing_elements:
            logger.warning(f"Missing radius data for elements: {missing_elements}")
            return 0.0  # Invalid because we can't fully check

        # For each pair of sites, compute the minimum allowed distance
        valid_pairs = 0
        total_pairs = 0

        for i in range(n_sites):
            for j in range(i + 1, n_sites):
                total_pairs += 1
                element_i = str(structure[i].specie)
                element_j = str(structure[j].specie)

                # Sum of atomic radii
                min_dist = (
                    0.7 + element_radii[element_i] + element_radii[element_j]
                ) * scaling_factor
                actual_dist = all_distances[i, j]

                if actual_dist >= min_dist:
                    valid_pairs += 1
                else:
                    logger.debug(
                        f"Distance between {element_i} and {element_j} is {actual_dist:.3f} Å, "
                        f"which is less than the minimum allowed {min_dist:.3f} Å"
                    )

        # Return the ratio of valid pairs
        if valid_pairs / total_pairs == 1.0:
            return 1.0
        else:
            return 0.0

    def aggregate_results(self, values: list[float]) -> Dict[str, Any]:
        """Aggregate results into final metric values.

        Parameters
        ----------
        values : list[float]
            Values indicating validity of interatomic distances for each structure.

        Returns
        -------
        dict
            Dictionary with aggregated metrics.
        """
        # Filter out NaN values
        valid_values = [v for v in values if not np.isnan(v)]

        if not valid_values:
            return {
                "metrics": {"min_distance_score": 0.0, "valid_structures_ratio": 0.0},
                "primary_metric": "min_distance_score",
                "uncertainties": {},
            }

        # Mean score across all structures
        mean_score = np.mean(valid_values)

        # Count structures with perfect distance validity
        perfect_structures = sum(1 for v in valid_values if v >= 0.999)
        valid_structures_ratio = perfect_structures / len(valid_values)

        return {
            "metrics": {
                "min_distance_score": mean_score,
                "valid_structures_ratio": valid_structures_ratio,
            },
            "primary_metric": "min_distance_score",
            "uncertainties": {
                "min_distance_score": {
                    "std": np.std(valid_values) if len(valid_values) > 1 else 0.0
                }
            },
        }


@dataclass
class CoordinationEnvironmentConfig(MetricConfig):
    """Configuration for the CoordinationEnvironment metric.

    Parameters
    ----------
    environment_database : dict[str, list[int]] | None, default=None
        Expected coordination numbers for each element.
        If None, default values will be used.
    nn_method : str, default="crystalnn"
        Method to determine nearest neighbors.
        Options: "crystalnn", "voronoinn".
    tolerance : float, default=0.2
        Tolerance for coordination number matching.
    """

    environment_database: Dict[str, List[int]] | None = None
    nn_method: str = "crystalnn"
    tolerance: float = 0.2


class CoordinationEnvironmentMetric(BaseMetric):
    """Evaluate if coordination environments match chemical expectations.

    This metric checks if the coordination numbers of atoms in the
    structure match expected values for each element.

    Parameters
    ----------
    environment_database : dict[str, list[int]] | None, default=None
        Expected coordination numbers for each element.
    nn_method : str, default="crystalnn"
        Method to determine nearest neighbors.
    tolerance : float, default=0.2
        Tolerance for coordination number matching.
    name : str, optional
        Custom name for the metric.
    description : str, optional
        Description of what the metric measures.
    lower_is_better : bool, default=False
        Higher values indicate better coordination environments.
    n_jobs : int, default=1
        Number of parallel jobs to run.
    """

    def __init__(
        self,
        environment_database: Dict[str, List[int]] | None = None,
        nn_method: str = "crystalnn",
        tolerance: float = 0.2,
        name: str | None = None,
        description: str | None = None,
        lower_is_better: bool = False,
        n_jobs: int = 1,
    ):
        super().__init__(
            name=name or "CoordinationEnvironment",
            description=description
            or "Evaluates if coordination environments match chemical expectations",
            lower_is_better=lower_is_better,
            n_jobs=n_jobs,
        )
        self.config = CoordinationEnvironmentConfig(
            name=self.config.name,
            description=self.config.description,
            lower_is_better=self.config.lower_is_better,
            n_jobs=self.config.n_jobs,
            environment_database=environment_database,
            nn_method=nn_method,
            tolerance=tolerance,
        )

        # Initialize default environment database if not provided
        if environment_database is None:
            self.environment_database = self._get_default_environment_database()
        else:
            self.environment_database = environment_database

    def _get_default_environment_database(self) -> Dict[str, List[int]]:
        """Get default expected coordination numbers for common elements.

        Returns
        -------
        dict[str, list[int]]
            Dictionary mapping element symbols to expected coordination numbers.
        """
        # Default expected coordination numbers based on common chemical knowledge
        # Format: {element_symbol: [list of common coordination numbers]}
        return {
            # Alkali metals
            "Li": [4, 6, 8],
            "Na": [6, 8],
            "K": [6, 8, 10, 12],
            "Rb": [8, 10, 12],
            "Cs": [8, 10, 12],
            # Alkaline earth metals
            "Be": [3, 4],
            "Mg": [4, 6, 8],
            "Ca": [6, 7, 8, 12],
            "Sr": [6, 7, 8, 10, 12],
            "Ba": [8, 10, 12],
            # Transition metals
            "Sc": [6],
            "Ti": [4, 6, 8],
            "V": [4, 5, 6, 8],
            "Cr": [4, 6],
            "Mn": [4, 5, 6, 8],
            "Fe": [4, 5, 6],
            "Co": [4, 5, 6],
            "Ni": [4, 5, 6],
            "Cu": [2, 4, 6],
            "Zn": [4, 6],
            "Y": [6, 8],
            "Zr": [6, 7, 8],
            "Nb": [6, 8],
            "Mo": [4, 6, 8],
            "Tc": [6],
            "Ru": [6],
            "Rh": [4, 6],
            "Pd": [4, 6],
            "Ag": [2, 4, 6],
            "Cd": [4, 6, 8],
            "Hf": [6, 7, 8],
            "Ta": [6, 8],
            "W": [4, 6, 8],
            "Re": [6],
            "Os": [6, 8],
            "Ir": [6],
            "Pt": [4, 6],
            "Au": [2, 4],
            "Hg": [2, 4, 6, 8],
            # Main group elements
            "B": [3, 4],
            "C": [3, 4],
            "N": [3, 4],
            "O": [1, 2, 3, 4],
            "F": [1, 2],
            "Al": [4, 6],
            "Si": [4, 6],
            "P": [3, 4, 5, 6],
            "S": [2, 4, 6],
            "Cl": [1, 6],
            "Ga": [4, 6],
            "Ge": [4, 6],
            "As": [3, 5, 6],
            "Se": [2, 4, 6],
            "Br": [1, 5],
            "In": [4, 6, 8],
            "Sn": [4, 6, 8],
            "Sb": [3, 5, 6],
            "Te": [2, 4, 6],
            "I": [1, 5, 7],
            "Tl": [3, 6],
            "Pb": [4, 6, 8],
            "Bi": [3, 5, 6, 8],
            # Noble gases (usually 0, but can form compounds)
            "He": [0, 2],
            "Ne": [0],
            "Ar": [0, 2],
            "Kr": [0, 2],
            "Xe": [0, 2, 4, 6, 8],
            "Rn": [0, 2, 8],
            # Lanthanides and actinides (typically have high and variable coordination numbers)
            "La": [6, 7, 8, 9, 10, 12],
            "Ce": [6, 7, 8, 9, 10, 12],
            "Pr": [6, 7, 8, 9],
            "Nd": [6, 8, 9, 12],
            "Pm": [6, 8, 9],
            "Sm": [6, 7, 8, 9, 12],
            "Eu": [6, 7, 8, 9, 10],
            "Gd": [6, 7, 8, 9],
            "Tb": [6, 7, 8, 9],
            "Dy": [6, 7, 8, 9],
            "Ho": [6, 8, 9],
            "Er": [6, 8, 9],
            "Tm": [6, 8, 9],
            "Yb": [6, 8],
            "Lu": [6, 8, 9],
            "Ac": [6, 8, 9, 12],
            "Th": [8, 9, 10, 12],
            "Pa": [6, 8, 9],
            "U": [6, 7, 8, 9, 12],
            "Np": [6, 8, 9],
            "Pu": [6, 8, 9],
            "Am": [6, 8, 9, 10, 12],
        }

    def _get_compute_attributes(self) -> dict[str, Any]:
        """Get the attributes for the compute_structure method."""
        return {
            "environment_database": self.environment_database,
            "nn_method": self.config.nn_method,
            "tolerance": self.config.tolerance,
        }

    @staticmethod
    def compute_structure(
        structure: Structure,
        environment_database: Dict[str, List[int]],
        nn_method: str,
        tolerance: float,
    ) -> float:
        """Evaluate if coordination environments match chemical expectations.

        Parameters
        ----------
        structure : Structure
            A pymatgen Structure object to evaluate.
        environment_database : dict[str, list[int]]
            Expected coordination numbers for each element.
        nn_method : str
            Method to determine nearest neighbors.
        tolerance : float
            Tolerance for coordination number matching.

        Returns
        -------
        float
            Score between 0.0 and 1.0 indicating how well the structure's
            coordination environments match chemical expectations.
        """
        # Initialize nearest neighbor finder
        if nn_method.lower() == "crystalnn":
            nn_finder = CrystalNN()
        elif nn_method.lower() == "voronoinn":
            nn_finder = VoronoiNN()
        else:
            raise ValueError(f"Unknown nearest neighbor method: {nn_method}")

        # Get coordination numbers for all sites
        coordination_numbers = []
        elements = []

        for i, site in enumerate(structure):
            element = str(site.specie.symbol)
            elements.append(element)

            try:
                # Get CN for the site
                cn = nn_finder.get_cn(structure, i)
                coordination_numbers.append(cn)
            except Exception as e:
                logger.warning(
                    f"Could not determine CN for site {i} ({element}): {str(e)}"
                )
                coordination_numbers.append(np.nan)

        # Check if we have expected CN data for all elements in the structure
        elements_in_structure = set(elements)
        elements_with_data = set(environment_database.keys())
        missing_elements = elements_in_structure - elements_with_data

        if missing_elements:
            logger.warning(
                f"Missing coordination data for elements: {missing_elements}"
            )
            # For elements without data, we'll assume their CN is valid

        # For each site, check if the coordination number matches expectations
        valid_sites = 0
        total_sites_checked = 0

        for i, (element, cn) in enumerate(zip(elements, coordination_numbers)):
            if np.isnan(cn) or element in missing_elements:
                continue

            expected_cns = environment_database.get(element, [])
            if not expected_cns:
                continue

            total_sites_checked += 1

            # Check if the CN is close to any expected value
            if any(abs(cn - expected_cn) <= tolerance for expected_cn in expected_cns):
                valid_sites += 1
            else:
                logger.debug(
                    f"Site {i} ({element}) has CN {cn}, which doesn't match "
                    f"expected values {expected_cns}"
                )

        # Return the ratio of valid sites
        if valid_sites / total_sites_checked == 1.0:
            return 1.0
        else:
            return 0.0

    def aggregate_results(self, values: list[float]) -> Dict[str, Any]:
        """Aggregate results into final metric values.

        Parameters
        ----------
        values : list[float]
            Coordination environment scores for each structure.

        Returns
        -------
        dict
            Dictionary with aggregated metrics.
        """
        # Filter out NaN values
        valid_values = [v for v in values if not np.isnan(v)]

        if not valid_values:
            return {
                "metrics": {"coordination_score": 0.0, "valid_structures_ratio": 0.0},
                "primary_metric": "coordination_score",
                "uncertainties": {},
            }

        # Mean score across all structures
        mean_score = np.mean(valid_values)

        # Count structures with good coordination (>80% match)
        good_structures = sum(1 for v in valid_values if v >= 0.8)
        good_structures_ratio = good_structures / len(valid_values)

        return {
            "metrics": {
                "coordination_score": mean_score,
                "valid_structures_ratio": good_structures_ratio,
            },
            "primary_metric": "coordination_score",
            "uncertainties": {
                "coordination_score": {
                    "std": np.std(valid_values) if len(valid_values) > 1 else 0.0
                }
            },
        }


@dataclass
class PhysicalPlausibilityConfig(MetricConfig):
    """Configuration for the PhysicalPlausibility metric.

    Parameters
    ----------
    min_density : float, default=1.0
        Minimum plausible density in g/cm³.
    max_density : float, default=25.0
        Maximum plausible density in g/cm³.
    check_format : bool, default=True
        Whether to check if the structure can be properly formatted.
    check_symmetry : bool, default=True
        Whether to check if the structure has valid symmetry.
    """

    min_density: float = 1.0
    max_density: float = 25.0
    check_format: bool = True
    check_symmetry: bool = True


class PhysicalPlausibilityMetric(BaseMetric):
    """Evaluate basic physical plausibility of structures.

    This metric checks for reasonable density and crystallographic format.

    Parameters
    ----------
    min_density : float, default=1.0
        Minimum plausible density in g/cm³.
    max_density : float, default=25.0
        Maximum plausible density in g/cm³.
    check_format : bool, default=True
        Whether to check if the structure can be properly formatted.
    check_symmetry : bool, default=True
        Whether to check if the structure has valid symmetry.
    name : str, optional
        Custom name for the metric.
    description : str, optional
        Description of what the metric measures.
    lower_is_better : bool, default=False
        Higher values indicate more physically plausible structures.
    n_jobs : int, default=1
        Number of parallel jobs to run.
    """

    def __init__(
        self,
        min_density: float = 1.0,
        max_density: float = 25.0,
        check_format: bool = True,
        check_symmetry: bool = True,
        name: str | None = None,
        description: str | None = None,
        lower_is_better: bool = False,
        n_jobs: int = 1,
    ):
        super().__init__(
            name=name or "PhysicalPlausibility",
            description=description
            or "Evaluates basic physical plausibility of structures",
            lower_is_better=lower_is_better,
            n_jobs=n_jobs,
        )
        self.config = PhysicalPlausibilityConfig(
            name=self.config.name,
            description=self.config.description,
            lower_is_better=self.config.lower_is_better,
            n_jobs=self.config.n_jobs,
            min_density=min_density,
            max_density=max_density,
            check_format=check_format,
            check_symmetry=check_symmetry,
        )

    def _get_compute_attributes(self) -> dict[str, Any]:
        """Get the attributes for the compute_structure method."""
        return {
            "min_density": self.config.min_density,
            "max_density": self.config.max_density,
            "check_format": self.config.check_format,
            "check_symmetry": self.config.check_symmetry,
        }

    @staticmethod
    def compute_structure(
        structure: Structure,
        min_density: float,
        max_density: float,
        check_format: bool,
        check_symmetry: bool,
    ) -> float:
        """Evaluate physical plausibility of a structure.

        Parameters
        ----------
        structure : Structure
            A pymatgen Structure object to evaluate.
        min_density : float
            Minimum plausible density in g/cm³.
        max_density : float
            Maximum plausible density in g/cm³.
        check_format : bool
            Whether to check if the structure can be properly formatted.
        check_symmetry : bool
            Whether to check if the structure has valid symmetry.

        Returns
        -------
        float
            Score between 0.0 and 1.0 indicating physical plausibility.
        """
        checks_passed = 0
        total_checks = 0

        # Check 1: Density within reasonable range
        total_checks += 1
        try:
            density = structure.density
            density_valid = min_density <= density <= max_density
            if density_valid:
                checks_passed += 1
            else:
                logger.debug(
                    f"Density check failed: {density:.3f} g/cm³ "
                    f"(not in range [{min_density}, {max_density}])"
                )
        except Exception as e:
            logger.debug(f"Could not compute density: {str(e)}")

        # Check 2: Valid lattice (not collapsed, not excessively large)
        total_checks += 1
        try:
            # Check if lattice is valid (not collapsed)
            lattice = structure.lattice
            volume = lattice.volume

            # Check if lattice parameters are reasonable
            a, b, c = lattice.abc
            angles = lattice.angles

            # Check for collapsed lattice or unreasonably large lattice
            if (
                volume > 0.1
                and a > 0.1
                and b > 0.1
                and c > 0.1
                and a < 100
                and b < 100
                and c < 100
                and all(0 < angle < 180 for angle in angles)
            ):
                checks_passed += 1
            else:
                logger.debug(
                    f"Lattice check failed: a={a:.3f}, b={b:.3f}, c={c:.3f}, "
                    f"angles={angles}, volume={volume:.3f}"
                )
        except Exception as e:
            logger.debug(f"Could not validate lattice: {str(e)}")

        # Check 3: Format representation check
        if check_format:
            total_checks += 1
            # Try to convert to CIF format and back

            # Write to CIF
            cif_writer = CifWriter(structure)
            cif_string = "temp.cif"
            cif_writer.write_file(cif_string)

            # Parse back from CIF
            parser = CifParser(cif_string)
            recovered_structure = parser.get_structures()[0]

            # Check if recovered structure is similar to original
            # by comparing composition and number of sites

            # IMPORTANT NOTE - CIF files will sometimes load primitive cells and sometimes conventional cells.
            # This is assuming the initial file is a conventional cell. If this IS NOT THE CASE, amend the input structure
            # to be a conventional cell.

            if (
                structure.composition.reduced_formula
                == recovered_structure.composition.reduced_formula
                and len(structure) == len(recovered_structure.to_conventional())
            ):
                checks_passed += 1
            else:
                logger.debug(
                    f"Format check failed: original={structure.composition}, "
                    f"recovered={recovered_structure.composition}"
                )

        # Check 4: Symmetry check
        if check_symmetry:
            total_checks += 1
            try:
                from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

                # Check if the structure has valid symmetry
                symm_analyzer = SpacegroupAnalyzer(structure)
                spacegroup = symm_analyzer.get_space_group_number()

                # Any valid spacegroup number (1-230) is acceptable
                if 0 < spacegroup <= 230:
                    checks_passed += 1
                else:
                    logger.debug(f"Symmetry check failed: spacegroup={spacegroup}")
            except Exception as e:
                print("Symmetry failed")
                logger.debug(f"Symmetry check failed: {str(e)}")

        # Return the ratio of passed checks
        if checks_passed / total_checks == 1.0:
            return 1.0
        else:
            return 0.0

    def aggregate_results(self, values: list[float]) -> Dict[str, Any]:
        """Aggregate results into final metric values.

        Parameters
        ----------
        values : list[float]
            Physical plausibility scores for each structure.

        Returns
        -------
        dict
            Dictionary with aggregated metrics.
        """
        # Filter out NaN values
        valid_values = [v for v in values if not np.isnan(v)]

        if not valid_values:
            return {
                "metrics": {
                    "physical_plausibility_score": 0.0,
                    "fully_plausible_ratio": 0.0,
                },
                "primary_metric": "physical_plausibility_score",
                "uncertainties": {},
            }

        # Mean score across all structures
        mean_score = np.mean(valid_values)

        # Count structures that pass all checks
        fully_plausible = sum(1 for v in valid_values if v >= 0.999)
        fully_plausible_ratio = fully_plausible / len(valid_values)

        return {
            "metrics": {
                "physical_plausibility_score": mean_score,
                "fully_plausible_ratio": fully_plausible_ratio,
            },
            "primary_metric": "physical_plausibility_score",
            "uncertainties": {
                "physical_plausibility_score": {
                    "std": np.std(valid_values) if len(valid_values) > 1 else 0.0
                }
            },
        }


# Composite validity metric that combines all fundamental validity metrics
@dataclass
class CompositeValidityConfig(MetricConfig):
    """Configuration for the CompositeValidity metric.

    Parameters
    ----------
    metrics : dict[str, MetricConfig]
        Dictionary of metric configurations to include in the composite.
    weights : dict[str, float] | None, default=None
        Weights for each metric.
    threshold : float, default=0.7
        Threshold for considering a structure valid.
    """

    # Use field with default_factory for required dictionary fields
    # Note: We're storing MetricConfig objects, not BaseMetric objects
    metrics: Dict[str, MetricConfig] = field(default_factory=dict)
    weights: Dict[str, float] | None = None
    threshold: float = 0.7


class CompositeValidityMetric(BaseMetric):
    """Combine multiple validity metrics into a single score.

    Parameters
    ----------
    metrics : dict[str, BaseMetric] | None, default=None
        Dictionary of metrics to include in the composite.
        If None, all fundamental validity metrics are included.
    weights : dict[str, float] | None, default=None
        Weights for each metric.
    threshold : float, default=0.7
        Threshold for considering a structure valid.
    name : str, optional
        Custom name for the metric.
    description : str, optional
        Description of what the metric measures.
    lower_is_better : bool, default=False
        Higher values indicate more valid structures.
    n_jobs : int, default=1
        Number of parallel jobs to run.
    """

    def __init__(
        self,
        metrics: Dict[str, BaseMetric] | None = None,
        weights: Dict[str, float] | None = None,
        threshold: float = 0.7,
        name: str | None = None,
        description: str | None = None,
        lower_is_better: bool = False,
        n_jobs: int = 1,
    ):
        super().__init__(
            name=name or "CompositeValidity",
            description=description or "Combined measure of structure validity",
            lower_is_better=lower_is_better,
            n_jobs=n_jobs,
        )

        # If no metrics provided, use all fundamental validity metrics
        if metrics is None:
            metrics = {
                "charge_neutrality": ChargeNeutralityMetric(),
                "min_distance": MinimumInteratomicDistanceMetric(),
                "coordination": CoordinationEnvironmentMetric(),
                "physical_plausibility": PhysicalPlausibilityMetric(),
            }

        # If no weights provided, use equal weights
        if weights is None:
            weights = {name: 1.0 / len(metrics) for name in metrics}
        else:
            # Ensure all metrics have weights
            missing = set(metrics.keys()) - set(weights.keys())
            if missing:
                for name in missing:
                    weights[name] = 1.0
                weights = {
                    name: w / sum(weights.values()) for name, w in weights.items()
                }

        self.metrics = metrics

        # Create a config with metrics configs (not the metrics themselves)
        metrics_configs = {name: metric.config for name, metric in metrics.items()}

        self.config = CompositeValidityConfig(
            name=self.config.name,
            description=self.config.description,
            lower_is_better=self.config.lower_is_better,
            n_jobs=self.config.n_jobs,
            metrics=metrics_configs,  # Store the configs, not the metric objects
            weights=weights,
            threshold=threshold,
        )

    @staticmethod
    def compute_structure(structure: Structure, **compute_args) -> float:
        """Compute the composite validity score for a structure.

        This method is not actually used for the composite metric,
        as we compute scores for individual metrics separately.
        Required to implement the abstract method.

        Parameters
        ----------
        structure : Structure
            A pymatgen Structure object to evaluate.

        Returns
        -------
        float
            Always returns 0.0, not used.
        """
        return 0.0

    def compute(self, structures: list[Structure]) -> MetricResult:
        """Compute the composite validity score for a batch of structures.

        Parameters
        ----------
        structures : list[Structure]
            Structures to evaluate.

        Returns
        -------
        MetricResult
            Result of the composite validity metric.
        """
        start_time = time.time()

        # Compute individual metric results
        metric_results = {}
        for name, metric in self.metrics.items():
            try:
                result = metric.compute(structures)
                metric_results[name] = result
            except Exception as e:
                logger.error(f"Failed to compute metric {name}", exc_info=True)
                # Create a placeholder result for failed metrics
                metric_results[name] = MetricResult(
                    metrics={name: float("nan")},
                    primary_metric=name,
                    uncertainties={},
                    config=metric.config,
                    computation_time=0.0,
                    n_structures=len(structures),
                    individual_values=[float("nan")] * len(structures),
                    failed_indices=list(range(len(structures))),
                    warnings=[f"Failed to compute: {str(e)}"],
                )

        # Compute the overall validity score for each structure
        individual_scores = []

        for i in range(len(structures)):
            # Collect scores for this structure from all metrics
            structure_scores = {}

            for name, result in metric_results.items():
                # Get individual value for this structure if available
                if i < len(result.individual_values) and not np.isnan(
                    result.individual_values[i]
                ):
                    structure_scores[name] = result.individual_values[i]

            # Compute weighted average if we have scores
            if structure_scores:
                weights = {name: self.config.weights[name] for name in structure_scores}
                total_weight = sum(weights.values())
                weighted_sum = sum(
                    score * weights[name] / total_weight
                    for name, score in structure_scores.items()
                )
                individual_scores.append(weighted_sum)
            else:
                individual_scores.append(0.0)

        # Aggregate results
        result_dict = self.aggregate_results(individual_scores)

        return MetricResult(
            metrics=result_dict["metrics"],
            primary_metric=result_dict["primary_metric"],
            uncertainties=result_dict["uncertainties"],
            config=self.config,
            computation_time=time.time() - start_time,
            n_structures=len(structures),
            individual_values=individual_scores,
            failed_indices=[],
            warnings=[],
        )

    def aggregate_results(self, values: list[float]) -> Dict[str, Any]:
        """Aggregate results into final metric values.

        Parameters
        ----------
        values : list[float]
            Composite validity scores for each structure.

        Returns
        -------
        dict
            Dictionary with aggregated metrics.
        """
        # Filter out NaN values
        valid_values = [v for v in values if not np.isnan(v)]

        if not valid_values:
            return {
                "metrics": {"validity_score": 0.0, "valid_structures_ratio": 0.0},
                "primary_metric": "validity_score",
                "uncertainties": {},
            }

        # Mean score across all structures
        mean_score = np.mean(valid_values)

        # Count structures above threshold
        valid_structures = sum(1 for v in valid_values if v >= self.config.threshold)
        valid_structures_ratio = valid_structures / len(valid_values)

        return {
            "metrics": {
                "validity_score": mean_score,
                "valid_structures_ratio": valid_structures_ratio,
            },
            "primary_metric": "validity_score",
            "uncertainties": {
                "validity_score": {
                    "std": np.std(valid_values) if len(valid_values) > 1 else 0.0
                }
            },
        }
