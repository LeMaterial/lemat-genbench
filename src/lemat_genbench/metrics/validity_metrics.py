"""Fixed validity metrics for material structures.

This module implements validity metrics that report both ratios and deviations
for interpretable validity assessment, following the original logic.
"""

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
from pymatgen.analysis.bond_valence import BVAnalyzer, calculate_bv_sum
from pymatgen.analysis.local_env import get_neighbors_of_site_with_index
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from smact.metallicity import metallicity_score

from lemat_genbench.metrics.base import BaseMetric, MetricConfig
from lemat_genbench.utils.logging import logger
from lemat_genbench.utils.oxidation_state import (
    compositional_oxi_state_guesses,
    get_inequivalent_site_info,
)

# Suppress common warnings
warnings.filterwarnings("ignore", message="No oxidation states specified on sites!")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


@dataclass
class ChargeNeutralityConfig(MetricConfig):
    """Configuration for the ChargeNeutrality metric.

    Parameters
    ----------
    tolerance : float, default=0.1
        Maximum allowed absolute charge deviation to consider a structure charge neutral.
    strict : bool, default=False
        If True, use strict bond valence analysis. If False, use compositional guessing.
    """

    tolerance: float = 0.1
    strict: bool = False


class ChargeNeutralityMetric(BaseMetric):
    """Metric to evaluate charge neutrality of crystal structures.

    Uses the original three-step approach:
    1. Bond valence analysis for metallic structures
    2. Pymatgen oxidation state determination
    3. Compositional oxidation state guessing

    Returns binary validity based on tolerance and tracks charge deviations.
    """

    def __init__(
        self,
        tolerance: float = 0.1,
        strict: bool = False,
        name: str = "ChargeNeutrality",
        description: str = "Evaluates charge neutrality of structures",
        n_jobs: int = 1,
    ):
        # Create the custom config and call super().__init__
        super().__init__(
            name=name,
            description=description,
            lower_is_better=False,  # Higher ratio = better
            n_jobs=n_jobs,
        )

        # Override with custom config
        self.config = ChargeNeutralityConfig(
            tolerance=tolerance,
            strict=strict,
            name=name,
            description=description,
            lower_is_better=False,
            n_jobs=n_jobs,
        )
        self.bv_analyzer = BVAnalyzer()

    def _get_compute_attributes(self) -> dict[str, Any]:
        return {
            "tolerance": self.config.tolerance,
            "strict": self.config.strict,
            "bv_analyzer": self.bv_analyzer,
        }

    @staticmethod
    def compute_structure(structure: Structure, **compute_args: Any) -> float:
        """Compute charge deviation from neutrality for a single structure.

        Returns the absolute deviation from charge neutrality.
        This will be used to determine binary validity in aggregation.

        Returns
        -------
        float
            Absolute deviation from charge neutrality (0.0 = perfectly neutral)
        """
        _ = compute_args.get("tolerance", 0.1)
        _ = compute_args.get("strict", False)
        bv_analyzer = compute_args.get("bv_analyzer")

        # Step 1: Bond valence analysis for metallic structures
        try:
            metal_score = metallicity_score(Composition(structure.formula))
            if metal_score > 0.7:
                return 0.0  # Perfectly charge neutral (metallic)
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

            # Check if all bond valence sums are ~0 (metallic structure)
            try:
                for bv in bvs:
                    if np.abs(bv[1]) < 10**-15:
                        pass
                    else:   
                        raise ValueError
                logger.debug(
                    "Valid structure - Metallic structure with bond valence equal to zero for all atoms"
                )
                return 0.0  # Perfectly charge neutral (metallic)
            except ValueError:
                logger.debug(
                    "Bond valence sum calculation yielded non-zero values - not metallic structure"
                )
        except Exception as e:
            logger.debug(f"Bond valence analysis failed: {str(e)}")

        # Step 2: Pymatgen oxidation state determination
        try:
            structure_with_oxi = bv_analyzer.get_oxi_state_decorated_structure(
                structure
            )
            charge_sum = sum(site.specie.oxi_state for site in structure_with_oxi.sites)
            logger.debug(
                "Valid structure - charge balanced based on Pymatgen's get_oxi_state_decorated_structure function"
            )
            return abs(charge_sum)  # Return actual absolute deviation
        except ValueError:
            logger.debug(
                "Could not determine oxidation states using get_oxi_state_decorated_structure"
            )

        # Step 3: Compositional oxidation state guessing
        # try:
        
        comp = Composition(structure.composition)
        # Create data directory if it doesn't exist
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)

        with open(data_dir / "lemat_icsd_oxi_state_mapping.json", "r") as f:
            oxi_state_mapping = json.load(f)

        oxi_states_override = {}
        for e in comp.elements:
            if str(e) in oxi_state_mapping:
                oxi_states_override[str(e)] = oxi_state_mapping[str(e)]
        # print("starting oxi state guesses")
        output = compositional_oxi_state_guesses(
            comp,
            all_oxi_states=False,
            max_sites=-1,
            target_charge=0,
            oxi_states_override=oxi_states_override,
        )
        logger.debug(
            f"Most valid oxidation state and score based on composition: "
            f"{output[1][0] if len(output[1]) > 0 else 'None'}, "
            f"{output[2][0] if len(output[2]) > 0 else 'None'}"
        )
        try:
            score = output[2][0]
            if score > 0.001:
                return 0.0  # Assume charge neutral (reasonable composition)
            else:
                return 0.0  # Small deviation penalty (charge balanced using LeMatBulk oxidation states, but requires unusual states)
        except IndexError:

            output = compositional_oxi_state_guesses(
                comp,
                all_oxi_states=True,
                max_sites=-1,
                target_charge=0,
                oxi_states_override=None,
                )

            try:
                score = -output[2][0] # correlation between oxidation state and electronegativity. Should be negative correlation for valid structures, reverse sign so logic 
                # maximizing score is consistent
                if score > 0.0:
                    return 0.0  # Assume charge neutral (reasonable composition)
                else:
                    return 10.0  # correlation between oxidation state and electronegativity is positive (scores is negative) not a reasonable composition
            
            except IndexError:
                return 10.0 # large deviation penalty 

    def aggregate_results(self, values: list[float]) -> Dict[str, Any]:
        """Aggregate results into final metric values.

        Parameters
        ----------
        values : list[float]
            Charge deviation values for each structure.

        Returns
        -------
        dict
            Dictionary with aggregated metrics.
        """
        # Filter out NaN values
        valid_values = [v for v in values if not np.isnan(v)]
        total_count = len(values)  # Use original count as denominator

        if not valid_values:
            return {
                "metrics": {
                    "charge_neutral_ratio": 0.0,
                    "charge_neutral_count": 0,
                    "avg_charge_deviation": float("nan"),
                    "total_structures": total_count,
                },
                "primary_metric": "charge_neutral_ratio",
                "uncertainties": {},
            }

        # Count charge neutral structures (within tolerance)
        charge_neutral_count = sum(
            1 for v in valid_values if v <= self.config.tolerance
        )
        charge_neutral_ratio = charge_neutral_count / total_count

        # Calculate average deviation
        avg_deviation = np.mean(valid_values)

        return {
            "metrics": {
                "charge_neutral_ratio": charge_neutral_ratio,
                "charge_neutral_count": int(charge_neutral_count),
                "avg_charge_deviation": avg_deviation,
                "total_structures": total_count,
            },
            "primary_metric": "charge_neutral_ratio",
            "uncertainties": {
                "charge_deviation_std": {
                    "std": np.std(valid_values) if len(valid_values) > 1 else 0.0
                }
            },
        }


@dataclass
class MinimumInteratomicDistanceConfig(MetricConfig):
    """Configuration for the MinimumInteratomicDistance metric.

    Parameters
    ----------
    scaling_factor : float, default=0.5
        Factor to scale the minimum distance (sum of atomic radii).
    """

    scaling_factor: float = 0.5


class MinimumInteratomicDistanceMetric(BaseMetric):
    """Metric to evaluate minimum interatomic distances in crystal structures."""

    def __init__(
        self,
        scaling_factor: float = 0.5,
        name: str = "MinimumInteratomicDistance",
        description: str = "Evaluates minimum distances between atoms",
        n_jobs: int = 1,
    ):
        # Create the custom config and call super().__init__
        super().__init__(
            name=name,
            description=description,
            lower_is_better=False,  # Higher ratio = better
            n_jobs=n_jobs,
        )

        # Override with custom config
        self.config = MinimumInteratomicDistanceConfig(
            scaling_factor=scaling_factor,
            name=name,
            description=description,
            lower_is_better=False,
            n_jobs=n_jobs,
        )

        # Initialize default element radii
        self.element_radii = {
            str(el): el.atomic_radius or 1.0  # Default to 1.0 if None
            for el in Element
            if hasattr(el, "atomic_radius")
        }

    def _get_compute_attributes(self) -> dict[str, Any]:
        return {
            "scaling_factor": self.config.scaling_factor,
            "element_radii": self.element_radii,
        }

    @staticmethod
    def compute_structure(structure: Structure, **compute_args: Any) -> float:
        """Compute minimum interatomic distance validity for a single structure.

        Returns
        -------
        float
            1.0 if all interatomic distances are valid, 0.0 otherwise.
        """
        scaling_factor = compute_args.get("scaling_factor", 0.5)
        element_radii = compute_args.get("element_radii", {})

        # Get all pairs of sites
        all_distances = structure.distance_matrix
        n_sites = len(structure)

        # No distances to check (single atom structure)
        if n_sites <= 1:
            return 1.0  # Single atom is always valid

        # Check if we have radius data for all elements in the structure
        elements_in_structure = {str(site.specie) for site in structure}
        missing_elements = elements_in_structure - set(element_radii.keys())

        if missing_elements:
            logger.debug(f"Missing radius data for elements: {missing_elements}")
            # Use default radius of 1.0 for missing elements
            default_radii = {el: 1.0 for el in missing_elements}
            element_radii = {**element_radii, **default_radii}

        # For each pair of sites, compute the minimum allowed distance
        for i in range(n_sites):
            for j in range(i + 1, n_sites):
                element_i = str(structure[i].specie)
                element_j = str(structure[j].specie)

                # Sum of atomic radii with scaling
                min_dist = (
                    0.7 + element_radii[element_i] + element_radii[element_j]
                ) * scaling_factor
                actual_dist = all_distances[i, j]

                if actual_dist < min_dist:
                    logger.debug(
                        f"Distance between {element_i} and {element_j} is {actual_dist:.3f} Å, "
                        f"which is less than the minimum allowed {min_dist:.3f} Å"
                    )
                    return 0.0  # Invalid distance found

        return 1.0  # All distances are valid

    def aggregate_results(self, values: list[float]) -> Dict[str, Any]:
        """Aggregate results into final metric values."""
        # Filter out NaN values
        valid_values = [v for v in values if not np.isnan(v)]
        total_count = len(values)  # Use original count as denominator

        if not valid_values:
            return {
                "metrics": {
                    "distance_valid_ratio": 0.0,
                    "distance_valid_count": 0,
                    "total_structures": total_count,
                },
                "primary_metric": "distance_valid_ratio",
                "uncertainties": {},
            }

        # Count valid structures
        valid_count = sum(valid_values)
        valid_ratio = valid_count / total_count

        return {
            "metrics": {
                "distance_valid_ratio": valid_ratio,
                "distance_valid_count": int(valid_count),
                "total_structures": total_count,
            },
            "primary_metric": "distance_valid_ratio",
            "uncertainties": {},
        }


@dataclass
class PhysicalPlausibilityConfig(MetricConfig):
    """Configuration for the PhysicalPlausibility metric.

    Parameters
    ----------
    min_atomic_density : float, default=0.01
        Minimum allowed density in atoms/A³.
    max_atomic_density : float, default=25.0
        Maximum allowed density in atoms/A³.
    min_mass_density : float, default=0.01
        Minimum allowed density in g/cm³.
    max_mass_density : float, default=25.0
        Maximum allowed density in g/cm³.
    check_format : bool, default=True
        Whether to check CIF format round-trip validity.
    check_symmetry : bool, default=True
        Whether to check space group validity.
    """
    min_atomic_density: float = 0.00001
    max_atomic_density: float = 0.5
    min_mass_density: float = 0.01
    max_mass_density: float = 25.0
    check_format: bool = True
    check_symmetry: bool = True


class PhysicalPlausibilityMetric(BaseMetric):
    """Metric to evaluate physical plausibility of crystal structures."""

    def __init__(
        self,
        min_atomic_density: float = 0.00001,
        max_atomic_density: float = 0.5,
        min_mass_density: float = 0.01,
        max_mass_density: float = 25.0,
        check_format: bool = True,
        check_symmetry: bool = True,
        name: str = "PhysicalPlausibility",
        description: str = "Evaluates physical plausibility of structures",
        n_jobs: int = 1,
    ):
        # Create the custom config and call super().__init__
        super().__init__(
            name=name,
            description=description,
            lower_is_better=False,  # Higher ratio = better
            n_jobs=n_jobs,
        )

        # Override with custom config
        self.config = PhysicalPlausibilityConfig(
            min_atomic_density=min_atomic_density,
            max_atomic_density=max_atomic_density,
            min_mass_density=min_mass_density,
            max_mass_density=max_mass_density,
            check_format=check_format,
            check_symmetry=check_symmetry,
            name=name,
            description=description,
            lower_is_better=False,
            n_jobs=n_jobs,
        )

    def _get_compute_attributes(self) -> dict[str, Any]:
        return {
            "min_atomic_density": self.config.min_atomic_density,
            "max_atomic_density": self.config.max_atomic_density,
            "min_mass_density": self.config.min_mass_density,
            "max_mass_density": self.config.max_mass_density,
            "check_format": self.config.check_format,
            "check_symmetry": self.config.check_symmetry,
        }

    @staticmethod
    def compute_structure(structure: Structure, **compute_args: Any) -> float:
        """Compute physical plausibility for a single structure.

        Returns
        -------
        float
            1.0 if structure passes all plausibility checks, 0.0 otherwise.
        """
        min_atomic_density = compute_args.get("min_atomic_density", 0.00001)
        max_atomic_density = compute_args.get("max_atomic_density", 0.5)
        min_mass_density = compute_args.get("min_mass_density", 0.01)
        max_mass_density = compute_args.get("max_mass_density", 25.0)
        check_format = compute_args.get("check_format", True)
        check_symmetry = compute_args.get("check_symmetry", True)

        checks_passed = 0
        total_checks = 3  # atomic density, mass density, and lattice checks are always done

        # 1. Mass density check
        try:
            mass_density = structure.density
            if min_mass_density <= mass_density <= max_mass_density:
                checks_passed += 1
            else:
                logger.debug(
                    f"Density check failed: {mass_density:.3f} g/cm³ "
                    f"(not in range [{min_mass_density}, {max_mass_density}])"
                )
        except Exception as e:
            logger.debug(f"Could not compute density: {str(e)}")

        # 2. Atomic density check
        try:
            volume = structure.volume
            num_atoms = len(structure)
            atomic_density = num_atoms / volume  


            if min_atomic_density <= atomic_density <= max_atomic_density:
                checks_passed += 1

            else:
                logger.debug(
                    f"Atomic density check failed: {atomic_density:.5f} atoms/A³ "
                    f"(not in range [{min_atomic_density}, {max_atomic_density}])"
                )
        except Exception as e:
            logger.debug(f"Could not compute density: {str(e)}")

        # 3. Lattice check
        try:
            lattice = structure.lattice
            volume = lattice.volume
            a, b, c = lattice.abc
            alpha, beta, gamma = lattice.angles

            # Check reasonable ranges
            if (
                volume > 1.0  # Volume > 1 Å³
                and all(
                    1.0 <= param <= 100.0 for param in [a, b, c]
                )  # Parameters 1-100 Å
                and all(0 < angle < 180 for angle in [alpha, beta, gamma])
            ):  # Angles 0-180°
                checks_passed += 1
            else:
                logger.debug(
                    f"Lattice check failed: a={a:.3f}, b={b:.3f}, c={c:.3f}, "
                    f"angles={[alpha, beta, gamma]}, volume={volume:.3f}"
                )
        except Exception as e:
            logger.debug(f"Could not validate lattice: {str(e)}")

        # 4. Format check (optional)
        if check_format:
            total_checks += 1
            try:
                # Test CIF round-trip
                cif_writer = CifWriter(structure)
                _ = str(cif_writer)
                # If we get here without exception, format is valid
                checks_passed += 1
            except Exception as e:
                logger.debug(f"Format check failed: {str(e)}")
        # 5. Symmetry check (optional)
        if check_symmetry:
            total_checks += 1
            try:
                sga = SpacegroupAnalyzer(structure)
                space_group_number = sga.get_space_group_number()
                if 1 <= space_group_number <= 230:
                    checks_passed += 1
                else:
                    logger.debug(
                        f"Symmetry check failed: spacegroup={space_group_number}"
                    )
            except Exception as e:
                logger.debug(f"Symmetry check failed: {str(e)}")

        logger.debug(
            f"Physical plausibility checks passed: {checks_passed}/{total_checks}"
        )

        # Return 1.0 if ALL checks passed, 0.0 otherwise

        return 1.0 if checks_passed == total_checks else 0.0

    def aggregate_results(self, values: list[float]) -> Dict[str, Any]:
        """Aggregate results into final metric values."""
        # Filter out NaN values
        valid_values = [v for v in values if not np.isnan(v)]
        total_count = len(values)  # Use original count as denominator

        if not valid_values:
            return {
                "metrics": {
                    "plausibility_valid_ratio": 0.0,
                    "plausibility_valid_count": 0,
                    "total_structures": total_count,
                },
                "primary_metric": "plausibility_valid_ratio",
                "uncertainties": {},
            }

        # Count valid structures
        valid_count = sum(valid_values)
        valid_ratio = valid_count / total_count

        return {
            "metrics": {
                "plausibility_valid_ratio": valid_ratio,
                "plausibility_valid_count": int(valid_count),
                "total_structures": total_count,
            },
            "primary_metric": "plausibility_valid_ratio",
            "uncertainties": {},
        }


class OverallValidityMetric(BaseMetric):
    """Metric that combines all validity checks into overall validity assessment.

    A structure is considered overall valid only if it passes ALL individual validity checks.
    """

    def __init__(
        self,
        charge_tolerance: float = 0.1,
        distance_scaling: float = 0.5,
        min_atomic_density: float = 0.00001,
        max_atomic_density: float = 0.5,
        min_mass_density: float = 0.01,
        max_mass_density: float = 25.0,
        check_format: bool = True,
        check_symmetry: bool = True,
        name: str = "OverallValidity",
        description: str = "Overall validity based on all individual checks",
        n_jobs: int = 1,
        verbose: bool = False,
    ):
        super().__init__(
            name=name,
            description=description,
            lower_is_better=False,  # Higher ratio = better
            n_jobs=n_jobs,
            verbose=verbose,
        )

        # Store parameters for individual metrics
        self.charge_tolerance = charge_tolerance
        self.distance_scaling = distance_scaling
        self.min_atomic_density = min_atomic_density
        self.max_atomic_density = max_atomic_density
        self.min_mass_density = min_mass_density
        self.max_mass_density = max_mass_density
        self.check_format = check_format
        self.check_symmetry = check_symmetry
        
        # Store diagnostic information during computation
        self._diagnostic_data = []

    def _get_compute_attributes(self) -> dict[str, Any]:
        attrs = super()._get_compute_attributes()
        return {
            **attrs,
            "charge_tolerance": self.charge_tolerance,
            "distance_scaling": self.distance_scaling,
            "min_atomic_density": self.min_atomic_density,
            "max_atomic_density": self.max_atomic_density,
            "min_mass_density": self.min_mass_density,
            "max_mass_density": self.max_mass_density,
            "check_format": self.check_format,
            "check_symmetry": self.check_symmetry,
            "verbose": self.verbose,
        }

    @staticmethod
    def compute_structure(structure: Structure, **compute_args: Any) -> float:
        """Compute overall validity for a single structure.

        Returns
        -------
        float
            1.0 if structure passes ALL validity checks, 0.0 otherwise.
        """

        # Extract parameters
        charge_tolerance = compute_args.get("charge_tolerance", 0.1)
        distance_scaling = compute_args.get("distance_scaling", 0.5)
        min_atomic_density = compute_args.get("min_atomic_density", 0.00001)
        max_atomic_density = compute_args.get("max_atomic_density", 0.5)
        min_mass_density = compute_args.get("min_mass_density", 0.01)
        max_mass_density = compute_args.get("max_mass_density", 25.0)
        check_format = compute_args.get("check_format", True)
        check_symmetry = compute_args.get("check_symmetry", True)

        # Check charge neutrality
        try:
            charge_deviation = ChargeNeutralityMetric.compute_structure(
                structure,
                tolerance=charge_tolerance,
                strict=False,
                bv_analyzer=BVAnalyzer(),
            )
            charge_valid = charge_deviation <= charge_tolerance
        except Exception:
            charge_valid = False

        # Check interatomic distances
        try:
            distance_score = MinimumInteratomicDistanceMetric.compute_structure(
                structure,
                scaling_factor=distance_scaling,
                element_radii={
                    str(el): el.atomic_radius or 1.0
                    for el in Element
                    if hasattr(el, "atomic_radius")
                },
            )
            distance_valid = distance_score >= 0.999  # Essentially 1.0
        except Exception:
            distance_valid = False

        # Check physical plausibility
        try:
            plausibility_score = PhysicalPlausibilityMetric.compute_structure(
                structure,
                min_atomic_density=min_atomic_density,
                max_atomic_density=max_atomic_density,
                min_mass_density=min_mass_density,
                max_mass_density=max_mass_density,
                check_format=check_format,
                check_symmetry=check_symmetry,
            )
            plausibility_valid = plausibility_score >= 0.999  # Essentially 1.0
        except Exception:
            plausibility_valid = False

        # Overall valid only if ALL checks pass
        overall_valid = (
            1.0 if (charge_valid and distance_valid and plausibility_valid) else 0.0
        )
        
        # Store diagnostic info in structure properties for later use
        # This is a way to pass additional info without breaking the base class contract
        if hasattr(structure, 'properties'):
            structure.properties['_validity_diagnostics'] = {
                "overall_valid": overall_valid,
                "charge_valid": charge_valid,
                "distance_valid": distance_valid,
                "plausibility_valid": plausibility_valid,
            }

        return overall_valid

    def aggregate_results(self, values: list[float]) -> Dict[str, Any]:
        """Aggregate results into final metric values."""
        # Filter out NaN values
        valid_values = [v for v in values if not np.isnan(v)]
        total_count = len(values)  # Use original count as denominator

        if not valid_values:
            return {
                "metrics": {
                    "overall_valid_ratio": 0.0,
                    "overall_valid_count": 0,
                    "total_structures": total_count,
                },
                "primary_metric": "overall_valid_ratio",
                "uncertainties": {},
            }

        # Count overall valid structures
        valid_count = sum(valid_values)
        valid_ratio = valid_count / total_count

        result = {
            "metrics": {
                "overall_valid_ratio": valid_ratio,
                "overall_valid_count": int(valid_count),
                "total_structures": total_count,
            },
            "primary_metric": "overall_valid_ratio",
            "uncertainties": {},
        }
        
        # Note: Diagnostic information could be accessed from structure properties
        # if needed, but we keep the standard interface clean
        
        return result


def lematbulk_item_to_structure(item: dict):
    sites = item["species_at_sites"]
    coords = item["cartesian_site_positions"]
    cell = item["lattice_vectors"]

    structure = Structure(
        species=sites, coords=coords, lattice=cell, coords_are_cartesian=True
    )

    return structure

if __name__ == "__main__":
    from datasets import load_dataset
    from tqdm import tqdm

    dataset_name = "Lematerial/LeMat-Bulk"
    name = "compatible_pbe"
    split = "train"
    dataset = load_dataset(dataset_name, name=name, split=split, streaming=False)

    np.random.seed(32)
    indicies = np.random.randint(0, len(dataset), 50)

    metric = PhysicalPlausibilityMetric()
    args = metric._get_compute_attributes()

    structures = []
    for i in tqdm(range(len(indicies))):
        index = int(indicies[i])
        strut = lematbulk_item_to_structure(dataset[index])
        structures.append(strut)

        val = metric.compute_structure(structures[i], **args)
        print(val)