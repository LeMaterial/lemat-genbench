"""Relaxation metrics for evaluating material structures.

This module implements metrics for evaluating structure stability and metastability.
"""

from typing import Any

import numpy as np
from pymatgen.core import Structure

from lematerial_forgebench.metrics.base import BaseMetric
from lematerial_forgebench.utils.logging import logger


class FormationEnergyMetric(BaseMetric):
    """Evaluate formation energy of a structure using an MLIP."""

    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
        lower_is_better: bool = True,
        n_jobs: int = 1,
    ):
        super().__init__(
            name=name or "FormationEnergyMetric",
            description=description
            or "Evaluates structure formation energy using the specified MLIP",
            lower_is_better=lower_is_better,
            n_jobs=n_jobs,
        )

    def _get_compute_attributes(self) -> dict[str, Any]:
        """Get the attributes for the compute_structure method."""
        return {}

    @staticmethod
    def compute_structure(structure: Structure) -> float:
        """Evaluates structure formation energy using the specified MLIP

        Parameters
        ----------
        structure : Structure
            A pymatgen Structure object.

        Returns
        -------
        float
            The precomputed formation energy value, or NaN if not available.
        """
        try:
            # Extract formation_energy from structure properties
            formation_energy = structure.properties.get("formation_energy", None)
            print(str(structure.composition) + " Formation Energy :", formation_energy)

            if formation_energy is None:
                logger.warning(
                    "Structure missing formation_energy in properties, please compute it first using StabilityPreprocess"
                )
                return np.nan

            return float(formation_energy)

        except Exception as e:
            logger.error(f"Failed to extract formation_energy: {str(e)}")
            return np.nan

    def aggregate_results(self, values: list[float]) -> dict[str, Any]:
        """Aggregate results into final metric values.

        Parameters
        ----------
        values : list[float]
            List of formation_energy values for each structure.

        Returns
        -------
        dict
            Dictionary with aggregated metrics.
        """
        # Convert to numpy array for efficient operations
        values_array = np.array(values)

        # Filter out NaN values
        valid_mask = ~np.isnan(values_array)
        formation_energy_values = values_array[valid_mask]

        if len(formation_energy_values) > 0:
            formation_energy_std = np.std(formation_energy_values)
            mean_formation_energy = np.mean(formation_energy_values)
        else:
            formation_energy_std = 0.0
            mean_formation_energy = 0.0

        return {
            "metrics": {
                "mean_formation_energy": mean_formation_energy,
            },
            "primary_metric": "mean_formation_energy",
            "uncertainties": {
                "formation_energy_std": {"std": formation_energy_std},
            },
        }


class StabilityMetric(BaseMetric):
    """Evaluate structure metastability using precomputed e_above_hull values.

    This metric assumes that e_above_hull values have already been computed
    and stored in structure.properties['e_above_hull']. It calculates stability
    statistics on the unrelaxed structure without performing additional recomputation.
    """

    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
        lower_is_better: bool = False,
        n_jobs: int = 1,
    ):
        super().__init__(
            name=name or "StabilityMetric",
            description=description
            or "Evaluates structure stability from precomputed e_above_hull",
            lower_is_better=lower_is_better,
            n_jobs=n_jobs,
        )

    def _get_compute_attributes(self) -> dict[str, Any]:
        """Get the attributes for the compute_structure method."""
        return {}

    @staticmethod
    def compute_structure(structure: Structure) -> float:
        """Extract precomputed e_above_hull from structure properties.

        Parameters
        ----------
        structure : Structure
            A pymatgen Structure object with e_above_hull in properties.

        Returns
        -------
        float
            The precomputed e_above_hull value, or NaN if not available.
        """
        try:
            # Extract e_above_hull from structure properties

            e_above_hull = structure.properties.get("e_above_hull", None)

            if e_above_hull is None:
                logger.warning(
                    "Structure missing e_above_hull in properties, please compute it first using StabilityMetric"
                )
                return np.nan

            return float(e_above_hull)

        except Exception as e:
            logger.error(f"Failed to extract e_above_hull: {str(e)}")
            return np.nan

    def aggregate_results(self, values: list[float]) -> dict[str, Any]:
        """Aggregate results into final metric values.

        Parameters
        ----------
        values : list[float]
            List of e_above_hull values for each structure.

        Returns
        -------
        dict
            Dictionary with aggregated metrics.
        """
        # Convert to numpy array for efficient operations
        values_array = np.array(values)

        # Filter out NaN values
        valid_mask = ~np.isnan(values_array)
        e_above_hull_values = values_array[valid_mask]

        if len(e_above_hull_values) > 0:
            # Calculate ratio of stable structures (e_above_hull <= 0) using numpy
            stable_ratio = np.sum(e_above_hull_values <= 0) / len(values)

        else:
            stable_ratio = np.nan

        return {
            "metrics": {
                "stable_ratio": stable_ratio,
            },
            "primary_metric": "stable_ratio",
            "uncertainties": {},
        }


class E_HullMetric(BaseMetric):
    """Evaluate structure metastability using precomputed e_above_hull values.

    This metric assumes that e_above_hull values have already been computed
    and stored in structure.properties['e_above_hull']. It calculates stability
    statistics on the unrelaxed structure without performing additional recomputation.
    """

    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
        lower_is_better: bool = False,
        n_jobs: int = 1,
    ):
        super().__init__(
            name=name or "StabilityMetric",
            description=description
            or "Evaluates structure stability from precomputed e_above_hull",
            lower_is_better=lower_is_better,
            n_jobs=n_jobs,
        )

    def _get_compute_attributes(self) -> dict[str, Any]:
        """Get the attributes for the compute_structure method."""
        return {}

    @staticmethod
    def compute_structure(structure: Structure) -> float:
        """Extract precomputed e_above_hull from structure properties.

        Parameters
        ----------
        structure : Structure
            A pymatgen Structure object with e_above_hull in properties.

        Returns
        -------
        float
            The precomputed e_above_hull value, or NaN if not available.
        """
        try:
            # Extract e_above_hull from structure properties

            e_above_hull = structure.properties.get("e_above_hull", None)

            if e_above_hull is None:
                logger.warning(
                    "Structure missing e_above_hull in properties, please compute it first using StabilityMetric"
                )
                return np.nan

            return float(e_above_hull)

        except Exception as e:
            logger.error(f"Failed to extract e_above_hull: {str(e)}")
            return np.nan

    def aggregate_results(self, values: list[float]) -> dict[str, Any]:
        """Aggregate results into final metric values.

        Parameters
        ----------
        values : list[float]
            List of e_above_hull values for each structure.

        Returns
        -------
        dict
            Dictionary with aggregated metrics.
        """
        # Convert to numpy array for efficient operations
        values_array = np.array(values)

        # Filter out NaN values
        valid_mask = ~np.isnan(values_array)
        e_above_hull_values = values_array[valid_mask]

        if len(e_above_hull_values) > 0:
            # Calculate ratio of stable structures (e_above_hull <= 0) using numpy
            e_above_hull_std = np.std(e_above_hull_values)
            mean_e_above_hull = np.mean(e_above_hull_values)
        else:
            e_above_hull_std = np.nan
            mean_e_above_hull = np.nan

        return {
            "metrics": {
                "mean_e_above_hull": mean_e_above_hull,
            },
            "primary_metric": "mean_e_above_hull",
            "uncertainties": {
                "e_above_hull_std": {"std": e_above_hull_std},
            },
        }


class MetastabilityMetric(BaseMetric):
    """Evaluate structure metastability using precomputed e_above_hull values.

    This metric assumes that e_above_hull values have already been computed
    and stored in structure.properties['e_above_hull']. It calculates stability
    statistics on the unrelaxed structure without performing additional recomputation.
    """

    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
        lower_is_better: bool = False,
        n_jobs: int = 1,
    ):
        super().__init__(
            name=name or "MetastabilityMetric",
            description=description
            or "Evaluates structure metastability from precomputed e_above_hull",
            lower_is_better=lower_is_better,
            n_jobs=n_jobs,
        )

    def _get_compute_attributes(self) -> dict[str, Any]:
        """Get the attributes for the compute_structure method."""
        return {}

    @staticmethod
    def compute_structure(structure: Structure) -> float:
        """Extract precomputed e_above_hull from structure properties.

        Parameters
        ----------
        structure : Structure
            A pymatgen Structure object with e_above_hull in properties.

        Returns
        -------
        float
            The precomputed e_above_hull value, or NaN if not available.
        """
        try:
            # Extract e_above_hull from structure properties
            e_above_hull = structure.properties.get("e_above_hull", None)

            if e_above_hull is None:
                logger.warning(
                    f"Structure `{structure.formula}` missing e_above_hull in properties, please compute it first using StabilityPreprocessor"
                )
                return np.nan

            return float(e_above_hull)

        except Exception as e:
            logger.error(f"Failed to extract e_above_hull: {str(e)}")
            return np.nan

    def aggregate_results(self, values: list[float]) -> dict[str, Any]:
        """Aggregate results into final metric values.

        Parameters
        ----------
        values : list[float]
            List of e_above_hull values for each structure.

        Returns
        -------
        dict
            Dictionary with aggregated metrics.
        """
        # Convert to numpy array for efficient operations
        values_array = np.array(values)

        # Filter out NaN values
        valid_mask = ~np.isnan(values_array)
        e_above_hull_values = values_array[valid_mask]

        if len(e_above_hull_values) > 0:
            # Calculate ratio of metastable structures (e_above_hull <= 0.1) using numpy
            metastable_ratio = np.sum(e_above_hull_values <= 0.1) / len(values)
        else:
            metastable_ratio = np.nan

        return {
            "metrics": {
                "metastable_ratio": metastable_ratio,
            },
            "primary_metric": "metastable_ratio",
            "uncertainties": {},
        }


class RelaxationStabilityMetric(BaseMetric):
    """Evaluate the RMSE between the atomic positions for the relaxed structure and
    generatd structure.

    This metric assumes that relaxed structures have already been computed and stored
    in the structure object at "structure.properties["relaxed_structure"]"
    It calculates relaxation stability statistics without performing any relaxation
    calculations.
    """

    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
        lower_is_better: bool = True,
        n_jobs: int = 1,
    ):
        super().__init__(
            name=name or "RelaxationStabilityMetric",
            description=description
            or "Evaluates RMSE between relaxed and unrelaxed structures",
            lower_is_better=lower_is_better,
            n_jobs=n_jobs,
        )

    def _get_compute_attributes(self) -> dict[str, Any]:
        """Get the attributes for the compute_structure method."""
        return {}

    @staticmethod
    def compute_structure(
        structure: Structure,
    ) -> float:
        """Extract precomputed relaxed structure from structure object.

        Parameters
        ----------
        structure : Structure
            A pymatgen Structure object with raw_structure as an attribute.

        Returns
        -------
        float
            The RMSE between the raw and relaxed structures.
        """
        relaxed_structure = structure.properties["relaxed_structure"]

        try:
            MSE = 0
            for site_index in range(0, len(structure)):
                strut_site = structure[site_index]
                relaxed_strut_site = relaxed_structure[site_index]
                MSE += (
                    np.linalg.norm(strut_site.coords - relaxed_strut_site.coords) ** 2
                )
            MSE = MSE / len(structure)
            RMSE = np.sqrt(MSE)
            print(str(structure.composition) + " Relaxation Stability RMSE :", RMSE)
            return RMSE

        except Exception as e:
            logger.error(f"Failed to extract RMSE: {str(e)}")
            return np.nan

    def aggregate_results(self, values: list[float]) -> dict[str, Any]:
        """Aggregate results into final metric values.

        Parameters
        ----------
        values : list[float]
            List of RMSE values for each pair of raw structure and relaxed structure.

        Returns
        -------
        dict
            Dictionary with aggregated metrics.
        """
        # Convert to numpy array for efficient operations
        values_array = np.array(values)

        # Filter out NaN values
        valid_mask = ~np.isnan(values_array)
        RMSE_values = values_array[valid_mask]

        if len(RMSE_values) > 0:
            mean_RMSE_values = np.mean(RMSE_values)
            std_RMSE_values = np.std(RMSE_values)
        else:
            mean_RMSE_values = np.nan
            std_RMSE_values = np.nan

        return {
            "metrics": {
                "mean_RMSE_values": mean_RMSE_values,
            },
            "primary_metric": "mean_RMSE_values",
            "uncertainties": {"std": std_RMSE_values},
        }
