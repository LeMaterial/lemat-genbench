"""Multi-MLIP stability metrics for ensemble-based stability evaluation.

This module provides stability metrics that work with the multi-MLIP preprocessing
results, using ensemble statistics for more robust stability assessment.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pymatgen.core import Structure

from lemat_genbench.metrics.base import BaseMetric
from lemat_genbench.utils.logging import logger


def safe_float_convert(value: Any) -> float:
    """Safely convert value to float with validation."""
    if value is None:
        return np.nan
    try:
        val = float(value)
        if np.isinf(val):
            logger.warning("Infinite value detected, returning NaN")
            return np.nan
        return val
    except (TypeError, ValueError, OverflowError):
        logger.warning(f"Cannot convert value to float: {value}")
        return np.nan


def extract_individual_values(
    structure: Structure, mlip_names: List[str], property_base: str
) -> Dict[str, float]:
    """Extract individual MLIP values for a property.

    Parameters
    ----------
    structure : Structure
        Structure with MLIP properties
    mlip_names : List[str]
        Names of MLIPs to extract from
    property_base : str
        Base property name (e.g., "e_above_hull", "formation_energy")

    Returns
    -------
    Dict[str, float]
        Dictionary mapping MLIP names to their values
    """
    individual_values = {}

    for mlip_name in mlip_names:
        prop_name = f"{property_base}_{mlip_name}"
        value = safe_float_convert(structure.properties.get(prop_name))
        individual_values[mlip_name] = value

    return individual_values


def extract_ensemble_value(
    structure: Structure, property_base: str, min_mlips_required: int = 2
) -> Tuple[float, float]:
    """Extract ensemble mean and std for a property.

    Parameters
    ----------
    structure : Structure
        Structure with ensemble properties
    property_base : str
        Base property name (e.g., "e_above_hull", "formation_energy")
    min_mlips_required : int, default=2
        Minimum MLIPs required for ensemble statistics

    Returns
    -------
    Tuple[float, float]
        Mean value and standard deviation, or (NaN, NaN) if insufficient data
    """
    mean_value = safe_float_convert(structure.properties.get(f"{property_base}_mean"))
    std_value = safe_float_convert(
        structure.properties.get(f"{property_base}_std", 0.0)
    )
    n_mlips = structure.properties.get(f"{property_base}_n_mlips", 0)

    # Check if we have sufficient MLIPs for ensemble
    if n_mlips < min_mlips_required:
        logger.warning(
            f"Insufficient MLIPs for ensemble {property_base}: {n_mlips} < {min_mlips_required}"
        )
        return np.nan, np.nan

    return mean_value, std_value


class StabilityMetric(BaseMetric):
    """Multi-MLIP stability metric supporting both individual and ensemble modes.

    Parameters
    ----------
    use_ensemble : bool, default=True
        Whether to use ensemble mean values or individual MLIP results
    mlip_names : List[str], optional
        Specific MLIPs to use if not using ensemble
    min_mlips_required : int, default=2
        Minimum MLIPs required for ensemble statistics
    include_individual_results : bool, default=True
        Whether to include individual MLIP results in output
    name : str, optional
        Custom name for the metric
    description : str, optional
        Description of what the metric evaluates
    lower_is_better : bool, default=False
        Whether lower values indicate better stability
    n_jobs : int, default=1
        Number of parallel jobs
    """

    def __init__(
        self,
        use_ensemble: bool = True,
        mlip_names: Optional[List[str]] = None,
        min_mlips_required: int = 2,
        include_individual_results: bool = True,
        name: str = None,
        description: str = None,
        lower_is_better: bool = False,
        n_jobs: int = 1,
    ):
        super().__init__(
            name=name or "StabilityMetric",
            description=description
            or "Evaluates structure stability from multi-MLIP e_above_hull predictions",
            lower_is_better=lower_is_better,
            n_jobs=n_jobs,
        )

        self.use_ensemble = use_ensemble
        self.mlip_names = mlip_names or ["orb", "mace", "uma"]
        self.min_mlips_required = min_mlips_required
        self.include_individual_results = include_individual_results

    def _get_compute_attributes(self) -> dict[str, Any]:
        """Get the attributes for the compute_structure method."""
        return {
            "use_ensemble": self.use_ensemble,
            "mlip_names": self.mlip_names,
            "min_mlips_required": self.min_mlips_required,
            "include_individual_results": self.include_individual_results,
        }

    @staticmethod
    def compute_structure(
        structure: Structure, **compute_args: Any
    ) -> Dict[str, float]:
        """Extract e_above_hull values from multi-MLIP structure properties.

        Returns
        -------
        Dict[str, float]
            Dictionary containing the primary value and optionally individual values and std
        """
        use_ensemble = compute_args.get("use_ensemble", True)
        mlip_names = compute_args.get("mlip_names", ["orb", "mace", "uma"])
        min_mlips_required = compute_args.get("min_mlips_required", 2)
        include_individual_results = compute_args.get(
            "include_individual_results", True
        )

        result = {}

        try:
            if use_ensemble:
                # Use ensemble statistics
                mean_value, std_value = extract_ensemble_value(
                    structure, "e_above_hull", min_mlips_required
                )
                result["value"] = mean_value
                result["std"] = std_value

                # Include individual results if requested
                if include_individual_results:
                    individual_values = extract_individual_values(
                        structure, mlip_names, "e_above_hull"
                    )
                    result.update(
                        {
                            f"value_{mlip}": val
                            for mlip, val in individual_values.items()
                        }
                    )
            else:
                # Use individual MLIP results
                individual_values = extract_individual_values(
                    structure, mlip_names, "e_above_hull"
                )

                # Calculate mean across available individual values
                valid_values = [
                    val for val in individual_values.values() if not np.isnan(val)
                ]
                if valid_values:
                    result["value"] = np.mean(valid_values)
                    result["std"] = (
                        np.std(valid_values) if len(valid_values) > 1 else 0.0
                    )
                else:
                    result["value"] = np.nan
                    result["std"] = np.nan

                # Always include individual results in individual mode
                result.update(
                    {f"value_{mlip}": val for mlip, val in individual_values.items()}
                )

        except Exception as e:
            logger.error(f"Failed to extract e_above_hull: {str(e)}")
            result["value"] = np.nan
            result["std"] = np.nan

        return result

    def aggregate_results(self, values: List[Dict[str, float]]) -> Dict[str, Any]:
        """Aggregate results for stability analysis."""

        # Extract primary values for main calculation
        primary_values = [v.get("value", np.nan) for v in values]
        values_array = np.array(primary_values)
        valid_mask = ~np.isnan(values_array)
        e_above_hull_values = values_array[valid_mask]

        metrics = {}
        uncertainties = {}

        if len(e_above_hull_values) > 0:
            # Calculate stable ratio and count (structures with e_above_hull <= 0)
            stable_count = np.sum(e_above_hull_values <= 0)
            stable_ratio = stable_count / len(values)  # Use total count including NaN

            # Raw statistics
            mean_e_above_hull = np.mean(e_above_hull_values)
            std_e_above_hull = np.std(e_above_hull_values)
            n_valid_structures = len(e_above_hull_values)

            metrics.update(
                {
                    "stable_ratio": stable_ratio,
                    "stable_count": int(stable_count),  # Add count
                    "mean_e_above_hull": mean_e_above_hull,
                    "std_e_above_hull": std_e_above_hull,
                    "n_valid_structures": n_valid_structures,
                    "total_structures_evaluated": len(values),  # Add total count
                }
            )

            uncertainties["stable_ratio"] = {
                "std": np.sqrt(stable_ratio * (1 - stable_ratio) / len(values))
                if len(values) > 0
                else np.nan,
                "sample_size": len(values),
            }
        else:
            metrics.update(
                {
                    "stable_ratio": np.nan,
                    "stable_count": 0,  # Add count
                    "mean_e_above_hull": np.nan,
                    "std_e_above_hull": np.nan,
                    "n_valid_structures": 0,
                    "total_structures_evaluated": len(values),  # Add total count
                }
            )

        # Add individual MLIP results if requested or if in individual mode
        if self.include_individual_results or not self.use_ensemble:
            for mlip_name in self.mlip_names:
                mlip_values = [v.get(f"value_{mlip_name}", np.nan) for v in values]
                mlip_array = np.array(mlip_values)
                mlip_valid = mlip_array[~np.isnan(mlip_array)]

                if len(mlip_valid) > 0:
                    # Calculate stable ratio and count for this MLIP
                    mlip_stable_count = np.sum(mlip_valid <= 0)
                    mlip_stable_ratio = mlip_stable_count / len(mlip_values)

                    metrics[f"stable_ratio_{mlip_name}"] = mlip_stable_ratio
                    metrics[f"stable_count_{mlip_name}"] = int(mlip_stable_count)  # Add count
                    metrics[f"mean_e_above_hull_{mlip_name}"] = np.mean(mlip_valid)
                    metrics[f"std_e_above_hull_{mlip_name}"] = np.std(mlip_valid)
                    metrics[f"n_valid_structures_{mlip_name}"] = len(mlip_valid)
                else:
                    metrics[f"stable_ratio_{mlip_name}"] = np.nan
                    metrics[f"stable_count_{mlip_name}"] = 0  # Add count
                    metrics[f"mean_e_above_hull_{mlip_name}"] = np.nan
                    metrics[f"std_e_above_hull_{mlip_name}"] = np.nan
                    metrics[f"n_valid_structures_{mlip_name}"] = 0

        # Add ensemble uncertainty metrics if available
        if self.use_ensemble:
            std_values = [v.get("std", np.nan) for v in values]
            std_array = np.array(std_values)
            valid_std = std_array[~np.isnan(std_array)]

            if len(valid_std) > 0:
                metrics["mean_ensemble_std"] = np.mean(valid_std)
                metrics["std_ensemble_std"] = np.std(valid_std)

        return {
            "metrics": metrics,
            "primary_metric": "stable_ratio",
            "uncertainties": uncertainties,
        }


class MetastabilityMetric(BaseMetric):
    """Multi-MLIP metastability metric supporting both individual and ensemble modes."""

    def __init__(
        self,
        use_ensemble: bool = True,
        mlip_names: Optional[List[str]] = None,
        metastable_threshold: float = 0.1,
        min_mlips_required: int = 2,
        include_individual_results: bool = True,
        name: str = None,
        description: str = None,
        lower_is_better: bool = False,
        n_jobs: int = 1,
    ):
        super().__init__(
            name=name or "MetastabilityMetric",
            description=description
            or "Evaluates structure metastability from multi-MLIP predictions",
            lower_is_better=lower_is_better,
            n_jobs=n_jobs,
        )

        self.use_ensemble = use_ensemble
        self.mlip_names = mlip_names or ["orb", "mace", "uma"]
        self.metastable_threshold = metastable_threshold
        self.min_mlips_required = min_mlips_required
        self.include_individual_results = include_individual_results

    def _get_compute_attributes(self) -> dict[str, Any]:
        """Get the attributes for the compute_structure method."""
        return {
            "use_ensemble": self.use_ensemble,
            "mlip_names": self.mlip_names,
            "metastable_threshold": self.metastable_threshold,
            "min_mlips_required": self.min_mlips_required,
            "include_individual_results": self.include_individual_results,
        }

    @staticmethod
    def compute_structure(
        structure: Structure, **compute_args: Any
    ) -> Dict[str, float]:
        """Extract e_above_hull for metastability evaluation."""
        # Reuse the same logic as StabilityMetric but pass through threshold
        return StabilityMetric.compute_structure(structure, **compute_args)

    def aggregate_results(self, values: List[Dict[str, float]]) -> Dict[str, Any]:
        """Aggregate results for metastability analysis."""
        # Extract primary values
        primary_values = [v.get("value", np.nan) for v in values]
        values_array = np.array(primary_values)
        valid_mask = ~np.isnan(values_array)
        e_above_hull_values = values_array[valid_mask]

        metrics = {}
        uncertainties = {}

        if len(e_above_hull_values) > 0:
            # Calculate metastable ratio and count
            metastable_count = np.sum(e_above_hull_values <= self.metastable_threshold)
            metastable_ratio = metastable_count / len(values)

            metrics.update(
                {
                    "metastable_ratio": metastable_ratio,
                    "metastable_count": int(metastable_count),  # Add count
                    "mean_e_above_hull": np.mean(e_above_hull_values),
                    "std_e_above_hull": np.std(e_above_hull_values),
                    "n_valid_structures": len(e_above_hull_values),
                    "metastable_threshold": self.metastable_threshold,
                    "total_structures_evaluated": len(values),  # Add total count
                }
            )

            uncertainties["metastable_ratio"] = {
                "std": np.sqrt(metastable_ratio * (1 - metastable_ratio) / len(values))
                if len(values) > 0
                else np.nan,
                "sample_size": len(values),
            }
        else:
            metrics.update(
                {
                    "metastable_ratio": np.nan,
                    "metastable_count": 0,  # Add count
                    "mean_e_above_hull": np.nan,
                    "std_e_above_hull": np.nan,
                    "n_valid_structures": 0,
                    "metastable_threshold": self.metastable_threshold,
                    "total_structures_evaluated": len(values),  # Add total count
                }
            )

        # Add individual MLIP results if requested or if in individual mode
        if self.include_individual_results or not self.use_ensemble:
            for mlip_name in self.mlip_names:
                mlip_values = [v.get(f"value_{mlip_name}", np.nan) for v in values]
                mlip_array = np.array(mlip_values)
                mlip_valid = mlip_array[~np.isnan(mlip_array)]

                if len(mlip_valid) > 0:
                    mlip_metastable_count = np.sum(
                        mlip_valid <= self.metastable_threshold
                    )
                    mlip_metastable_ratio = mlip_metastable_count / len(mlip_values)

                    metrics[f"metastable_ratio_{mlip_name}"] = mlip_metastable_ratio
                    metrics[f"metastable_count_{mlip_name}"] = int(mlip_metastable_count)  # Add count
                    metrics[f"mean_e_above_hull_{mlip_name}"] = np.mean(mlip_valid)
                    metrics[f"std_e_above_hull_{mlip_name}"] = np.std(mlip_valid)

        return {
            "metrics": metrics,
            "primary_metric": "metastable_ratio",
            "uncertainties": uncertainties,
        }


class E_HullMetric(BaseMetric):
    """Multi-MLIP energy above hull metric supporting both individual and ensemble modes."""

    def __init__(
        self,
        use_ensemble: bool = True,
        mlip_names: Optional[List[str]] = None,
        min_mlips_required: int = 2,
        include_individual_results: bool = True,
        name: str = None,
        description: str = None,
        lower_is_better: bool = True,
        n_jobs: int = 1,
    ):
        super().__init__(
            name=name or "E_HullMetric",
            description=description
            or "Evaluates mean energy above hull from multi-MLIP predictions",
            lower_is_better=lower_is_better,
            n_jobs=n_jobs,
        )

        self.use_ensemble = use_ensemble
        self.mlip_names = mlip_names or ["orb", "mace", "uma"]
        self.min_mlips_required = min_mlips_required
        self.include_individual_results = include_individual_results

    def _get_compute_attributes(self) -> dict[str, Any]:
        """Get the attributes for the compute_structure method."""
        return {
            "use_ensemble": self.use_ensemble,
            "mlip_names": self.mlip_names,
            "min_mlips_required": self.min_mlips_required,
            "include_individual_results": self.include_individual_results,
        }

    @staticmethod
    def compute_structure(
        structure: Structure, **compute_args: Any
    ) -> Dict[str, float]:
        """Extract e_above_hull for mean calculation."""
        return StabilityMetric.compute_structure(structure, **compute_args)

    def aggregate_results(self, values: List[Dict[str, float]]) -> Dict[str, Any]:
        """Aggregate results for mean E_hull analysis."""
        primary_values = [v.get("value", np.nan) for v in values]
        values_array = np.array(primary_values)
        valid_mask = ~np.isnan(values_array)
        e_above_hull_values = values_array[valid_mask]

        metrics = {}

        if len(e_above_hull_values) > 0:
            metrics.update(
                {
                    "mean_e_above_hull": np.mean(e_above_hull_values),
                    "std_e_above_hull": np.std(e_above_hull_values),
                    "n_valid_structures": len(e_above_hull_values),
                    "total_structures_evaluated": len(values),  # Add total count
                }
            )
        else:
            metrics.update(
                {
                    "mean_e_above_hull": np.nan,
                    "std_e_above_hull": np.nan,
                    "n_valid_structures": 0,
                    "total_structures_evaluated": len(values),  # Add total count
                }
            )

        # Add individual MLIP results if requested or if in individual mode
        if self.include_individual_results or not self.use_ensemble:
            for mlip_name in self.mlip_names:
                mlip_values = [v.get(f"value_{mlip_name}", np.nan) for v in values]
                mlip_array = np.array(mlip_values)
                mlip_valid = mlip_array[~np.isnan(mlip_array)]

                if len(mlip_valid) > 0:
                    metrics[f"mean_e_above_hull_{mlip_name}"] = np.mean(mlip_valid)
                    metrics[f"std_e_above_hull_{mlip_name}"] = np.std(mlip_valid)

        return {
            "metrics": metrics,
            "primary_metric": "mean_e_above_hull",
            "uncertainties": {
                "mean_e_above_hull": {"std": metrics.get("std_e_above_hull", np.nan)}
            },
        }


class FormationEnergyMetric(BaseMetric):
    """Multi-MLIP formation energy metric supporting both individual and ensemble modes."""

    def __init__(
        self,
        use_ensemble: bool = True,
        mlip_names: Optional[List[str]] = None,
        min_mlips_required: int = 2,
        include_individual_results: bool = True,
        name: str = None,
        description: str = None,
        lower_is_better: bool = True,
        n_jobs: int = 1,
    ):
        super().__init__(
            name=name or "FormationEnergyMetric",
            description=description
            or "Evaluates formation energy from multi-MLIP predictions",
            lower_is_better=lower_is_better,
            n_jobs=n_jobs,
        )

        self.use_ensemble = use_ensemble
        self.mlip_names = mlip_names or ["orb", "mace", "uma"]
        self.min_mlips_required = min_mlips_required
        self.include_individual_results = include_individual_results

    def _get_compute_attributes(self) -> dict[str, Any]:
        """Get the attributes for the compute_structure method."""
        return {
            "use_ensemble": self.use_ensemble,
            "mlip_names": self.mlip_names,
            "min_mlips_required": self.min_mlips_required,
            "include_individual_results": self.include_individual_results,
        }

    @staticmethod
    def compute_structure(
        structure: Structure, **compute_args: Any
    ) -> Dict[str, float]:
        """Extract formation energy values from multi-MLIP structure properties."""
        use_ensemble = compute_args.get("use_ensemble", True)
        mlip_names = compute_args.get("mlip_names", ["orb", "mace", "uma"])
        min_mlips_required = compute_args.get("min_mlips_required", 2)
        include_individual_results = compute_args.get(
            "include_individual_results", True
        )

        result = {}

        try:
            if use_ensemble:
                mean_value, std_value = extract_ensemble_value(
                    structure, "formation_energy", min_mlips_required
                )
                result["value"] = mean_value
                result["std"] = std_value

                if include_individual_results:
                    individual_values = extract_individual_values(
                        structure, mlip_names, "formation_energy"
                    )
                    result.update(
                        {
                            f"value_{mlip}": val
                            for mlip, val in individual_values.items()
                        }
                    )
            else:
                individual_values = extract_individual_values(
                    structure, mlip_names, "formation_energy"
                )
                valid_values = [
                    val for val in individual_values.values() if not np.isnan(val)
                ]

                if valid_values:
                    result["value"] = np.mean(valid_values)
                    result["std"] = (
                        np.std(valid_values) if len(valid_values) > 1 else 0.0
                    )
                else:
                    result["value"] = np.nan
                    result["std"] = np.nan

                result.update(
                    {f"value_{mlip}": val for mlip, val in individual_values.items()}
                )

        except Exception as e:
            logger.error(f"Failed to extract formation_energy: {str(e)}")
            result["value"] = np.nan
            result["std"] = np.nan

        return result

    def aggregate_results(self, values: List[Dict[str, float]]) -> Dict[str, Any]:
        """Aggregate results for formation energy analysis."""
        primary_values = [v.get("value", np.nan) for v in values]
        values_array = np.array(primary_values)
        valid_mask = ~np.isnan(values_array)
        formation_energy_values = values_array[valid_mask]

        metrics = {}

        if len(formation_energy_values) > 0:
            metrics.update(
                {
                    "mean_formation_energy": np.mean(formation_energy_values),
                    "std_formation_energy": np.std(formation_energy_values),
                    "n_valid_structures": len(formation_energy_values),
                    "total_structures_evaluated": len(values),  # Add total count
                }
            )
        else:
            metrics.update(
                {
                    "mean_formation_energy": np.nan,
                    "std_formation_energy": np.nan,
                    "n_valid_structures": 0,
                    "total_structures_evaluated": len(values),  # Add total count
                }
            )

        # Add individual MLIP results if requested or if in individual mode
        if self.include_individual_results or not self.use_ensemble:
            for mlip_name in self.mlip_names:
                mlip_values = [v.get(f"value_{mlip_name}", np.nan) for v in values]
                mlip_array = np.array(mlip_values)
                mlip_valid = mlip_array[~np.isnan(mlip_array)]

                if len(mlip_valid) > 0:
                    metrics[f"mean_formation_energy_{mlip_name}"] = np.mean(mlip_valid)
                    metrics[f"std_formation_energy_{mlip_name}"] = np.std(mlip_valid)

        return {
            "metrics": metrics,
            "primary_metric": "mean_formation_energy",
            "uncertainties": {
                "mean_formation_energy": {
                    "std": metrics.get("std_formation_energy", np.nan)
                }
            },
        }


class RelaxationStabilityMetric(BaseMetric):
    """Multi-MLIP relaxation stability metric supporting both individual and ensemble modes."""

    def __init__(
        self,
        use_ensemble: bool = True,
        mlip_names: Optional[List[str]] = None,
        min_mlips_required: int = 2,
        include_individual_results: bool = True,
        name: str = None,
        description: str = None,
        lower_is_better: bool = True,
        n_jobs: int = 1,
    ):
        super().__init__(
            name=name or "RelaxationStabilityMetric",
            description=description
            or "Evaluates relaxation stability from multi-MLIP RMSE predictions",
            lower_is_better=lower_is_better,
            n_jobs=n_jobs,
        )

        self.use_ensemble = use_ensemble
        self.mlip_names = mlip_names or ["orb", "mace", "uma"]
        self.min_mlips_required = min_mlips_required
        self.include_individual_results = include_individual_results

    def _get_compute_attributes(self) -> dict[str, Any]:
        """Get the attributes for the compute_structure method."""
        return {
            "use_ensemble": self.use_ensemble,
            "mlip_names": self.mlip_names,
            "min_mlips_required": self.min_mlips_required,
            "include_individual_results": self.include_individual_results,
        }

    @staticmethod
    def compute_structure(
        structure: Structure, **compute_args: Any
    ) -> Dict[str, float]:
        """Extract relaxation_rmse values from multi-MLIP structure properties."""
        use_ensemble = compute_args.get("use_ensemble", True)
        mlip_names = compute_args.get("mlip_names", ["orb", "mace", "uma"])
        min_mlips_required = compute_args.get("min_mlips_required", 2)
        include_individual_results = compute_args.get(
            "include_individual_results", True
        )

        result = {}

        try:
            if use_ensemble:
                mean_value, std_value = extract_ensemble_value(
                    structure, "relaxation_rmse", min_mlips_required
                )
                result["value"] = mean_value
                result["std"] = std_value

                if include_individual_results:
                    individual_values = extract_individual_values(
                        structure, mlip_names, "relaxation_rmse"
                    )
                    result.update(
                        {
                            f"value_{mlip}": val
                            for mlip, val in individual_values.items()
                        }
                    )
            else:
                individual_values = extract_individual_values(
                    structure, mlip_names, "relaxation_rmse"
                )
                valid_values = [
                    val for val in individual_values.values() if not np.isnan(val)
                ]

                if valid_values:
                    result["value"] = np.mean(valid_values)
                    result["std"] = (
                        np.std(valid_values) if len(valid_values) > 1 else 0.0
                    )
                else:
                    result["value"] = np.nan
                    result["std"] = np.nan

                result.update(
                    {f"value_{mlip}": val for mlip, val in individual_values.items()}
                )

        except Exception as e:
            logger.error(f"Failed to extract relaxation_rmse: {str(e)}")
            result["value"] = np.nan
            result["std"] = np.nan

        return result

    def aggregate_results(self, values: List[Dict[str, float]]) -> Dict[str, Any]:
        """Aggregate results for relaxation stability analysis."""
        primary_values = [v.get("value", np.nan) for v in values]
        values_array = np.array(primary_values)
        valid_mask = ~np.isnan(values_array)
        rmse_values = values_array[valid_mask]

        metrics = {}

        if len(rmse_values) > 0:
            metrics.update(
                {
                    "mean_relaxation_RMSE": np.mean(rmse_values),
                    "std_relaxation_RMSE": np.std(rmse_values),
                    "n_valid_structures": len(rmse_values),
                    "total_structures_evaluated": len(values),  # Add total count
                }
            )
        else:
            metrics.update(
                {
                    "mean_relaxation_RMSE": np.nan,
                    "std_relaxation_RMSE": np.nan,
                    "n_valid_structures": 0,
                    "total_structures_evaluated": len(values),  # Add total count
                }
            )

        # Add individual MLIP results if requested or if in individual mode
        if self.include_individual_results or not self.use_ensemble:
            for mlip_name in self.mlip_names:
                mlip_values = [v.get(f"value_{mlip_name}", np.nan) for v in values]
                mlip_array = np.array(mlip_values)
                mlip_valid = mlip_array[~np.isnan(mlip_array)]

                if len(mlip_valid) > 0:
                    metrics[f"mean_relaxation_RMSE_{mlip_name}"] = np.mean(mlip_valid)
                    metrics[f"std_relaxation_RMSE_{mlip_name}"] = np.std(mlip_valid)

        return {
            "metrics": metrics,
            "primary_metric": "mean_relaxation_RMSE",
            "uncertainties": {
                "mean_relaxation_RMSE": {
                    "std": metrics.get("std_relaxation_RMSE", np.nan)
                }
            },
        }