"""Multi-MLIP stability preprocessor for robust metric calculation.

This module implements a preprocessor that uses multiple MLIPs (Machine Learning
Interatomic Potentials) to calculate stability-related properties for structures.
It provides ensemble statistics and uncertainty quantification by combining
results from multiple MLIPs.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
from func_timeout import FunctionTimedOut, func_timeout
from pymatgen.core import Structure

from lemat_genbench.models.registry import get_calculator
from lemat_genbench.preprocess.base import BasePreprocessor, PreprocessorConfig
from lemat_genbench.utils.logging import logger

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def _detach_tensors(obj):
    """Recursively detach PyTorch tensors to make them serializable for multiprocessing."""
    import torch
    
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    elif isinstance(obj, dict):
        return {key: _detach_tensors(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_detach_tensors(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_detach_tensors(item) for item in obj)
    elif hasattr(obj, 'properties') and hasattr(obj.properties, 'items'):
        # Handle pymatgen Structure objects
        for key, value in obj.properties.items():
            obj.properties[key] = _detach_tensors(value)
        return obj
    else:
        return obj

def _create_clean_structure_copy(structure):
    """Create a clean copy of a Structure object without any tensors."""
    from pymatgen.core import Structure
    
    # Create a new structure with the same lattice and sites
    clean_structure = Structure(
        lattice=structure.lattice,
        species=[site.specie for site in structure],
        coords=[site.coords for site in structure],
        coords_are_cartesian=False
    )
    
    # Copy properties but detach any tensors
    if hasattr(structure, 'properties') and structure.properties:
        clean_structure.properties = _detach_tensors(structure.properties)
    
    return clean_structure
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Process-local model cache (each worker process gets its own models)
_PROCESS_MODEL_CACHE = {}

def _get_process_model_cache():
    """Get the process-local model cache."""
    return _PROCESS_MODEL_CACHE

def _get_or_create_calculator(mlip_name: str, mlip_config: Dict[str, Any]):
    """Get calculator from process cache or create if not exists."""
    cache = _get_process_model_cache()
    
    cache_key = f"{mlip_name}_{hash(str(mlip_config))}"
    
    if cache_key not in cache:
        logger.info(f"Loading {mlip_name} model in worker process...")
        cache[cache_key] = get_calculator(mlip_name, **mlip_config)
        logger.info(f"✅ {mlip_name} model loaded in worker")
    else:
        logger.debug(f"Using cached {mlip_name} model in worker")
    
    return cache[cache_key]

def _initialize_worker_models(mlip_names: List[str], mlip_configs: Dict[str, Dict[str, Any]]):
    """Initialize models in the main process (for logging only)."""
    logger.info("Models will be loaded in each worker process as needed...")
    
    for mlip_name in mlip_names:
        try:
            mlip_config = mlip_configs.get(mlip_name, {})

            # Use specific default configurations for each MLIP
            if mlip_name == "mace":
                default_config = {"model_type": "mp", "device": "cpu"}
            elif mlip_name == "orb":
                default_config = {
                    "model_type": "orb_v3_conservative_inf_omat",
                    "device": "cpu",
                }
            elif mlip_name == "uma":
                default_config = {"task": "omat", "device": "cpu"}
            else:
                default_config = {"device": "cpu"}

            # Merge user config with defaults (user config takes precedence)
            final_config = {**default_config, **mlip_config}
            
            logger.info(f"Will load {mlip_name} with config: {final_config}")
            
        except Exception as e:
            logger.error(f"Failed to configure {mlip_name} model: {str(e)}")
    
    logger.info(f"✅ Model configuration ready for {len(mlip_names)} models")


@dataclass
class MultiMLIPStabilityPreprocessorConfig(PreprocessorConfig):
    """Configuration for the Multi-MLIP Stability Preprocessor.

    Parameters
    ----------
    mlip_names : List[str]
        List of MLIP names to use ("orb", "mace", "uma", "equiformer")
    mlip_configs : Dict[str, Dict[str, Any]]
        Configuration for each specific MLIP
    relax_structures : bool
        Whether to relax structures during preprocessing
    relaxation_config : Dict[str, Any]
        Configuration for structure relaxation
    calculate_formation_energy : bool
        Whether to calculate formation energy
    calculate_energy_above_hull : bool
        Whether to calculate energy above hull
    extract_embeddings : bool
        Whether to extract embeddings
    timeout : int
        Timeout per structure per MLIP (seconds)
    """

    mlip_names: List[str] = field(default_factory=lambda: ["orb", "mace", "uma"])
    mlip_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    relax_structures: bool = True
    relaxation_config: Dict[str, Any] = field(
        default_factory=lambda: {"fmax": 0.05, "steps": 100}
    )
    calculate_formation_energy: bool = True
    calculate_energy_above_hull: bool = True
    extract_embeddings: bool = True
    timeout: int = 300


class MultiMLIPStabilityPreprocessor(BasePreprocessor):
    """Multi-MLIP stability preprocessor for ensemble predictions.

    This preprocessor can use multiple MLIPs simultaneously to calculate
    stability metrics and provide statistical robustness through ensemble
    predictions and uncertainty quantification.

    Parameters
    ----------
    mlip_names : List[str]
        List of MLIP names to use
    mlip_configs : Dict[str, Dict[str, Any]]
        Configuration for each specific MLIP
    relax_structures : bool
        Whether to relax structures
    relaxation_config : Dict[str, Any]
        Configuration for relaxation (fmax, steps, etc.)
    calculate_formation_energy : bool
        Whether to calculate formation energy
    calculate_energy_above_hull : bool
        Whether to calculate energy above hull
    extract_embeddings : bool
        Whether to extract embeddings
    timeout : int
        Timeout per structure per MLIP (seconds)
    name : str, optional
        Custom name for the preprocessor
    description : str, optional
        Description of what the preprocessor does
    n_jobs : int, default=1
        Number of parallel jobs to run
    """

    def __init__(
        self,
        mlip_names: List[str] = None,
        mlip_configs: Dict[str, Dict[str, Any]] = None,
        relax_structures: bool = True,
        relaxation_config: Dict[str, Any] = None,
        calculate_formation_energy: bool = True,
        calculate_energy_above_hull: bool = True,
        extract_embeddings: bool = True,
        timeout: int = 300,
        name: str = None,
        description: str = None,
        n_jobs: int = 4,  # Use 4 cores to balance speed and memory
    ):
        # Set defaults
        if mlip_names is None:
            mlip_names = ["orb", "mace", "uma"]
        if mlip_configs is None:
            mlip_configs = {}
        if relaxation_config is None:
            relaxation_config = {"fmax": 0.05, "steps": 100}

        # Handle n_jobs=-1 to use all CPU cores
        if n_jobs == -1:
            import multiprocessing
            n_jobs = multiprocessing.cpu_count()
            logger.info(f"Using all available CPU cores: {n_jobs}")

        super().__init__(
            name=name or f"MultiMLIPStabilityPreprocessor_{'-'.join(mlip_names)}",
            description=description
            or f"Multi-MLIP stability preprocessing using {', '.join(mlip_names)}",
            n_jobs=n_jobs,
        )

        self.config = MultiMLIPStabilityPreprocessorConfig(
            name=self.config.name,
            description=self.config.description,
            n_jobs=self.config.n_jobs,
            mlip_names=mlip_names,
            mlip_configs=mlip_configs,
            relax_structures=relax_structures,
            relaxation_config=relaxation_config,
            calculate_formation_energy=calculate_formation_energy,
            calculate_energy_above_hull=calculate_energy_above_hull,
            extract_embeddings=extract_embeddings,
            timeout=timeout,
        )

        # Configure models (will be loaded in each worker process)
        _initialize_worker_models(self.config.mlip_names, self.config.mlip_configs)

    def _get_process_attributes(self) -> Dict[str, Any]:
        """Get the attributes for the process_structure method."""
        return {
            "mlip_names": self.config.mlip_names,
            "mlip_configs": self.config.mlip_configs,
            "timeout": self.config.timeout,
            "relax_structures": self.config.relax_structures,
            "relaxation_config": self.config.relaxation_config,
            "calculate_formation_energy": self.config.calculate_formation_energy,
            "calculate_energy_above_hull": self.config.calculate_energy_above_hull,
            "extract_embeddings": self.config.extract_embeddings,
        }

    @staticmethod
    def process_structure(
        structure: Structure,
        mlip_names: List[str],
        mlip_configs: Dict[str, Dict[str, Any]],
        timeout: int,
        relax_structures: bool,
        relaxation_config: Dict[str, Any],
        calculate_formation_energy: bool,
        calculate_energy_above_hull: bool,
        extract_embeddings: bool,
    ) -> Structure:
        """Process a single structure using multiple MLIPs.

        Parameters
        ----------
        structure : Structure
            A pymatgen Structure object to process
        mlip_names : List[str]
            List of MLIP names to use
        mlip_configs : Dict[str, Dict[str, Any]]
            Configuration for each MLIP
        timeout : int
            Timeout per MLIP calculation
        relax_structures : bool
            Whether to relax the structure
        relaxation_config : Dict[str, Any]
            Configuration for relaxation
        calculate_formation_energy : bool
            Whether to calculate formation energy
        calculate_energy_above_hull : bool
            Whether to calculate energy above hull
        extract_embeddings : bool
            Whether to extract embeddings

        Returns
        -------
        Structure
            The processed Structure with calculated properties from all MLIPs
        """
        try:
            # Create calculators locally to avoid pickling issues
            calculators = {}
            for mlip_name in mlip_names:
                try:
                    mlip_config = mlip_configs.get(mlip_name, {})

                    # Use specific default configurations for each MLIP
                    if mlip_name == "mace":
                        default_config = {"model_type": "mp", "device": "cpu"}
                    elif mlip_name == "orb":
                        default_config = {
                            "model_type": "orb_v3_conservative_inf_omat",
                            "device": "cpu",
                        }
                    elif mlip_name == "uma":
                        default_config = {"task": "omat", "device": "cpu"}
                    else:
                        default_config = {"device": "cpu"}

                    # Merge user config with defaults (user config takes precedence)
                    final_config = {**default_config, **mlip_config}

                    calculators[mlip_name] = _get_or_create_calculator(mlip_name, final_config)
                except Exception as e:
                    logger.warning(f"Failed to get calculator for {mlip_name}: {str(e)}")
                    continue

            # Storage for results from each MLIP
            mlip_results = {}

            # Storage for scalar metrics for statistical analysis
            scalar_metrics = {
                "energy": [],
                "formation_energy": [],
                "e_above_hull": [],
                "relaxation_energy": [],
                "relaxation_rmse": [],
                "relaxation_steps": [],
                "relaxed_formation_energy": [],
                "relaxed_e_above_hull": [],
            }

            # Storage for vector metrics (to be averaged)
            vector_metrics = {
                "forces": [],
            }

            # Process with each MLIP
            for mlip_name, calculator in calculators.items():
                logger.debug(f"Processing {structure.formula} with {mlip_name}")

                try:
                    # Run MLIP-specific calculations with timeout
                    mlip_result = func_timeout(
                        timeout,
                        _process_single_mlip,
                        args=[
                            structure,
                            calculator,
                            mlip_name,
                            relax_structures,
                            relaxation_config,
                            calculate_formation_energy,
                            calculate_energy_above_hull,
                            extract_embeddings,
                        ],
                    )

                    mlip_results[mlip_name] = mlip_result

                    # Collect scalar metrics for statistics
                    for metric_name in scalar_metrics.keys():
                        if (
                            metric_name in mlip_result
                            and mlip_result[metric_name] is not None
                        ):
                            scalar_metrics[metric_name].append(mlip_result[metric_name])

                    # Collect vector metrics for averaging
                    if "forces" in mlip_result and mlip_result["forces"] is not None:
                        vector_metrics["forces"].append(mlip_result["forces"])

                except FunctionTimedOut:
                    logger.warning(
                        f"Timeout processing {structure.formula} with {mlip_name}"
                    )
                    mlip_results[mlip_name] = {"error": "timeout"}
                except Exception as e:
                    logger.warning(
                        f"Error processing {structure.formula} with {mlip_name}: {str(e)}"
                    )
                    mlip_results[mlip_name] = {"error": str(e)}

            # Store individual MLIP results
            for mlip_name, result in mlip_results.items():
                if "error" not in result:
                    _store_mlip_properties(structure, mlip_name, result)

            # Calculate and store ensemble statistics
            _calculate_ensemble_statistics(structure, scalar_metrics, vector_metrics)

            # Create a clean copy of the structure before returning (for multiprocessing serialization)
            clean_structure = _create_clean_structure_copy(structure)

            return clean_structure

        except Exception as e:
            logger.error(f"Failed to process structure {structure.formula}: {str(e)}")
            raise


def _process_single_mlip(
    structure: Structure,
    calculator: Any,
    mlip_name: str,
    relax_structures: bool,
    relaxation_config: Dict[str, Any],
    calculate_formation_energy: bool,
    calculate_energy_above_hull: bool,
    extract_embeddings: bool,
) -> Dict[str, Any]:
    """Process structure with a single MLIP and return results."""

    results = {}

    try:
        # Store model information
        results["mlip_model"] = calculator.__class__.__name__
        results["model_config"] = getattr(calculator, "model_type", "unknown")

        # Calculate basic energy and forces
        energy_result = calculator.calculate_energy_forces(structure)
        results["energy"] = _detach_tensors(energy_result.energy)
        results["forces"] = _detach_tensors(energy_result.forces)

        # Calculate formation energy if requested
        if calculate_formation_energy:
            try:
                formation_energy = calculator.calculate_formation_energy(structure)
                results["formation_energy"] = _detach_tensors(formation_energy)
                logger.debug(
                    f"[{mlip_name}] Formation energy: {formation_energy:.3f} eV/atom for {structure.formula}"
                )
            except Exception as e:
                logger.warning(
                    f"[{mlip_name}] Failed to compute formation_energy for {structure.formula}: {str(e)}"
                )
                results["formation_energy"] = None

        # Calculate energy above hull if requested
        if calculate_energy_above_hull:
            try:
                e_above_hull = calculator.calculate_energy_above_hull(structure)
                results["e_above_hull"] = _detach_tensors(e_above_hull)
                logger.debug(
                    f"[{mlip_name}] E_above_hull: {e_above_hull:.3f} eV/atom for {structure.formula}"
                )
            except Exception as e:
                logger.warning(
                    f"[{mlip_name}] Failed to compute e_above_hull for {structure.formula}: {str(e)}"
                )
                results["e_above_hull"] = None

        # Extract embeddings if requested
        if extract_embeddings:
            try:
                embeddings = calculator.extract_embeddings(structure)
                results["node_embeddings"] = _detach_tensors(embeddings.node_embeddings)
                results["graph_embedding"] = _detach_tensors(embeddings.graph_embedding)
            except Exception as e:
                logger.warning(
                    f"[{mlip_name}] Failed to extract embeddings for {structure.formula}: {str(e)}"
                )
                results["node_embeddings"] = None
                results["graph_embedding"] = None

        # Relax structure if requested
        if relax_structures:
            try:
                relaxed_structure, relaxation_result = calculator.relax_structure(
                    structure, **relaxation_config
                )

                # Calculate RMSE between original and relaxed positions
                rmse = _calculate_rmse(structure, relaxed_structure)

                # Store relaxation results - create clean copy of relaxed structure
                relaxed_structure_clean = _create_clean_structure_copy(relaxed_structure)
                results["relaxed_structure"] = relaxed_structure_clean
                results["relaxation_rmse"] = rmse
                results["relaxation_energy"] = _detach_tensors(relaxation_result.energy)
                results["relaxation_steps"] = relaxation_result.metadata.get(
                    "relaxation_steps", None
                )

                logger.debug(
                    f"[{mlip_name}] Relaxation RMSE: {rmse:.3f} Å for {structure.formula}"
                )

                # Calculate properties for relaxed structure if requested
                if calculate_formation_energy:
                    try:
                        relaxed_formation_energy = (
                            calculator.calculate_formation_energy(relaxed_structure)
                        )
                        results["relaxed_formation_energy"] = _detach_tensors(relaxed_formation_energy)
                    except Exception as e:
                        logger.warning(
                            f"[{mlip_name}] Failed to compute relaxed formation_energy: {str(e)}"
                        )
                        results["relaxed_formation_energy"] = None

                if calculate_energy_above_hull:
                    try:
                        relaxed_e_above_hull = calculator.calculate_energy_above_hull(
                            relaxed_structure
                        )
                        results["relaxed_e_above_hull"] = _detach_tensors(relaxed_e_above_hull)
                    except Exception as e:
                        logger.warning(
                            f"[{mlip_name}] Failed to compute relaxed e_above_hull: {str(e)}"
                        )
                        results["relaxed_e_above_hull"] = None

            except Exception as e:
                logger.warning(
                    f"[{mlip_name}] Failed to relax structure {structure.formula}: {str(e)}"
                )
                results["relaxation_failed"] = True
                results["relaxation_error"] = str(e)

        return results

    except Exception as e:
        logger.error(
            f"[{mlip_name}] Failed to process structure {structure.formula}: {str(e)}"
        )
        raise


def _store_mlip_properties(
    structure: Structure, mlip_name: str, results: Dict[str, Any]
) -> None:
    """Store MLIP-specific properties in the structure."""

    # Scalar properties
    scalar_props = [
        "energy",
        "formation_energy",
        "e_above_hull",
        "relaxation_energy",
        "relaxation_rmse",
        "relaxation_steps",
        "relaxed_formation_energy",
        "relaxed_e_above_hull",
    ]

    for prop in scalar_props:
        if prop in results and results[prop] is not None:
            structure.properties[f"{prop}_{mlip_name}"] = results[prop]

    # Vector properties
    if "forces" in results and results["forces"] is not None:
        structure.properties[f"forces_{mlip_name}"] = results["forces"]

    # Embedding properties
    if "node_embeddings" in results and results["node_embeddings"] is not None:
        structure.properties[f"node_embeddings_{mlip_name}"] = results[
            "node_embeddings"
        ]

    if "graph_embedding" in results and results["graph_embedding"] is not None:
        structure.properties[f"graph_embedding_{mlip_name}"] = results[
            "graph_embedding"
        ]

    # Structure objects
    if "relaxed_structure" in results and results["relaxed_structure"] is not None:
        structure.properties[f"relaxed_structure_{mlip_name}"] = results[
            "relaxed_structure"
        ]

    # String/metadata properties
    if "mlip_model" in results:
        structure.properties[f"mlip_model_{mlip_name}"] = results["mlip_model"]

    if "model_config" in results:
        structure.properties[f"model_config_{mlip_name}"] = results["model_config"]

    # Boolean/error properties
    if "relaxation_failed" in results:
        structure.properties[f"relaxation_failed_{mlip_name}"] = results[
            "relaxation_failed"
        ]

    if "relaxation_error" in results:
        structure.properties[f"relaxation_error_{mlip_name}"] = results[
            "relaxation_error"
        ]


def _calculate_ensemble_statistics(
    structure: Structure,
    scalar_metrics: Dict[str, List[float]],
    vector_metrics: Dict[str, List[np.ndarray]],
) -> None:
    """Calculate ensemble statistics and store in structure properties."""

    # Calculate statistics for scalar metrics
    for metric_name, values in scalar_metrics.items():
        if values:  # Only calculate if we have values
            values_array = np.array([v for v in values if v is not None])

            if len(values_array) > 0:
                # Store mean and std
                structure.properties[f"{metric_name}_mean"] = float(
                    np.mean(values_array)
                )
                structure.properties[f"{metric_name}_std"] = float(np.std(values_array))

                # Store number of MLIPs that contributed
                structure.properties[f"{metric_name}_n_mlips"] = len(values_array)
            else:
                structure.properties[f"{metric_name}_mean"] = None
                structure.properties[f"{metric_name}_std"] = None
                structure.properties[f"{metric_name}_n_mlips"] = 0

    # Calculate averages for vector metrics
    for metric_name, arrays in vector_metrics.items():
        if arrays:  # Only calculate if we have arrays
            # Filter out None values
            valid_arrays = [arr for arr in arrays if arr is not None]

            if valid_arrays:
                # Check that all arrays have the same shape
                shapes = [arr.shape for arr in valid_arrays]
                if len(set(shapes)) == 1:  # All shapes are the same
                    # Calculate mean across MLIPs
                    stacked_arrays = np.stack(valid_arrays, axis=0)
                    mean_array = np.mean(stacked_arrays, axis=0)
                    std_array = np.std(stacked_arrays, axis=0)

                    structure.properties[f"{metric_name}_mean"] = mean_array
                    structure.properties[f"{metric_name}_std"] = std_array
                    structure.properties[f"{metric_name}_n_mlips"] = len(valid_arrays)
                else:
                    logger.warning(f"Inconsistent shapes for {metric_name}: {shapes}")
                    structure.properties[f"{metric_name}_mean"] = None
                    structure.properties[f"{metric_name}_std"] = None
                    structure.properties[f"{metric_name}_n_mlips"] = 0


def _calculate_rmse(original: Structure, relaxed: Structure) -> float:
    """Calculate RMSE between atomic positions of original and relaxed structures.

    Parameters
    ----------
    original : Structure
        Original structure
    relaxed : Structure
        Relaxed structure

    Returns
    -------
    float
        RMSE in Angstroms
    """
    if len(original) != len(relaxed):
        raise ValueError("Structures must have the same number of atoms")

    mse = 0.0
    for i in range(len(original)):
        original_coords = original[i].coords
        relaxed_coords = relaxed[i].coords
        mse += np.linalg.norm(original_coords - relaxed_coords) ** 2

    mse /= len(original)
    return np.sqrt(mse)


# Factory functions for common configurations
def create_multi_mlip_preprocessor(
    mlip_names: List[str] = None, relax_structures: bool = True, **kwargs
) -> MultiMLIPStabilityPreprocessor:
    """Factory function to create multi-MLIP preprocessor with common configurations.

    Parameters
    ----------
    mlip_names : List[str], optional
        List of MLIP names to use. Defaults to ["orb", "mace", "uma"]
    relax_structures : bool
        Whether to relax structures
    **kwargs
        Additional arguments for the preprocessor

    Returns
    -------
    MultiMLIPStabilityPreprocessor
        Configured multi-MLIP preprocessor
    """
    if mlip_names is None:
        mlip_names = ["orb", "mace", "uma"]

    # Default configurations for each MLIP (matching your specifications)
    default_configs = {
        "mace": {"model_type": "mp", "device": "cpu"},
        "orb": {"model_type": "orb_v3_conservative_inf_omat", "device": "cpu"},
        "uma": {"task": "omat", "device": "cpu"},
    }

    # Use provided configs or defaults
    mlip_configs = kwargs.pop("mlip_configs", {})
    for mlip_name in mlip_names:
        if mlip_name not in mlip_configs:
            mlip_configs[mlip_name] = default_configs.get(mlip_name, {"device": "cpu"})

    return MultiMLIPStabilityPreprocessor(
        mlip_names=mlip_names,
        mlip_configs=mlip_configs,
        relax_structures=relax_structures,
        **kwargs,
    )


def create_orb_mace_uma_preprocessor(**kwargs) -> MultiMLIPStabilityPreprocessor:
    """Create preprocessor using ORB, MACE, and UMA with default configurations."""
    return create_multi_mlip_preprocessor(mlip_names=["orb", "mace", "uma"], **kwargs)


def create_all_mlip_preprocessor(**kwargs) -> MultiMLIPStabilityPreprocessor:
    """Create preprocessor using ORB, MACE, and UMA (no Equiformer)."""
    return create_multi_mlip_preprocessor(mlip_names=["orb", "mace", "uma"], **kwargs)
