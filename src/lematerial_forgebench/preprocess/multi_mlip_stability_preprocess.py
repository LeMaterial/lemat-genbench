"""Multi-MLIP stability preprocessor for ensemble calculations."""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
from func_timeout import FunctionTimedOut, func_timeout
from pymatgen.core import Structure

from lematerial_forgebench.models.registry import (
    get_calculator,
    list_available_models,
)
from lematerial_forgebench.preprocess.base import (
    BasePreprocessor,
    PreprocessorConfig,
)
from lematerial_forgebench.utils.logging import logger


@dataclass
class MultiMLIPStabilityPreprocessorConfig(PreprocessorConfig):
    """Configuration for Multi-MLIP Stability Preprocessor.

    Parameters
    ----------
    models : List[str]
        List of MLIP model names to use (e.g., ["orb", "mace", "uma"])
    timeout : int
        Timeout in seconds for each MLIP calculation
    model_configs : Dict[str, Dict[str, Any]]
        Model-specific configurations, keyed by model name
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
    energy_aggregation : str
        How to aggregate energy results: "individual", "aggregated", "both"
    rmse_aggregation : str
        How to aggregate RMSE results: "individual", "aggregated", "both"
    embedding_strategy : str
        How to handle embeddings: "individual", "aggregated", "both"
    """

    models: List[str] = field(default_factory=lambda: ["orb", "mace", "uma"])
    timeout: int = 60
    model_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Processing options
    relax_structures: bool = True
    relaxation_config: Dict[str, Any] = field(
        default_factory=lambda: {"fmax": 0.02, "steps": 500}
    )
    calculate_formation_energy: bool = True
    calculate_energy_above_hull: bool = True
    extract_embeddings: bool = True

    # Aggregation strategies
    energy_aggregation: str = "both"  # "individual", "aggregated", "both"
    rmse_aggregation: str = "both"  # "individual", "aggregated", "both"
    embedding_strategy: str = "both"  # "individual", "aggregated", "both"


class MultiMLIPStabilityPreprocessor(BasePreprocessor):
    """Multi-MLIP stability preprocessor for ensemble calculations.

    This preprocessor runs multiple MLIPs on the same structures and provides
    both individual model results and ensemble aggregations.

    Parameters
    ----------
    models : List[str], default=["orb", "mace", "uma"]
        List of MLIP model names to use
    model_configs : Dict[str, Dict[str, Any]], optional
        Model-specific configurations
    energy_aggregation : str, default="both"
        How to aggregate energy results
    rmse_aggregation : str, default="both"
        How to aggregate RMSE results
    embedding_strategy : str, default="both"
        How to handle embeddings
    **kwargs
        Additional arguments passed to BasePreprocessor
    """

    def __init__(
        self,
        models: List[str] = None,
        model_configs: Dict[str, Dict[str, Any]] = None,
        energy_aggregation: str = "both",
        rmse_aggregation: str = "both",
        embedding_strategy: str = "both",
        timeout: int = 60,
        relax_structures: bool = True,
        relaxation_config: Dict[str, Any] = None,
        calculate_formation_energy: bool = True,
        calculate_energy_above_hull: bool = True,
        extract_embeddings: bool = True,
        name: str = None,
        description: str = None,
        n_jobs: int = 1,
    ):
        models = models or ["orb", "mace", "uma"]
        model_configs = model_configs or {}

        super().__init__(
            name=name or f"MultiMLIPStabilityPreprocessor_{'+'.join(models)}",
            description=description
            or f"Multi-MLIP stability preprocessing using {', '.join(models)}",
            n_jobs=n_jobs,
        )

        self.config = MultiMLIPStabilityPreprocessorConfig(
            name=self.config.name,
            description=self.config.description,
            n_jobs=self.config.n_jobs,
            models=models,
            timeout=timeout,
            model_configs=model_configs,
            relax_structures=relax_structures,
            relaxation_config=relaxation_config or {"fmax": 0.02, "steps": 500},
            calculate_formation_energy=calculate_formation_energy,
            calculate_energy_above_hull=calculate_energy_above_hull,
            extract_embeddings=extract_embeddings,
            energy_aggregation=energy_aggregation,
            rmse_aggregation=rmse_aggregation,
            embedding_strategy=embedding_strategy,
        )

        # Initialize calculators for all models
        self.calculators = {}
        available_models = list_available_models()

        for model_name in models:
            if model_name not in available_models:
                raise ValueError(
                    f"Model '{model_name}' not available. "
                    f"Available models: {available_models}"
                )

            try:
                model_config = model_configs.get(model_name, {})
                calculator = get_calculator(model_name, **model_config)
                self.calculators[model_name] = calculator
                logger.info(f"Successfully initialized {model_name} calculator")
            except Exception as e:
                raise ValueError(
                    f"Failed to initialize {model_name} calculator: {str(e)}"
                ) from e

    def _get_process_attributes(self) -> Dict[str, Any]:
        """Get the attributes for the process_structure method."""
        return {
            "calculators": self.calculators,
            "timeout": self.config.timeout,
            "models": self.config.models,
            "relax_structures": self.config.relax_structures,
            "relaxation_config": self.config.relaxation_config,
            "calculate_formation_energy": self.config.calculate_formation_energy,
            "calculate_energy_above_hull": self.config.calculate_energy_above_hull,
            "extract_embeddings": self.config.extract_embeddings,
            "energy_aggregation": self.config.energy_aggregation,
            "rmse_aggregation": self.config.rmse_aggregation,
            "embedding_strategy": self.config.embedding_strategy,
        }

    @staticmethod
    def process_structure(
        structure: Structure,
        calculators: Dict[str, Any],
        timeout: int,
        models: List[str],
        relax_structures: bool,
        relaxation_config: Dict[str, Any],
        calculate_formation_energy: bool,
        calculate_energy_above_hull: bool,
        extract_embeddings: bool,
        energy_aggregation: str,
        rmse_aggregation: str,
        embedding_strategy: str,
    ) -> Structure:
        """Process a single structure using multiple MLIPs.

        Parameters
        ----------
        structure : Structure
            A pymatgen Structure object to process
        calculators : Dict[str, Any]
            Dictionary of MLIP calculators keyed by model name
        timeout : int
            Timeout for each MLIP calculation
        models : List[str]
            List of model names to use
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
        energy_aggregation : str
            How to aggregate energy results
        rmse_aggregation : str
            How to aggregate RMSE results
        embedding_strategy : str
            How to handle embeddings

        Returns
        -------
        Structure
            The processed Structure with calculated properties from all models

        Raises
        ------
        Exception
            If computation fails for any model
        """
        try:
            result = func_timeout(
                timeout,
                _process_structure_with_multi_mlip,
                [
                    structure,
                    calculators,
                    models,
                    relax_structures,
                    relaxation_config,
                    calculate_formation_energy,
                    calculate_energy_above_hull,
                    extract_embeddings,
                    energy_aggregation,
                    rmse_aggregation,
                    embedding_strategy,
                ],
            )
            return result
        except FunctionTimedOut:
            logger.warning(f"Multi-MLIP processing timed out for {structure.formula}")
            return structure


def _process_structure_with_multi_mlip(
    structure: Structure,
    calculators: Dict[str, Any],
    models: List[str],
    relax_structures: bool,
    relaxation_config: Dict[str, Any],
    calculate_formation_energy: bool,
    calculate_energy_above_hull: bool,
    extract_embeddings: bool,
    energy_aggregation: str,
    rmse_aggregation: str,
    embedding_strategy: str,
) -> Structure:
    """Process structure with multiple MLIPs and aggregate results."""

    # Store results for each model
    model_results = {}

    for model_name in models:
        calculator = calculators[model_name]

        try:
            logger.debug(f"Processing {structure.formula} with {model_name}")

            # Calculate basic energy and forces
            energy_result = calculator.calculate_energy_forces(structure)

            model_results[model_name] = {
                "energy": energy_result.energy,
                "forces": energy_result.forces,
                "model_info": energy_result.metadata,
            }

            # Calculate formation energy if requested
            if calculate_formation_energy:
                try:
                    formation_energy = calculator.calculate_formation_energy(structure)
                    model_results[model_name]["formation_energy"] = formation_energy
                    logger.debug(
                        f"Computed formation_energy ({model_name}): {formation_energy:.3f} eV/atom for {structure.formula}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to compute formation_energy with {model_name} for {structure.formula}: {str(e)}"
                    )
                    model_results[model_name]["formation_energy"] = None

            # Calculate energy above hull if requested
            if calculate_energy_above_hull:
                try:
                    e_above_hull = calculator.calculate_energy_above_hull(structure)
                    model_results[model_name]["e_above_hull"] = e_above_hull
                    logger.debug(
                        f"Computed e_above_hull ({model_name}): {e_above_hull:.3f} eV/atom for {structure.formula}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to compute e_above_hull with {model_name} for {structure.formula}: {str(e)}"
                    )
                    model_results[model_name]["e_above_hull"] = None

            # Extract embeddings if requested
            if extract_embeddings:
                try:
                    embedding_result = calculator.extract_embeddings(structure)
                    model_results[model_name]["node_embeddings"] = (
                        embedding_result.node_embeddings
                    )
                    model_results[model_name]["graph_embedding"] = (
                        embedding_result.graph_embedding
                    )
                    logger.debug(
                        f"Extracted embeddings ({model_name}) for {structure.formula}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to extract embeddings with {model_name} for {structure.formula}: {str(e)}"
                    )
                    model_results[model_name]["node_embeddings"] = None
                    model_results[model_name]["graph_embedding"] = None

            # Handle relaxation if requested
            if relax_structures:
                try:
                    # The relax_structure method returns (relaxed_structure, calculation_result)
                    relaxed_structure, relaxation_result = calculator.relax_structure(
                        structure, **relaxation_config
                    )

                    # Calculate RMSE between original and relaxed
                    rmse = _calculate_rmse(structure, relaxed_structure)
                    model_results[model_name]["relaxation_rmse"] = rmse
                    model_results[model_name]["relaxed_structure"] = relaxed_structure
                    model_results[model_name]["relaxed_energy"] = (
                        relaxation_result.energy
                    )

                    logger.debug(
                        f"Relaxed structure ({model_name}): RMSE: {rmse:.3f} Å for {structure.formula}"
                    )

                except Exception as e:
                    logger.warning(
                        f"Failed to relax structure with {model_name} for {structure.formula}: {str(e)}"
                    )
                    model_results[model_name]["relaxation_rmse"] = None
                    model_results[model_name]["relaxed_structure"] = None

        except Exception as e:
            logger.error(f"Failed to process structure with {model_name}: {str(e)}")
            raise

    # Store individual model results in structure properties
    _store_individual_results(
        structure,
        model_results,
        models,
        energy_aggregation,
        rmse_aggregation,
        embedding_strategy,
    )

    # Aggregate results based on configuration
    _aggregate_results(
        structure,
        model_results,
        models,
        energy_aggregation,
        rmse_aggregation,
        embedding_strategy,
    )

    return structure


def _store_individual_results(
    structure: Structure, 
    model_results: Dict[str, Dict[str, Any]], 
    models: List[str],
    energy_aggregation: str,  # Ignored - we always store individual results
    rmse_aggregation: str,    # Ignored - we always store individual results
    embedding_strategy: str,  # Ignored - we always store individual results
) -> None:
    """Store individual model results in structure properties.
    
    Always stores all individual model results regardless of aggregation settings.
    """
    
    for model_name in models:
        if model_name in model_results:
            results = model_results[model_name]
            
            # Always store model identifier
            structure.properties[f"mlip_model_{model_name}"] = results.get("model_info", {})
            
            # Always store individual energy results
            structure.properties[f"energy_{model_name}"] = results.get("energy")
            structure.properties[f"formation_energy_{model_name}"] = results.get("formation_energy")
            structure.properties[f"e_above_hull_{model_name}"] = results.get("e_above_hull")
            structure.properties[f"forces_{model_name}"] = results.get("forces")
            
            # Always store individual relaxation results
            structure.properties[f"relaxation_rmse_{model_name}"] = results.get("relaxation_rmse")
            
            # Always store individual embeddings
            structure.properties[f"node_embeddings_{model_name}"] = results.get("node_embeddings")
            structure.properties[f"graph_embedding_{model_name}"] = results.get("graph_embedding")


def _aggregate_results(
    structure: Structure,
    model_results: Dict[str, Dict[str, Any]], 
    models: List[str],
    energy_aggregation: str,  # Ignored - we aggregate if 2+ models
    rmse_aggregation: str,    # Ignored - we aggregate if 2+ models  
    embedding_strategy: str,  # Ignored - we aggregate if 2+ models
) -> None:
    """Aggregate results across models based on number of models.
    
    Only stores aggregated results if there are 2 or more models.
    """
    
    # Only aggregate if we have multiple models
    if len(models) < 2:
        logger.debug(f"Only {len(models)} model(s), skipping aggregation")
        return
    
    # Helper function to compute statistics
    def compute_stats(values: List[float], prefix: str) -> None:
        valid_values = [v for v in values if v is not None and not np.isnan(v)]
        if valid_values:
            structure.properties[prefix] = np.mean(valid_values)
            structure.properties[f"{prefix}_std"] = np.std(valid_values) if len(valid_values) > 1 else 0.0
            structure.properties[f"{prefix}_count"] = len(valid_values)
        else:
            structure.properties[prefix] = None
            structure.properties[f"{prefix}_std"] = None
            structure.properties[f"{prefix}_count"] = 0

    # Aggregate energy-based metrics
    # Formation energy
    formation_energies = [
        model_results[model].get("formation_energy") 
        for model in models if model in model_results
    ]
    compute_stats(formation_energies, "formation_energy")
    
    # Energy above hull
    e_above_hulls = [
        model_results[model].get("e_above_hull") 
        for model in models if model in model_results
    ]
    compute_stats(e_above_hulls, "e_above_hull")
    
    # Total energies
    energies = [
        model_results[model].get("energy") 
        for model in models if model in model_results
    ]
    compute_stats(energies, "energy")

    # Aggregate RMSE metrics
    rmse_values = [
        model_results[model].get("relaxation_rmse") 
        for model in models if model in model_results
    ]
    compute_stats(rmse_values, "relaxation_rmse")

    # Aggregate embeddings
    # For graph embeddings, we can average them if they have the same shape
    graph_embeddings = []
    for model in models:
        if model in model_results and model_results[model].get("graph_embedding") is not None:
            emb = model_results[model]["graph_embedding"]
            graph_embeddings.append(emb)
    
    if graph_embeddings:
        try:
            # Check if all embeddings have the same shape
            shapes = [emb.shape for emb in graph_embeddings]
            if len(set(shapes)) == 1:  # All shapes are the same
                # Average graph embeddings
                structure.properties["graph_embedding"] = np.mean(graph_embeddings, axis=0)
                structure.properties["graph_embedding_std"] = np.std(graph_embeddings, axis=0)
                structure.properties["graph_embedding_count"] = len(graph_embeddings)
            else:
                # Different shapes - cannot aggregate directly
                logger.warning(f"Cannot aggregate graph embeddings with different shapes: {shapes}")
                # Store concatenated embedding instead
                structure.properties["graph_embedding"] = np.concatenate(graph_embeddings)
                structure.properties["graph_embedding_count"] = len(graph_embeddings)
                # No std for concatenated embeddings
                structure.properties["graph_embedding_std"] = None
        except Exception as e:
            logger.warning(f"Failed to aggregate graph embeddings: {str(e)}")
            # Fall back to just storing count
            structure.properties["graph_embedding"] = None
            structure.properties["graph_embedding_std"] = None
            structure.properties["graph_embedding_count"] = len(graph_embeddings)


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


# Factory functions for convenience
def create_multi_mlip_stability_preprocessor(
    models: List[str] = None, device: str = "cpu", **kwargs
) -> MultiMLIPStabilityPreprocessor:
    """Factory function to create multi-MLIP stability preprocessor.

    Parameters
    ----------
    models : List[str], optional
        List of models to use. Defaults to ["orb", "mace", "uma"]
    device : str, default="cpu"
        Device to run models on
    **kwargs
        Additional arguments passed to MultiMLIPStabilityPreprocessor

    Returns
    -------
    MultiMLIPStabilityPreprocessor
        Configured preprocessor
    """
    models = models or ["orb", "mace", "uma"]

    # Set device for all models
    model_configs = {}
    for model in models:
        model_configs[model] = {"device": device}

    return MultiMLIPStabilityPreprocessor(
        models=models, model_configs=model_configs, **kwargs
    )
