"""Complete stability preprocessor using the new modular MLIP system."""

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
from func_timeout import FunctionTimedOut, func_timeout
from pymatgen.core import Structure

from lematerial_forgebench.models.registry import get_calculator, list_available_models
from lematerial_forgebench.preprocess.base import BasePreprocessor, PreprocessorConfig
from lematerial_forgebench.utils.logging import logger


@dataclass
class UniversalStabilityPreprocessorConfig(PreprocessorConfig):
    """Configuration for the Universal Stability Preprocessor.

    Parameters
    ----------
    model_name : str
        Name of the MLIP to use ("orb", "mace", "equiformer", "uma")
        Use :func:`~lematerial_forgebench.models.registry.get_available_models`
        to see which models are available.
    model_config : dict
        Configuration for the specific model
    relax_structures : bool
        Whether to relax structures during preprocessing
    relaxation_config : dict
        Configuration for structure relaxation
    calculate_formation_energy : bool
        Whether to calculate formation energy
    calculate_energy_above_hull : bool
        Whether to calculate energy above hull
    """

    model_name: str = "orb"
    timeout: int = 60
    model_config: Dict[str, Any] = field(default_factory=dict)
    relax_structures: bool = True
    relaxation_config: Dict[str, Any] = field(default_factory=dict)
    calculate_formation_energy: bool = True
    calculate_energy_above_hull: bool = True
    extract_embeddings: bool = True


class UniversalStabilityPreprocessor(BasePreprocessor):
    """Universal stability preprocessor that works with any MLIP.

    This preprocessor can use any of the available MLIPs (ORB, MACE, Equiformer, UMA)
    to calculate formation energy, energy above hull, and optionally relax structures.

    Parameters
    ----------
    model_type : str
        Type of MLIP to use
    model_config : dict
        Configuration for the specific model
    relax_structures : bool
        Whether to relax structures
    relaxation_config : dict
        Configuration for relaxation (fmax, steps, etc.)
    calculate_formation_energy : bool
        Whether to calculate formation energy
    calculate_energy_above_hull : bool
        Whether to calculate energy above hull
    extract_embeddings : bool
        Whether to extract embeddings
    name : str, optional
        Custom name for the preprocessor
    description : str, optional
        Description of what the preprocessor does
    n_jobs : int, default=1
        Number of parallel jobs to run
    """

    def __init__(
        self,
        model_name: str = "orb",
        timeout: int = 60, 
        model_config: Dict[str, Any] = None,
        relax_structures: bool = True,
        relaxation_config: Dict[str, Any] = None,
        calculate_formation_energy: bool = True,
        calculate_energy_above_hull: bool = True,
        extract_embeddings: bool = True,
        name: str | None = None,
        description: str | None = None,
        n_jobs: int = 1,
    ):
        super().__init__(
            name=name or f"UniversalStabilityPreprocessor_{model_name}",
            description=description or f"Stability preprocessing using {model_name}",
            n_jobs=n_jobs,
        )

        self.config = UniversalStabilityPreprocessorConfig(
            name=self.config.name,
            description=self.config.description,
            n_jobs=self.config.n_jobs,
            model_name=model_name,
            timeout=timeout,
            model_config=model_config or {},
            relax_structures=relax_structures,
            relaxation_config=relaxation_config or {"fmax": 0.02, "steps": 500},
            calculate_formation_energy=calculate_formation_energy,
            calculate_energy_above_hull=calculate_energy_above_hull,
            extract_embeddings=extract_embeddings,
        )

        # Create calculator using the registry
        try:
            self.calculator = get_calculator(model_name, **self.config.model_config)
        except ValueError as e:
            available_models = list_available_models()
            raise ValueError(
                f"Model type '{model_name}' not supported. "
                f"Available models: {available_models}"
            ) from e

    def _get_process_attributes(self) -> dict[str, Any]:
        """Get the attributes for the process_structure method."""
        return {
            "calculator": self.calculator,
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
        calculator,
        timeout: int, 
        relax_structures: bool,
        relaxation_config: Dict[str, Any],
        calculate_formation_energy: bool,
        calculate_energy_above_hull: bool,
        extract_embeddings: bool,
    ) -> Structure:
        """Process a single structure by calculating energies and optionally relaxing.

        Parameters
        ----------
        structure : Structure
            A pymatgen Structure object to process
        calculator : BaseMLIPCalculator
            Calculator to use for computations
        relax_structures : bool
            Whether to relax the structure
        relaxation_config : dict
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
            The processed Structure with calculated properties

        Raises
        ------
        Exception
            If computation fails
        """


        try:
            result = func_timeout(
                timeout, _process_structure_action, [
                                                structure, 
                                                calculator, 
                                                relax_structures,
                                                relaxation_config, 
                                                calculate_formation_energy, 
                                                calculate_energy_above_hull, 
                                                extract_embeddings,
                                                ]
            )
            return result
        except FunctionTimedOut:

            print("Function timed out and was skipped")
            return structure

def _process_structure_action(        
    structure: Structure,
    calculator,
    relax_structures: bool,
    relaxation_config: Dict[str, Any],
    calculate_formation_energy: bool,
    calculate_energy_above_hull: bool,
    extract_embeddings: bool,
    ) -> Structure:


    try:
        # Store model information
        structure.properties["mlip_model"] = calculator.__class__.__name__
        structure.properties["model_config"] = getattr(
            calculator, "model_type", "unknown"
        )

        # Calculate basic energy and forces for original structure
        energy_result = calculator.calculate_energy_forces(structure)
        structure.properties["energy"] = energy_result.energy
        structure.properties["forces"] = energy_result.forces

        # Calculate formation energy if requested
        if calculate_formation_energy:
            try:
                formation_energy = calculator.calculate_formation_energy(structure)
                structure.properties["formation_energy"] = formation_energy
                logger.debug(
                    f"Computed formation_energy: {formation_energy:.3f} eV/atom for {structure.formula}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to compute formation_energy for {structure.formula}: {str(e)}"
                )
                structure.properties["formation_energy"] = None

        # Calculate energy above hull if requested
        if calculate_energy_above_hull:
            try:
                e_above_hull = calculator.calculate_energy_above_hull(structure)
                structure.properties["e_above_hull"] = e_above_hull
                logger.debug(
                    f"Computed e_above_hull: {e_above_hull:.3f} eV/atom for {structure.formula}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to compute e_above_hull for {structure.formula}: {str(e)}"
                )
                structure.properties["e_above_hull"] = None

        # Extract embeddings if requested
        if extract_embeddings:
            embeddings = calculator.extract_embeddings(structure)
            structure.properties["node_embeddings"] = embeddings.node_embeddings
            structure.properties["graph_embedding"] = embeddings.graph_embedding

        # Optionally relax the structure
        if relax_structures:
            try:
                relaxed_structure, relaxation_result = calculator.relax_structure(
                    structure, **relaxation_config
                )

                # Calculate RMSE between original and relaxed positions
                rmse = _calculate_rmse(structure, relaxed_structure)

                # Store relaxed structure and properties
                structure.properties["relaxed_structure"] = relaxed_structure
                structure.properties["relaxation_rmse"] = rmse
                structure.properties["relaxation_energy"] = relaxation_result.energy

                if (
                    relaxation_result.metadata
                    and "relaxation_steps" in relaxation_result.metadata
                ):
                    structure.properties["relaxation_steps"] = (
                        relaxation_result.metadata["relaxation_steps"]
                    )

                # Calculate properties for relaxed structure if requested
                if calculate_formation_energy:
                    try:
                        relaxed_formation_energy = (
                            calculator.calculate_formation_energy(relaxed_structure)
                        )
                        structure.properties["relaxed_formation_energy"] = (
                            relaxed_formation_energy
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to compute formation_energy for relaxed {relaxed_structure.formula}: {str(e)}"
                        )
                        structure.properties["relaxed_formation_energy"] = None

                if calculate_energy_above_hull:
                    try:
                        relaxed_e_above_hull = (
                            calculator.calculate_energy_above_hull(
                                relaxed_structure
                            )
                        )
                        structure.properties["relaxed_e_above_hull"] = (
                            relaxed_e_above_hull
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to compute e_above_hull for relaxed {relaxed_structure.formula}: {str(e)}"
                        )
                        structure.properties["relaxed_e_above_hull"] = None

                logger.debug(
                    f"Relaxed structure: RMSE: {rmse:.3f} Å, "
                    f"energy: {relaxation_result.energy:.3f} eV for {structure.formula}"
                )

            except Exception as e:
                logger.warning(
                    f"Failed to relax structure {structure.formula}: {str(e)}"
                )
                # Continue without relaxation
                structure.properties["relaxation_failed"] = True
                structure.properties["relaxation_error"] = str(e)

        return structure

    except Exception as e:
        logger.error(f"Failed to process structure {structure.formula}: {str(e)}")
        raise


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
