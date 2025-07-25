"""MACE model calculator implementation with comprehensive e3nn serialization fixes."""

from pymatgen.core.structure import Structure

from lematerial_forgebench.models.base import (
    BaseMLIPCalculator,
    CalculationResult,
    EmbeddingResult,
    get_energy_above_hull_from_total_energy,
    get_formation_energy_from_total_energy,
)
from lematerial_forgebench.models.mace.embeddings import MACEEmbeddingExtractor
from lematerial_forgebench.utils.logging import logger

try:
    from mace.calculators import MACECalculator as MACEASECalculator
    from mace.calculators import mace_mp, mace_off

    MACE_AVAILABLE = True
except ImportError:
    MACE_AVAILABLE = False


class MACECalculator(BaseMLIPCalculator):
    """MACE calculator for energy/force calculations and embedding extraction."""

    def __init__(
        self,
        model_type: str = "mp",  # "mp" for Materials Project, "off" for off-the-shelf
        model_path: str = None,  # Path to custom model
        device: str = "cpu",
        **kwargs,
    ):
        if not MACE_AVAILABLE:
            raise ImportError(
                "MACE is not available. Please install it with: pip install mace-torch"
            )

        self.model_type = model_type
        self.model_path = model_path
        super().__init__(device=device, **kwargs)

    def _setup_model(self, **kwargs):
        """Initialize the MACE model."""
        try:
            # Convert torch.device back to string for MACE compatibility
            device_str = (
                str(self.device) if hasattr(self.device, "type") else self.device
            )

            # Force disable weights_only loading to avoid serialization issues
            import os

            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

            if self.model_type == "mp":
                # Materials Project foundation model
                self.ase_calc = mace_mp(device=device_str, **kwargs)
            elif self.model_type == "off":
                # Off-the-shelf models
                self.ase_calc = mace_off(device=device_str, **kwargs)
            elif self.model_path:
                # Custom model from file
                self.ase_calc = MACEASECalculator(
                    model_paths=self.model_path, device=device_str, **kwargs
                )
            else:
                raise ValueError(
                    "Must specify either model_type ('mp' or 'off') or model_path"
                )

            # Create embedding extractor
            self.embedding_extractor = MACEEmbeddingExtractor(
                self.ase_calc, self.device
            )

            logger.info(f"Successfully loaded MACE model: {self.model_type}")

        except Exception as e:
            logger.error(f"Failed to load MACE model: {str(e)}")
            # Try alternative loading approach
            logger.info("Attempting alternative model loading...")
            try:
                self._alternative_model_loading(device_str, **kwargs)
            except Exception as e2:
                logger.error(f"Alternative loading also failed: {str(e2)}")
                raise e

    def _alternative_model_loading(self, device_str, **kwargs):
        """Alternative model loading approach for problematic models."""
        # This method can be used to implement alternative loading strategies
        # if the standard approach fails

        if self.model_type == "mp":
            # Try loading with different parameters
            kwargs_alt = kwargs.copy()
            kwargs_alt.update(
                {
                    "model": "small",  # Try smaller model
                    "default_dtype": "float32",
                }
            )
            self.ase_calc = mace_mp(device=device_str, **kwargs_alt)
        else:
            raise ValueError("Alternative loading only implemented for MP models")

        # Create embedding extractor
        self.embedding_extractor = MACEEmbeddingExtractor(self.ase_calc, self.device)

        logger.info("Successfully loaded MACE model using alternative approach")

    def calculate_energy_forces(self, structure: Structure) -> CalculationResult:
        """Calculate energy and forces using MACE.

        Parameters
        ----------
        structure : Structure
            Input structure

        Returns
        -------
        CalculationResult
            Energy, forces, and metadata
        """
        atoms = self._structure_to_atoms(structure)
        atoms.calc = self.ase_calc

        try:
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()

            # Try to get stress if available
            stress = None
            try:
                stress = atoms.get_stress()
            except Exception:
                pass

            return CalculationResult(
                energy=energy,
                forces=forces,
                stress=stress,
                metadata={"model_type": f"MACE-{self.model_type}"},
            )

        except Exception as e:
            logger.error(f"MACE calculation failed: {str(e)}")

            return CalculationResult(
                energy=None,
                forces=None,
                stress=None,
                metadata={"model_type": f"MACE-{self.model_type}"},
            )

    def extract_embeddings(self, structure: Structure) -> EmbeddingResult:
        """Extract embeddings using MACE.

        Parameters
        ----------
        structure : Structure
            Input structure

        Returns
        -------
        EmbeddingResult
            Node and graph embeddings
        """
        return self.embedding_extractor.extract_embeddings(structure)

    def _get_ase_calculator(self):
        """Get ASE calculator for MACE."""
        return self.ase_calc

    def calculate_formation_energy(self, structure: Structure) -> float:
        """Calculate formation energy using MACE.

        Parameters
        ----------
        structure : Structure
            Input structure

        Returns
        -------
        float
            Formation energy in eV/atom
        """
        result = self.calculate_energy_forces(structure)
        total_energy = result.energy

        return get_formation_energy_from_total_energy(
            total_energy, structure.composition
        )

    def calculate_energy_above_hull(self, structure: Structure) -> float:
        """Calculate energy above hull using MACE.

        Parameters
        ----------
        structure : Structure
            Input structure

        Returns
        -------
        float
            Energy above hull in eV/atom
        """
        result = self.calculate_energy_forces(structure)
        total_energy = result.energy

        return get_energy_above_hull_from_total_energy(
            total_energy, structure.composition
        )


def create_mace_calculator(
    model_type: str = "mp", model_path: str = None, device: str = "cpu", **kwargs
) -> MACECalculator:
    """Factory function to create MACE calculator.

    Parameters
    ----------
    model_type : str
        MACE model type ("mp", "off")
    model_path : str
        Path to custom MACE model
    device : str
        Device for computation
    **kwargs
        Additional arguments for the calculator

    Returns
    -------
    MACECalculator
        Configured MACE calculator
    """
    return MACECalculator(
        model_type=model_type, model_path=model_path, device=device, **kwargs
    )


# Available MACE model types
AVAILABLE_MACE_MODELS = [
    "mp",  # Materials Project foundation model
    "off",  # Off-the-shelf models
]
