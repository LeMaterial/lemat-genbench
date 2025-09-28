"""ORB model calculator implementation."""

from pymatgen.core.structure import Structure

from lemat_genbench.models.base import (
    BaseMLIPCalculator,
    CalculationResult,
    EmbeddingResult,
    get_energy_above_hull_from_total_energy,
    get_formation_energy_per_atom_from_total_energy,
)
from lemat_genbench.models.orb.embeddings import ORBEmbeddingExtractor
from lemat_genbench.utils.logging import logger

try:
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.calculator import ORBCalculator as ORBASECalculator

    ORB_AVAILABLE = True
except ImportError:
    ORB_AVAILABLE = False


class ORBCalculator(BaseMLIPCalculator):
    """ORB calculator for energy/force calculations and embedding extraction."""

    def __init__(
        self,
        model_type: str = "orb_v3_conservative_inf_omat",
        device: str = "cpu",
        precision: str = "float32-high",
        **kwargs,
    ):
        if not ORB_AVAILABLE:
            raise ImportError(
                "ORB is not available. You may run uv sync --extra orb to install it."
            )

        self.model_type = model_type
        super().__init__(device=device, precision=precision, **kwargs)

    def _setup_model(self, **kwargs):
        """Initialize the ORB model."""
        try:
            # Load the pretrained model
            model_func = getattr(pretrained, self.model_type)
            self.model = model_func(
                device=self.device,
                precision=self.precision,
                compile=False,  # Avoid compilation issues
            ).eval()

            # Create ASE calculator
            self.ase_calc = ORBASECalculator(self.model, device=self.device)

            # Create embedding extractor
            self.embedding_extractor = ORBEmbeddingExtractor(self.model, self.device)

            logger.info(f"Successfully loaded ORB model: {self.model_type}")

        except Exception as e:
            logger.error(f"Failed to load ORB model {self.model_type}: {str(e)}")
            raise

    def calculate_energy_forces(self, structure: Structure) -> CalculationResult:
        """Calculate energy and forces using ORB.

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

        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()

        return CalculationResult(
            energy=energy, forces=forces, metadata={"model_type": self.model_type}
        )

    def extract_embeddings(self, structure: Structure) -> EmbeddingResult:
        """Extract embeddings using ORB.

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
        """Get ASE calculator for ORB."""
        return self.ase_calc

    def calculate_formation_energy(self, structure: Structure) -> float:
        """Calculate formation energy using ORB.

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

        return get_formation_energy_per_atom_from_total_energy(
            total_energy, structure.composition
        )

    def calculate_energy_above_hull(self, structure: Structure) -> float:
        """Calculate energy above hull using ORB.

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


def create_orb_calculator(
    model_type: str = "orb_v3_conservative_inf_omat", device: str = "cpu", **kwargs
) -> ORBCalculator:
    """Factory function to create ORB calculator.

    Parameters
    ----------
    model_type : str
        ORB model variant to use
    device : str
        Device for computation
    **kwargs
        Additional arguments for the calculator

    Returns
    -------
    ORBCalculator
        Configured ORB calculator
    """
    return ORBCalculator(model_type=model_type, device=device, **kwargs)


# Available ORB model types
AVAILABLE_ORB_MODELS = [
    "orb_v3_conservative_inf_omat",
    "orb_v3_conservative_inf_mpa",
    "orb_v3_direct_inf_mpa",
    "orb_v2_conservative_inf",
    "orb_v2_direct_inf",
]
