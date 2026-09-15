"""MACE model calculator implementation with comprehensive e3nn serialization fixes."""

from pymatgen.core.structure import Structure

from lemat_genbench.models.base import (
    BaseMLIPCalculator,
    CalculationResult,
    EmbeddingResult,
    get_energy_above_hull_from_total_energy,
    get_formation_energy_per_atom_from_total_energy,
)
from lemat_genbench.models.mace.embeddings import MACEEmbeddingExtractor
from lemat_genbench.utils.logging import logger

try:
    from mace.calculators import MACECalculator as MACEASECalculator
    from mace.calculators import mace_mp, mace_off

    MACE_AVAILABLE = True
except ImportError:
    MACE_AVAILABLE = False

#: Checkpoint for energies, forces and relaxation with ``model_type="mp"``.  It
#: must be the one that produced ``mace_mp_energy`` in
#: LeMaterial/LeMat-Bulk-MLIP-Hull, or every energy above the ``mace_mp`` hull
#: mixes two energy scales.  MACE-MP-0b3-medium reproduces the published energies
#: to float32 rounding (median 3.5e-7 eV/atom over 3350 structures); every other
#: MACE-MP checkpoint misses by 14-30 meV/atom.  Named explicitly because
#: ``mace_mp()``'s default changed in mace-torch 0.3.10 and is neither.
MACE_MP_ENERGY_MODEL = "medium-0b3"

#: Checkpoint for embeddings with ``model_type="mp"``.  It must be the one that
#: produced ``mace_embeddings`` in LeMaterial/LeMat-GenBench-embeddings, from
#: which the Fréchet distance reference statistics are computed.
#: MACE-MPA-0-medium reproduces them to a relative L2 error of 2e-7; the
#: energy checkpoint above is off by ~150%.
MACE_MP_EMBEDDING_MODEL = "medium-mpa-0"


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
                self._setup_mp_models(device_str, **kwargs)
                logger.info(f"Successfully loaded MACE model: {self.model_type}")
                return
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

    def _setup_mp_models(self, device_str, **kwargs):
        """Load the MACE-MP checkpoints that match LeMat-GenBench's references.

        Energies and embeddings are compared against references built with two
        different checkpoints, so each is computed with its own: see
        :data:`MACE_MP_ENERGY_MODEL` and :data:`MACE_MP_EMBEDDING_MODEL`.
        Passing ``model`` or ``embedding_model`` overrides them, at the cost of
        no longer matching the corresponding reference.
        """
        energy_model = kwargs.pop("model", MACE_MP_ENERGY_MODEL)
        embedding_model = kwargs.pop("embedding_model", MACE_MP_EMBEDDING_MODEL)

        self.ase_calc = mace_mp(model=energy_model, device=device_str, **kwargs)
        if embedding_model == energy_model:
            embedding_calc = self.ase_calc
        else:
            embedding_calc = mace_mp(
                model=embedding_model, device=device_str, **kwargs
            )
        self.embedding_extractor = MACEEmbeddingExtractor(embedding_calc, self.device)
        logger.info(
            f"MACE-MP energies from '{energy_model}', "
            f"embeddings from '{embedding_model}'"
        )

    def _alternative_model_loading(self, device_str, **kwargs):
        """Alternative model loading approach for problematic models."""
        # This method can be used to implement alternative loading strategies
        # if the standard approach fails

        if self.model_type == "mp":
            # Retry in float32, keeping the checkpoints: substituting a different
            # model would silently break the pairing with the hull and the
            # embedding reference.
            kwargs_alt = kwargs.copy()
            kwargs_alt["default_dtype"] = "float32"
            self._setup_mp_models(device_str, **kwargs_alt)
        else:
            raise ValueError("Alternative loading only implemented for MP models")

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

        return get_formation_energy_per_atom_from_total_energy(
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
            total_energy, structure.composition, hull_type=self.hull_type
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
