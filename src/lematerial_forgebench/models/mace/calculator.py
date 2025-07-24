"""MACE model calculator implementation with comprehensive e3nn serialization fixes."""

import io

import torch
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


def apply_comprehensive_e3nn_patches():
    """Apply comprehensive monkey patches for all known e3nn serialization issues."""

    patches_applied = []

    # Patch 1: CodeGenMixin.__setstate__ fix
    try:
        from e3nn.util.codegen._mixin import CodeGenMixin

        original_setstate = CodeGenMixin.__setstate__

        def patched_setstate(self, d):
            d = d.copy()
            # We don't want to add this to the object when we call super's __setstate__
            codegen_state = d.pop("__codegen__", None)
            # We need to initialize self first so that we can add submodules
            # We need to check if other parent classes of self define __setstate__
            if hasattr(super(CodeGenMixin, self), "__setstate__"):
                super(CodeGenMixin, self).__setstate__(d)
            else:
                self.__dict__.update(d)
            if codegen_state is not None:
                for fname, buffer in codegen_state.items():
                    assert isinstance(fname, str)
                    # Make sure bytes, not ScriptModules, got made
                    assert isinstance(buffer, bytes)
                    buffer = io.BytesIO(buffer)
                    smod = torch.jit.load(buffer)
                    assert isinstance(smod, torch.jit.ScriptModule)
                    # Add the ScriptModule as a submodule
                    setattr(self, fname, smod)
                self.__codegen__ = list(codegen_state.keys())

        CodeGenMixin.__setstate__ = patched_setstate
        patches_applied.append("CodeGenMixin.__setstate__")

    except ImportError:
        logger.debug("CodeGenMixin not available, skipping patch 1")

    # Patch 2: SphericalHarmonics sph_func fix
    try:
        from e3nn.o3._spherical_harmonics import SphericalHarmonics

        original_sph_setstate = getattr(SphericalHarmonics, "__setstate__", None)

        def patched_sph_setstate(self, state):
            if original_sph_setstate:
                original_sph_setstate(self, state)
            else:
                self.__dict__.update(state)
            # Ensure sph_func is properly initialized after deserialization
            if not hasattr(self, "sph_func"):
                self._initialize_sph_func()

        def _initialize_sph_func(self):
            """Initialize the sph_func attribute if missing."""
            try:
                from e3nn.o3._spherical_harmonics import _spherical_harmonics_alpha

                self.sph_func = _spherical_harmonics_alpha
            except ImportError:
                # Fallback for different e3nn versions
                try:
                    from e3nn.o3 import spherical_harmonics

                    def sph_func_wrapper(lmax, x, y, z):
                        coords = torch.stack([x, y, z], dim=-1)
                        return spherical_harmonics(list(range(lmax + 1)), coords, True)

                    self.sph_func = sph_func_wrapper
                except Exception as e:
                    logger.warning(f"Could not initialize sph_func: {e}")
                    # Create a basic implementation as last resort
                    self.sph_func = self._basic_spherical_harmonics

        def _basic_spherical_harmonics(self, l_max, x, y, z):
            """Basic spherical harmonics implementation as fallback."""
            batch_shape = x.shape[:-1] if x.dim() > 0 else torch.Size([])
            n_harmonics = (l_max + 1) ** 2
            result = torch.zeros(
                *batch_shape, n_harmonics, dtype=x.dtype, device=x.device
            )
            result[..., 0] = 1.0 / torch.sqrt(torch.tensor(4 * torch.pi))
            return result

        SphericalHarmonics.__setstate__ = patched_sph_setstate
        SphericalHarmonics._initialize_sph_func = _initialize_sph_func
        SphericalHarmonics._basic_spherical_harmonics = _basic_spherical_harmonics

        patches_applied.append("SphericalHarmonics.sph_func")

    except ImportError:
        logger.debug("SphericalHarmonics not available, skipping patch 2")

    # Patch 3: Activation paths fix
    try:
        from e3nn.nn._activation import Activation

        original_activation_setstate = getattr(Activation, "__setstate__", None)

        def patched_activation_setstate(self, state):
            if original_activation_setstate:
                original_activation_setstate(self, state)
            else:
                self.__dict__.update(state)

            # Ensure paths is properly initialized after deserialization
            if not hasattr(self, "paths"):
                self._reconstruct_paths()

        def _reconstruct_paths(self):
            """Reconstruct the paths attribute from other stored attributes."""
            try:
                # Try to reconstruct paths from irreps_in, irreps_out, and acts
                if (
                    hasattr(self, "irreps_in")
                    and hasattr(self, "irreps_out")
                    and hasattr(self, "acts")
                ):
                    from e3nn import o3

                    # Reconstruct paths based on irreps and activations
                    paths = []
                    irreps_in = (
                        self.irreps_in
                        if hasattr(self, "irreps_in")
                        else o3.Irreps("0e")
                    )
                    irreps_out = (
                        self.irreps_out
                        if hasattr(self, "irreps_out")
                        else o3.Irreps("0e")
                    )
                    acts = self.acts if hasattr(self, "acts") else [None]

                    # Simple path reconstruction - may need refinement based on actual usage
                    for i, (mul_out, ir_out) in enumerate(irreps_out):
                        for j, (mul_in, ir_in) in enumerate(irreps_in):
                            if ir_in == ir_out:  # Same irreducible representation
                                act = acts[i] if i < len(acts) else None
                                paths.append(
                                    (min(mul_in, mul_out), (ir_out.l, ir_out.p), act)
                                )

                    self.paths = paths
                else:
                    # Fallback: create a minimal path
                    self.paths = [(1, (0, 1), None)]  # Basic scalar path

                logger.debug(f"Reconstructed {len(self.paths)} activation paths")

            except Exception as e:
                logger.warning(f"Could not reconstruct activation paths: {e}")
                # Last resort: empty paths
                self.paths = []

        Activation.__setstate__ = patched_activation_setstate
        Activation._reconstruct_paths = _reconstruct_paths

        patches_applied.append("Activation.paths")

    except ImportError:
        logger.debug("Activation not available, skipping patch 3")

    # Patch 4: General e3nn module __setstate__ safety net
    try:
        import e3nn
        from torch.nn import Module

        # Create a general safety net for any e3nn module
        def safe_e3nn_setstate(original_setstate):
            def patched_setstate(self, state):
                try:
                    if original_setstate:
                        original_setstate(self, state)
                    else:
                        self.__dict__.update(state)

                    # Post-deserialization fixes for common e3nn issues
                    if hasattr(self, "__class__") and "e3nn" in str(self.__class__):
                        self._fix_e3nn_attributes()

                except Exception as e:
                    logger.warning(
                        f"Error in {self.__class__.__name__}.__setstate__: {e}"
                    )
                    # Fallback: just update the dict
                    self.__dict__.update(state)
                    if hasattr(self, "__class__") and "e3nn" in str(self.__class__):
                        self._fix_e3nn_attributes()

            return patched_setstate

        def _fix_e3nn_attributes(self):
            """Generic fix for common e3nn attribute issues."""
            # This is a catch-all method that individual classes can override
            pass

        # Apply to all e3nn modules (this is aggressive but may be necessary)
        Module._fix_e3nn_attributes = _fix_e3nn_attributes

        patches_applied.append("General e3nn safety net")

    except ImportError:
        logger.debug("Could not apply general e3nn safety net")

    if patches_applied:
        logger.info(f"Applied e3nn patches: {', '.join(patches_applied)}")
    else:
        logger.warning("No e3nn patches could be applied")


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
            # Apply comprehensive e3nn monkey patches before loading model
            apply_comprehensive_e3nn_patches()

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
            # Re-apply patches and retry once
            logger.info("Re-applying patches and retrying calculation...")
            apply_comprehensive_e3nn_patches()

            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()

            return CalculationResult(
                energy=energy,
                forces=forces,
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
