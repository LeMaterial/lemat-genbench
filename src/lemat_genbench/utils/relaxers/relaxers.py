"""relaxer implementations using the MLIP system."""

import tempfile

from lemat_genbench.models.registry import get_calculator
from lemat_genbench.utils.logging import logger
from lemat_genbench.utils.relaxers.registry import (
    BaseRelaxer,
    RelaxationResult,
    register_relaxer,
)
from pymatgen.core import Structure
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.vasp.inputs import Incar, Poscar
from pymatgen.io.vasp.sets import MPRelaxSet


class BaseVASPRelaxer(BaseRelaxer):
    """Base class for relaxers that use VASP-like parameters."""

    def get_computed_entry(
        self, structure: Structure, energy: float
    ) -> ComputedStructureEntry:
        """Create a ComputedStructureEntry from a relaxed structure.

        Parameters
        ----------
        structure : Structure
            The relaxed structure.
        energy : float
            The energy of the relaxed structure.

        Returns
        -------
        ComputedStructureEntry
            The computed structure entry with MP2020 corrections applied.
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            b = MPRelaxSet(structure)
            b.write_input(f"{tmpdirname}/", potcar_spec=True)
            poscar = Poscar.from_file(f"{tmpdirname}/POSCAR")
            incar = Incar.from_file(f"{tmpdirname}/INCAR")
            clean_structure = Structure.from_file(f"{tmpdirname}/POSCAR")

        # Get the U values and figure out if we should have run a GGA+U calc
        param = {"hubbards": {}}
        if "LDAUU" in incar:
            param["hubbards"] = dict(zip(poscar.site_symbols, incar["LDAUU"]))
        param["is_hubbard"] = (
            incar.get("LDAU", True) and sum(param["hubbards"].values()) > 0
        )
        if param["is_hubbard"]:
            param["run_type"] = "GGA+U"

        # Make a ComputedStructureEntry without the correction
        cse_d = {
            "structure": clean_structure,
            "energy": energy,
            "correction": 0.0,
            "parameters": param,
        }

        # Apply the MP 2020 correction scheme (anion/+U/etc)
        cse = ComputedStructureEntry.from_dict(cse_d)
        _ = MaterialsProject2020Compatibility(check_potcar=False).process_entries(
            cse,
            clean=True,
        )

        return cse


class MLIPRelaxer(BaseVASPRelaxer):
    """Universal relaxer that uses any MLIP calculator."""

    def __init__(self, calculator, **kwargs):
        """Initialize with a MLIP calculator.

        Parameters
        ----------
        calculator : BaseMLIPCalculator
            Calculator to use for relaxation
        **kwargs
            Additional parameters (fmax, steps, etc.)
        """
        self.calculator = calculator
        self.fmax = kwargs.get("fmax", 0.02)
        self.steps = kwargs.get("steps", 500)

    def relax(self, structure: Structure, relax: bool = True) -> RelaxationResult:
        """Relax a structure using the MLIP calculator.

        Parameters
        ----------
        structure : Structure
            Structure to relax
        relax : bool
            Whether to actually perform relaxation

        Returns
        -------
        RelaxationResult
            Result of the relaxation
        """
        try:
            if not structure.is_valid():
                return RelaxationResult(
                    success=False,
                    message="Invalid crystal structure",
                )

            if relax:
                # Use the calculator's built-in relaxation
                relaxed_structure, calc_result = self.calculator.relax_structure(
                    structure, fmax=self.fmax, steps=self.steps
                )

                return RelaxationResult(
                    success=True,
                    energy=calc_result.energy,
                    structure=relaxed_structure,
                    message=f"Relaxed using {self.calculator.__class__.__name__}",
                )
            else:
                # Just calculate energy without relaxation
                calc_result = self.calculator.calculate_energy_forces(structure)

                return RelaxationResult(
                    success=True,
                    energy=calc_result.energy,
                    structure=structure,
                    message=f"Energy calculated using {self.calculator.__class__.__name__}",
                )

        except Exception as e:
            logger.error(f"MLIP relaxation failed: {str(e)}")
            return RelaxationResult(
                success=False,
                message=str(e),
            )


# Register MLIP-based relaxers using the factory pattern
@register_relaxer("orb")
class ORBRelaxer(MLIPRelaxer):
    """ORB-based relaxer."""

    def __init__(self, **kwargs):
        # Extract ORB-specific parameters
        model_type = kwargs.pop("model_type", "orb_v3_conservative_inf_omat")
        device = kwargs.pop("device", "cpu")
        precision = kwargs.pop("precision", "float32-high")

        # Create ORB calculator
        calculator = get_calculator(
            "orb", model_type=model_type, device=device, precision=precision
        )

        super().__init__(calculator, **kwargs)


@register_relaxer("mace")
class MACERelaxer(MLIPRelaxer):
    """MACE-based relaxer."""

    def __init__(self, **kwargs):
        # Extract MACE-specific parameters
        model_type = kwargs.pop("model_type", "mp")
        device = kwargs.pop("device", "cpu")
        model_path = kwargs.pop("model_path", None)

        # Create MACE calculator
        calculator = get_calculator(
            "mace", model_type=model_type, device=device, model_path=model_path
        )

        super().__init__(calculator, **kwargs)


@register_relaxer("uma")
class UMARelaxer(MLIPRelaxer):
    """UMA-based relaxer."""

    def __init__(self, **kwargs):
        # Extract UMA-specific parameters
        model_name = kwargs.pop("model_name", "uma-s-1")
        task = kwargs.pop("task", "omat")
        device = kwargs.pop("device", "cpu")
        precision = kwargs.pop("precision", "float32")

        # Create UMA calculator
        calculator = get_calculator(
            "uma", model_name=model_name, task=task, device=device, precision=precision
        )

        super().__init__(calculator, **kwargs)


@register_relaxer("equiformer")
class EquiformerRelaxer(MLIPRelaxer):
    """Equiformer-based relaxer."""

    def __init__(self, **kwargs):
        # Extract Equiformer-specific parameters
        model_path = kwargs.pop("model_path")  # Required parameter
        device = kwargs.pop("device", "cpu")

        # Create Equiformer calculator
        calculator = get_calculator("equiformer", model_path=model_path, device=device)

        super().__init__(calculator, **kwargs)
