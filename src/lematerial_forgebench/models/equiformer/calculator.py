"""Equiformer v2 model calculator implementation."""

import numpy as np
import torch
from pymatgen.core.structure import Structure

try:
    from fairchem.core import OCPCalculator as EquiformerASECalculator

    EQUIFORMER_AVAILABLE = True
except ImportError:
    EQUIFORMER_AVAILABLE = False

from lematerial_forgebench.models.base import (
    BaseMLIPCalculator,
    CalculationResult,
    EmbeddingResult,
    get_energy_above_hull_from_total_energy,
    get_formation_energy_from_total_energy,
)
from lematerial_forgebench.models.equiformer.embeddings import (
    EquiformerEmbeddingExtractor,
)
from lematerial_forgebench.utils.logging import logger


class EquiformerCalculator(BaseMLIPCalculator):
    """Equiformer v2 calculator for energy/force calculations and embedding extraction."""

    def __init__(
        self,
        model_path: str,
        max_neigh: int = 50,
        radius: float = 6.0,
        device: str = "cpu",
        **kwargs,
    ):
        if not EQUIFORMER_AVAILABLE:
            raise ImportError(
                "Equiformer v2 is not available. Please install OCP and equiformer_v2."
            )

        self.model_path = model_path
        self.max_neigh = max_neigh
        self.radius = radius
        super().__init__(device=device, **kwargs)

    def _setup_model(self, **kwargs):
        """Initialize the Equiformer v2 model."""
        try:
            self.ase_calc = EquiformerASECalculator(
                checkpoint_path=self.model_path, cpu=False
            )
            # Create embedding extractor
            self.embedding_extractor = EquiformerEmbeddingExtractor(
                self.ase_calc, self.device
            )

            logger.info(
                f"Successfully loaded Equiformer v2 model from {self.model_path}"
            )

        except Exception as e:
            logger.error(f"Failed to load Equiformer v2 model: {str(e)}")
            raise

    def calculate_energy_forces(self, structure: Structure) -> CalculationResult:
        """Calculate energy and forces using Equiformer v2.

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
            energy=energy, forces=forces, metadata={"model_type": "Equiformer_v2"}
        )

    def extract_embeddings(self, structure: Structure) -> EmbeddingResult:
        """Extract embeddings using Equiformer v2.

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
        """Get ASE calculator for Equiformer v2."""
        return self.ase_calc

    def calculate_formation_energy(self, structure: Structure) -> float:
        """Calculate formation energy using Equiformer v2.

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
        """Calculate energy above hull using Equiformer v2.

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


def create_equiformer_calculator(
    model_path: str,
    max_neigh: int = 50,
    radius: float = 6.0,
    device: str = "cpu",
    **kwargs,
) -> EquiformerCalculator:
    """Factory function to create Equiformer v2 calculator.

    Parameters
    ----------
    model_path : str
        Path to Equiformer v2 checkpoint
    max_neigh : int
        Maximum number of neighbors
    radius : float
        Cutoff radius for neighbor search
    device : str
        Device for computation
    **kwargs
        Additional arguments for the calculator

    Returns
    -------
    EquiformerCalculator
        Configured Equiformer v2 calculator
    """
    return EquiformerCalculator(
        model_path=model_path,
        max_neigh=max_neigh,
        radius=radius,
        device=device,
        **kwargs,
    )
