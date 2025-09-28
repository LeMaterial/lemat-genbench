"""Base classes for ML interatomic potentials and calculators.

This module provides the foundation for integrating various MLIPs
into the benchmark framework.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Literal, Tuple, Union

import numpy as np
import torch
from ase import Atoms
from pymatgen.core.structure import Structure

from lemat_genbench.preprocess.reference_energies import (
    get_energy_above_hull,
    get_formation_energy_per_atom_from_composition_energy,
)


@dataclass
class EmbeddingResult:
    """Result container for embedding calculations.

    Parameters
    ----------
    node_embeddings : np.ndarray
        Per-atom embeddings with shape (n_atoms, embedding_dim)
    graph_embedding : np.ndarray
        Graph-level embedding with shape (embedding_dim,)
    metadata : dict
        Additional metadata about the embedding calculation
    """

    node_embeddings: np.ndarray
    graph_embedding: np.ndarray
    metadata: Dict[str, Any]


@dataclass
class CalculationResult:
    """Result container for energy and force calculations.

    Parameters
    ----------
    energy : float
        Total energy in eV
    forces : np.ndarray
        Forces on atoms in eV/Å with shape (n_atoms, 3)
    stress : np.ndarray | None
        Stress tensor in eV/Å³ with shape (3, 3) or None if not available
    metadata : dict
        Additional calculation metadata
    """

    energy: float
    forces: np.ndarray
    stress: np.ndarray | None = None
    metadata: Dict[str, Any] = None


class BaseMLIPCalculator(ABC):
    """Base class for all MLIP calculators.

    This provides a unified interface for different ML interatomic potentials
    to perform energy/force calculations and extract embeddings.
    """

    def __init__(
        self,
        device: Union[str, torch.device] = "cpu",
        precision: str = "float32",
        hull_type: Literal["dft", "uma", "orb", "mace_mp", "mace_omat"] = "dft",
        **kwargs,
    ):
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self.precision = precision
        self.model = None
        self.hull_type = hull_type
        self._setup_model(**kwargs)

    @abstractmethod
    def _setup_model(self, **kwargs) -> None:
        """Initialize the specific model. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def calculate_energy_forces(self, structure: Structure) -> CalculationResult:
        """Calculate energy and forces for a structure.

        Parameters
        ----------
        structure : Structure
            Pymatgen Structure object

        Returns
        -------
        CalculationResult
            Energy, forces, and metadata
        """
        pass

    @abstractmethod
    def extract_embeddings(
        self, structure: Union[Structure, list[Structure]]
    ) -> Union[EmbeddingResult, list[EmbeddingResult]]:
        """Extract node and graph embeddings from structure(s).

        This method supports both single structure and batched processing of multiple structures.
        When a single structure is provided, returns a single EmbeddingResult.
        When a list of structures is provided, returns a list of EmbeddingResults.

        Parameters
        ----------
        structure : Union[Structure, list[Structure]]
            Either a single Pymatgen Structure object or a list of Structure objects

        Returns
        -------
        Union[EmbeddingResult, list[EmbeddingResult]]
            If input is a single structure: EmbeddingResult with node embeddings,
            graph embedding, and metadata.
            If input is a list: List of EmbeddingResults, one for each input structure.
        """
        pass

    def relax_structure(
        self, structure: Structure, fmax: float = 0.02, steps: int = 50
    ) -> Tuple[Structure, CalculationResult]:
        """Relax a structure using the MLIP.

        Parameters
        ----------
        structure : Structure
            Initial structure to relax
        fmax : float
            Force convergence criterion in eV/Å
        steps : int
            Maximum optimization steps

        Returns
        -------
        Tuple[Structure, CalculationResult]
            Relaxed structure and final calculation result
        """
        from ase.filters import FrechetCellFilter

        # from ase.optimize import FIRE
        from ase.optimize import BFGSLineSearch
        from pymatgen.io.ase import AseAtomsAdaptor

        # Convert to ASE atoms
        atoms = structure.to_ase_atoms()

        # Set up calculator
        calc = self._get_ase_calculator()
        atoms.calc = calc

        # Relax structure
        # dyn = FIRE(FrechetCellFilter(atoms), logfile=None)
        dyn = BFGSLineSearch(FrechetCellFilter(atoms), logfile=None)
        dyn.run(fmax=fmax, steps=steps)

        # Get final results
        final_energy = atoms.get_potential_energy()
        final_forces = atoms.get_forces()

        # Convert back to pymatgen
        final_structure = AseAtomsAdaptor.get_structure(atoms)

        result = CalculationResult(
            energy=final_energy,
            forces=final_forces,
            metadata={"relaxation_steps": dyn.get_number_of_steps()},
        )

        return final_structure, result

    @abstractmethod
    def _get_ase_calculator(self):
        """Get ASE calculator for this MLIP. Must be implemented by subclasses."""
        pass

    def _structure_to_atoms(self, structure: Structure) -> Atoms:
        """Convert pymatgen Structure to ASE Atoms."""
        return structure.to_ase_atoms()

    @staticmethod
    def _aggregate_node_embeddings(
        node_embeddings: np.ndarray, method: str = "mean"
    ) -> np.ndarray:
        """Aggregate node embeddings to create graph-level embedding.

        Parameters
        ----------
        node_embeddings : np.ndarray
            Node embeddings with shape (n_atoms, embedding_dim)
        method : str
            Aggregation method: "mean", "sum", "max", "min"

        Returns
        -------
        np.ndarray
            Graph-level embedding with shape (embedding_dim,)
        """
        if method == "mean":
            return np.mean(node_embeddings, axis=0)
        elif method == "sum":
            return np.sum(node_embeddings, axis=0)
        elif method == "max":
            return np.max(node_embeddings, axis=0)
        elif method == "min":
            return np.min(node_embeddings, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")


class BaseEmbeddingExtractor(ABC):
    """Base class for embedding extraction from MLIPs."""

    def __init__(self, model, device: Union[str, torch.device] = "cpu"):
        self.model = model
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )

    @abstractmethod
    def extract_node_embeddings(
        self, structure: Union[Structure, list[Structure]]
    ) -> Union[np.ndarray, list[np.ndarray]]:
        """Extract per-atom embeddings.

        Parameters
        ----------
        structure : Union[Structure, list[Structure]]
            Input structure or list of structures

        Returns
        -------
        Union[np.ndarray, list[np.ndarray]]
            If single structure: Node embeddings with shape (n_atoms, embedding_dim)
            If list of structures: List of node embeddings arrays
        """
        pass

    def extract_graph_embedding(
        self, structure: Union[Structure, list[Structure]], aggregation: str = "mean"
    ) -> Union[np.ndarray, list[np.ndarray]]:
        """Extract graph-level embedding by aggregating node embeddings.

        Parameters
        ----------
        structure : Union[Structure, list[Structure]]
            Input structure or list of structures
        aggregation : str
            Aggregation method for node embeddings

        Returns
        -------
        Union[np.ndarray, list[np.ndarray]]
            If single structure: Graph-level embedding
            If list of structures: List of graph-level embeddings
        """
        node_embs = self.extract_node_embeddings(structure)

        if isinstance(structure, list):
            return [
                BaseMLIPCalculator._aggregate_node_embeddings(emb, aggregation)
                for emb in node_embs
            ]
        return BaseMLIPCalculator._aggregate_node_embeddings(node_embs, aggregation)

    def extract_embeddings(
        self, structure: Union[Structure, list[Structure]], aggregation: str = "mean"
    ) -> Union[EmbeddingResult, list[EmbeddingResult]]:
        """Extract both node and graph embeddings.

        Parameters
        ----------
        structure : Union[Structure, list[Structure]]
            Input structure or list of structures
        aggregation : str
            Aggregation method for graph embedding

        Returns
        -------
        Union[EmbeddingResult, list[EmbeddingResult]]
            If single structure: EmbeddingResult with node and graph embeddings
            If list of structures: List of EmbeddingResults
        """
        if isinstance(structure, list):
            node_embs = self.extract_node_embeddings(structure)
            graph_embs = [
                BaseMLIPCalculator._aggregate_node_embeddings(emb, aggregation)
                for emb in node_embs
            ]

            return [
                EmbeddingResult(
                    node_embeddings=node_emb,
                    graph_embedding=graph_emb,
                    metadata={
                        "aggregation_method": aggregation,
                        "embedding_dim": node_emb.shape[1],
                        "n_atoms": node_emb.shape[0],
                    },
                )
                for node_emb, graph_emb in zip(node_embs, graph_embs)
            ]

        node_embs = self.extract_node_embeddings(structure)
        graph_emb = BaseMLIPCalculator._aggregate_node_embeddings(
            node_embs, aggregation
        )

        return EmbeddingResult(
            node_embeddings=node_embs,
            graph_embedding=graph_emb,
            metadata={
                "aggregation_method": aggregation,
                "embedding_dim": node_embs.shape[1],
                "n_atoms": node_embs.shape[0],
            },
        )


def get_formation_energy_per_atom_from_total_energy(
    total_energy: float, composition, functional: str = "pbe"
) -> float:
    """Calculate formation energy per atom from total energy and composition.

    This uses the same reference energies as the current codebase.

    Parameters
    ----------
    total_energy : float
        Total energy in eV
    composition : Composition
        Pymatgen composition object
    functional : str
        DFT functional used for reference energies

    Returns
    -------
    float
        Formation energy per atom in eV/atom
    """

    return get_formation_energy_per_atom_from_composition_energy(
        total_energy, composition, functional
    )


def get_energy_above_hull_from_total_energy(
    total_energy: float,
    composition,
    hull_type: Literal["dft", "uma", "orb", "mace_mp", "mace_omat"] = "dft",
) -> float:
    """Calculate energy above hull from total energy and composition.

    Parameters
    ----------
    total_energy : float
        Total energy in eV
    composition : Composition
        Pymatgen composition object

    Returns
    -------
    float
        Energy above hull in eV/atom
    """

    return get_energy_above_hull(total_energy, composition, hull_type=hull_type)
