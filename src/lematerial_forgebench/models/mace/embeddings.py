"""MACE embedding extraction utilities."""

import numpy as np
from pymatgen.core.structure import Structure

from lematerial_forgebench.models.base import BaseEmbeddingExtractor


class MACEEmbeddingExtractor(BaseEmbeddingExtractor):
    """Embedding extractor for MACE models."""

    def __init__(self, calculator, device="cpu"):
        """Initialize with a MACE calculator."""
        self.calculator = calculator
        self.device = device

    def extract_node_embeddings(self, structure: Structure) -> np.ndarray:
        """Extract per-atom embeddings from MACE model.

        Parameters
        ----------
        structure : Structure
            Input structure

        Returns
        -------
        np.ndarray
            Node embeddings with shape (n_atoms, descriptor_dim)
        """
        atoms = structure.to_ase_atoms()

        # Use MACE's built-in descriptor extraction
        descriptors = self.calculator.get_descriptors(
            atoms,
            invariants_only=False,  # Keep equivariant parts
            num_layers=None,  # Use all layers
        )

        return descriptors
