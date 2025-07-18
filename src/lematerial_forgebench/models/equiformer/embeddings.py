"""Equiformer v2 embedding extraction utilities."""

import numpy as np
import torch
from fairchem.core.datasets import data_list_collater
from pymatgen.core.structure import Structure

from lematerial_forgebench.models.base import BaseEmbeddingExtractor


class EquiformerEmbeddingExtractor(BaseEmbeddingExtractor):
    """Embedding extractor for Equiformer v2 models."""

    def __init__(self, ase_calc, device="cpu"):
        self.ase_calc = ase_calc
        self.model = self.ase_calc.trainer.model.to(device)
        self.device = device

    def extract_node_embeddings(self, structure: Structure) -> np.ndarray:
        """Extract per-atom embeddings from Equiformer v2.

        Parameters
        ----------
        structure : Structure
            Input structure

        Returns
        -------
        np.ndarray
            Node embeddings with shape (n_atoms, (1+L_max)^2, C)
            Flattened to (n_atoms, (1+L_max)^2 * C)
        """
        atoms = structure.to_ase_atoms()
        data = self.ase_calc.a2g.convert(atoms)
        batch = data_list_collater([data], otf_graph=True)
        batch = batch.to(self.device)

        with torch.no_grad():
            out = self.model(batch, return_embeddings=True)
            node_embeddings = out["node_embeddings"]

        node_irreps = out["node_embeddings"]  # Shape: (N, (1+L_max)^2, C)

        # Flatten the irreps dimension for easier use
        n_atoms, irreps_dim, channels = node_irreps.shape
        node_embeddings = node_irreps.view(n_atoms, -1)  # (N, irreps_dim * C)

        return node_embeddings.cpu().numpy()
