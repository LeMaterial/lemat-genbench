"""ORB embedding extraction utilities."""

from typing import Union

import numpy as np
import torch
from lemat_genbench.models.base import BaseEmbeddingExtractor
from pymatgen.core.structure import Structure

try:
    from orb_models.forcefield import atomic_system
    from orb_models.forcefield.base import batch_graphs

    ORB_AVAILABLE = True
except ImportError:
    ORB_AVAILABLE = False


class ORBEmbeddingExtractor(BaseEmbeddingExtractor):
    """Embedding extractor for ORB models."""

    def __init__(self, model, device="cpu"):
        if not ORB_AVAILABLE:
            raise ImportError(
                "ORB is not available. Please install it with: uv pip install orb-models"
            )

        super().__init__(model, device)
        self.system_config = model._system_config

    def extract_node_embeddings(
        self, structure: Union[Structure, list[Structure]]
    ) -> Union[np.ndarray, list[np.ndarray]]:
        """Extract per-atom embeddings from ORB model.

        Parameters
        ----------
        structure : Union[Structure, list[Structure]]
            Input structure or list of structures

        Returns
        -------
        Union[np.ndarray, list[np.ndarray]]
            If single structure: Node embeddings with shape (n_atoms, 1024)
            If list of structures: List of node embeddings arrays
        """
        if isinstance(structure, list):
            # Convert all structures to ASE atoms
            atoms_list = [s.to_ase_atoms() for s in structure]

            # Convert all to ORB graph format
            graphs = [
                atomic_system.ase_atoms_to_atom_graphs(atoms, self.system_config)
                for atoms in atoms_list
            ]

            # Move all graphs to device
            batch = batch_graphs(graphs)
            batch = batch.to(self.device)

            # Forward pass to get embeddings for each graph
            out = self.model(batch)
            node_features = out["node_features"].detach()  # Shape: (N_atoms, 1024)
            node_features_list = torch.split(node_features, batch.n_node.tolist())
            node_features_list = [
                node_features.detach().cpu().numpy()
                for node_features in node_features_list
            ]

            return node_features_list

        # Convert to ASE atoms
        atoms = structure.to_ase_atoms()

        # Convert to ORB graph format
        graph = atomic_system.ase_atoms_to_atom_graphs(atoms, self.system_config)
        graph = graph.to(self.device)

        # Forward pass to get embeddings
        out = self.model(graph)
        node_features = out["node_features"]  # Shape: (N_atoms, 1024)

        return node_features.detach().cpu().numpy()

    def extract_graph_embedding_with_learned_pooling(
        self, structure: Structure
    ) -> np.ndarray:
        """Extract graph embedding using ORB's learned global pooling.

        This uses the same pooling layer that ORB uses before its energy head.

        Parameters
        ----------
        structure : Structure
            Input structure

        Returns
        -------
        np.ndarray
            Graph embedding from learned pooling
        """
        import torch_scatter

        atoms = structure.to_ase_atoms()
        graph = atomic_system.ase_atoms_to_atom_graphs(atoms, self.system_config)
        graph = graph.to(self.device)

        # Forward pass through the full model to get pooled representation
        out = self.model(graph)

        # The model should have a global pooling layer before energy prediction
        # This varies by ORB version, so we'll use the manual pooling as fallback
        if "graph_features" in out:
            graph_features = out["graph_features"]
        else:
            # Manual pooling as fallback
            node_features = out["node_features"]
            graph_features = torch_scatter.scatter_mean(
                node_features, graph.batch, dim=0
            )

        return graph_features.detach().cpu().numpy().squeeze()
