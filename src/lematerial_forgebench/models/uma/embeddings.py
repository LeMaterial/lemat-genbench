"""UMA embedding extraction utilities."""

from typing import Union

import numpy as np
import torch
from pymatgen.core.structure import Structure

try:
    from fairchem.core.datasets import data_list_collater
    from fairchem.core.datasets.atomic_data import AtomicData

    UMA_AVAILABLE = True
except ImportError:
    UMA_AVAILABLE = False

from lematerial_forgebench.models.base import BaseEmbeddingExtractor


class UMAEmbeddingExtractor(BaseEmbeddingExtractor):
    """Embedding extractor for UMA models."""

    def __init__(self, model, task="omat", device="cpu"):
        self.model = model
        self.device = device
        self.task = task
        self.features = {}
        self.register_forward_hook()

    def register_forward_hook(self):
        def hook(module, input, output):
            self.features["node_embeddings"] = input[1]["node_embedding"][:, 0, :]
            return output

        self.model.module.output_heads.energyandforcehead.register_forward_hook(hook)

    def extract_node_embeddings(
        self, structure: Union[Structure, list[Structure]]
    ) -> Union[np.ndarray, list[np.ndarray]]:
        """Extract per-atom embeddings from UMA model.

        Parameters
        ----------
        structure : Union[Structure, list[Structure]]
            Input structure or list of structures

        Returns
        -------
        Union[np.ndarray, list[np.ndarray]]
            If single structure: Node embeddings with shape (n_atoms, hidden_dim)
            If list of structures: List of node embeddings arrays
        """
        if isinstance(structure, list):
            # Convert all structures to ASE atoms
            atoms_list = [s.to_ase_atoms() for s in structure]

            # Convert all to FAIRChem's AtomicData format
            adata_list = [
                AtomicData.from_ase(atoms, task_name=self.task).to(self.device)
                for atoms in atoms_list
            ]

            # Batch process all structures
            batch = data_list_collater(adata_list, otf_graph=True)

            # Forward pass through model
            self.model(batch)
            node_embeddings = self.features["node_embeddings"]

            # Split batch back into individual structures
            split_sizes = [len(adata.pos) for adata in adata_list]
            node_embeddings_list = torch.split(node_embeddings, split_sizes)

            return [emb.detach().cpu().numpy() for emb in node_embeddings_list]

        atoms = structure.to_ase_atoms()

        # Convert to FAIRChem's AtomicData format
        adata = AtomicData.from_ase(atoms, task_name=self.task).to(self.device)
        batch = data_list_collater([adata], otf_graph=True)

        self.model(batch)
        node_embeddings = self.features["node_embeddings"]

        return node_embeddings.detach().cpu().numpy()

    def extract_graph_embedding_with_pooling(
        self, structure: Structure, pooling_method: str = "mean"
    ) -> np.ndarray:
        """Extract graph embedding using explicit pooling.

        Parameters
        ----------
        structure : Structure
            Input structure
        pooling_method : str
            Pooling method: "mean", "sum", "max"

        Returns
        -------
        np.ndarray
            Graph-level embedding
        """
        atoms = structure.to_ase_atoms()
        adata = AtomicData.from_ase(atoms, task_name=self.task).to(self.device)
        batch = data_list_collater([adata], otf_graph=True)

        self.model(batch)
        node_embeddings = self.features["node_embeddings"]

        # Pool over atoms to get graph representation
        if pooling_method in {"mean", "sum", "max"}:
            graph_emb = torch.scatter_reduce(
                node_embeddings,
                dim=0,
                index=batch.batch,
                reduce=pooling_method,
                include_self=False,
            )
        else:
            raise ValueError(f"Unknown pooling method: {pooling_method}")

        return graph_emb.detach().cpu().numpy().squeeze()


class UMAASECalculator:
    """ASE calculator wrapper for UMA."""

    def __init__(self, predictor, task="omat", device="cpu"):
        self.predictor = predictor
        self.device = device
        self.task = task
        self.implemented_properties = ["energy", "forces"]
        self.results = {}

    def calculate(self, atoms, properties=None, system_changes=None):
        """Calculate properties using UMA."""
        # Convert to FAIRChem format
        adata = AtomicData.from_ase(atoms, task_name=self.task).to(self.device)
        batch = data_list_collater([adata], otf_graph=True)

        # Predict using UMA
        predictions = self.predictor.predict(batch)

        # Store results
        self.results = {}
        if "energy" in predictions:
            self.results["energy"] = predictions["energy"].detach().cpu().numpy().item()
        if "forces" in predictions:
            self.results["forces"] = predictions["forces"].detach().cpu().numpy()
        if "stress" in predictions:
            self.results["stress"] = predictions["stress"].detach().cpu().numpy()

    def get_potential_energy(self, atoms=None):
        if atoms is not None:
            self.calculate(atoms)
        return self.results.get("energy", 0.0)

    def get_forces(self, atoms=None):
        if atoms is not None:
            self.calculate(atoms)
        return self.results.get("forces", np.zeros((len(atoms), 3)))

    def get_stress(self, atoms=None):
        if atoms is not None:
            self.calculate(atoms)
        return self.results.get("stress", np.zeros((3, 3)))
