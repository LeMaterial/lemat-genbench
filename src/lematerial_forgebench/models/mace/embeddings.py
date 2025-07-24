"""MACE embedding extraction utilities."""

from typing import Union

import numpy as np
import torch
from mace import data
from pymatgen.core.structure import Structure
from torch_geometric.data import Batch, Data

from lematerial_forgebench.models.base import BaseEmbeddingExtractor


class MACEEmbeddingExtractor(BaseEmbeddingExtractor):
    """Embedding extractor for MACE models."""

    def __init__(self, calculator, device="cpu"):
        """Initialize with a MACE calculator."""
        self.calculator = calculator
        self.device = device

    def extract_node_embeddings(
        self, structure: Union[Structure, list[Structure]]
    ) -> Union[np.ndarray, list[np.ndarray]]:
        """Extract per-atom embeddings from MACE model.

        Parameters
        ----------
        structure : Union[Structure, list[Structure]]
            Input structure or list of structures

        Returns
        -------
        np.ndarray
            Node embeddings with shape (n_atoms, descriptor_dim)
        """
        if isinstance(structure, list):
            keyspec = data.KeySpecification(
                info_keys={}, arrays_keys={"charges": self.calculator.charges_key}
            )
            configs = [
                data.config_from_atoms(
                    _structure.to_ase_atoms(),
                    key_specification=keyspec,
                    head_name=self.calculator.head,
                )
                for _structure in structure
            ]
            atomic_data_list = [
                Data(
                    **data.AtomicData.from_config(
                        config,
                        z_table=self.calculator.z_table,
                        cutoff=self.calculator.r_max,
                        heads=self.calculator.available_heads,
                    ).__dict__
                )
                for config in configs
            ]

            batch = Batch.from_data_list(atomic_data_list)
            batch = batch.to(self.device)
            output = self.calculator.models[0](batch)
            node_features = output["node_feats"]
            node_features_list = torch.split(node_features, batch.ptr.diff().tolist())
            node_features_list = [
                node_features.detach().cpu().numpy()
                for node_features in node_features_list
            ]

            return node_features_list

        atoms = structure.to_ase_atoms()

        # Use MACE's built-in descriptor extraction
        descriptors = self.calculator.get_descriptors(
            atoms,
            invariants_only=False,  # Keep equivariant parts
            num_layers=None,  # Use all layers
        )

        return descriptors
