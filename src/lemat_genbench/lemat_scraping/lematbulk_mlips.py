import os
from dataclasses import dataclass

import numpy as np
from datasets import load_dataset
from pymatgen.core import Structure

from lemat_genbench.preprocess.universal_stability_preprocess import (
    UniversalStabilityPreprocessor,
)


@dataclass
class StabilityPreprocessors:
    """ """

    stability_preprocessor_orb: UniversalStabilityPreprocessor | None = None
    stability_preprocessor_mace: UniversalStabilityPreprocessor | None = None
    stability_preprocessor_uma: UniversalStabilityPreprocessor | None = None
    stability_preprocessor_equiformer: UniversalStabilityPreprocessor | None = None


def lematbulk_item_to_structure(item: dict) -> Structure:
    """Convert a LeMat-Bulk item to a pymatgen Structure object.

    Parameters
    ----------
    item : dict
        The item to convert.

    Returns
    -------
    Structure
        The pymatgen Structure object.
    """
    sites = item["species_at_sites"]
    coords = item["cartesian_site_positions"]
    cell = item["lattice_vectors"]

    structure = Structure(
        species=sites, coords=coords, lattice=cell, coords_are_cartesian=True
    )

    return structure


def stability_calculations(structures, stability_preprocessor, batch_size=32):
    """Process a list of structures in batches using the stability preprocessor.

    Parameters
    ----------
    structures : list[Structure]
        List of pymatgen Structure objects to process
    stability_preprocessor : UniversalStabilityPreprocessor
        Preprocessor to use for calculations
    batch_size : int
        Number of structures to process at once

    Returns
    -------
    dict
        Dictionary containing the processed results with keys:
        - GraphEmbeddings: list of graph embeddings
        - NodeEmbeddings: list of node embeddings
    """
    # Initialize result containers
    results = {
        "GraphEmbeddings": [],
        "NodeEmbeddings": [],
    }

    # Process structures in batches
    for i in range(0, len(structures), batch_size):
        batch = structures[i : i + batch_size]
        print(
            f"Processing batch {i // batch_size + 1}/{(len(structures) + batch_size - 1) // batch_size}"
        )

        graph_embeddings = stability_preprocessor.calculator.extract_embeddings(batch)

        # Extract results for each structure in the batch
        for graph_embedding in graph_embeddings:
            results["GraphEmbeddings"].append(graph_embedding.graph_embedding)
            results["NodeEmbeddings"].append(graph_embedding.node_embeddings)

    return results


def process_item_action(dataset, stability_processors, batch_size=32):
    """Process a batch of items from the dataset using multiple stability processors.

    Parameters
    ----------
    dataset : Dataset
        HuggingFace dataset containing structures to process
    stability_processors : StabilityPreprocessors
        Container for different MLIP stability processors
    batch_size : int
        Number of structures to process at once

    Returns
    -------
    dict
        Dictionary containing results for each MLIP model
    """
    # Convert all items to structures at once
    structures = []
    lemat_ids = []
    for item in dataset:
        structures.append(lematbulk_item_to_structure(item))
        lemat_ids.append(item["immutable_id"])

    output = {"orb": [], "mace": [], "uma": [], "equiformer": []}

    # Process with each available preprocessor
    processors = {
        "uma": stability_processors.stability_preprocessor_uma,
        "orb": stability_processors.stability_preprocessor_orb,
        "mace": stability_processors.stability_preprocessor_mace,
        "equiformer": stability_processors.stability_preprocessor_equiformer,
    }

    for name, processor in processors.items():
        if processor is not None:
            print(f"Starting {name} calculation")
            print(processor.config.name)
            output[name] = stability_calculations(
                structures, processor, batch_size=batch_size
            )
            print(f"Finished {name} calculation")

    return output


if __name__ == "__main__":
    import pandas as pd

    full_dataset = False
    vals_spacing = 100
    vals = np.arange(
            0, 1000, vals_spacing
    )
    batch_size = 8
    dir_name = "test_small_lematbulk"

    dataset_name = "Lematerial/LeMat-Bulk"
    name = "compatible_pbe"
    split = "train"
    dataset = load_dataset(dataset_name, name=name, split=split, streaming=False)

    mlips = ["uma"]

    timeout = 60  # seconds, for one MLIP calculation

    try:
        if "orb" not in mlips:
            raise ValueError
        orb_stability_preprocessor = UniversalStabilityPreprocessor(
            model_name="orb",
            model_config={"device": "cpu"},
            timeout=timeout,
            relax_structures=False,
        )
    except (ImportError, ValueError) as e:
        print(e)
        orb_stability_preprocessor = None

    try:
        if "mace" not in mlips:
            raise ValueError
        mace_stability_preprocessor = UniversalStabilityPreprocessor(
            model_name="mace",
            model_config={"device": "cpu"},
            timeout=timeout,
            relax_structures=False,
        )
    except (ImportError, ValueError) as e:
        print(e)
        mace_stability_preprocessor = None

    try:
        if "uma" not in mlips:
            raise ValueError
        uma_stability_preprocessor = UniversalStabilityPreprocessor(
            model_name="uma",
            model_config={"device": "cpu"},
            timeout=timeout,
            relax_structures=False,
        )
    except (ImportError, ValueError) as e:
        print(e)
        uma_stability_preprocessor = None

    try:
        if "equiformer" not in mlips:
            raise ValueError
        equiformer_stability_preprocessor = UniversalStabilityPreprocessor(
            model_name="equiformer",
            model_config={"device": "cpu"},
            timeout=timeout,
            relax_structures=False,
        )
    except (ImportError, ValueError) as e:
        print(e)
        equiformer_stability_preprocessor = None

    preprocessors = StabilityPreprocessors(
        orb_stability_preprocessor,
        mace_stability_preprocessor,
        uma_stability_preprocessor,
        equiformer_stability_preprocessor,
    )

    if full_dataset:
        vals = np.arange(
            0, len(dataset), vals_spacing
        )  # example for running on all of LeMatBulk

    if os.path.exists("data/" + dir_name):
        pass
    else:
        os.mkdir("data/" + dir_name)

    for i in vals:
        dataset_temp = dataset.select(range(i, min(len(dataset), i + vals_spacing)))

        output = process_item_action(
            dataset=dataset_temp,
            stability_processors=preprocessors,
            batch_size=batch_size,
        )
        print("Creating DataFrame and saving to pkl...")
        for key in output.keys():
            if isinstance(output[key], dict):
                temp_df = pd.DataFrame(output[key], columns=output[key].keys())
                if os.path.exists("data/" + dir_name + "/" + key):
                    pass
                else:
                    os.mkdir("data/" + dir_name + "/" + key)

                temp_df.to_pickle(
                    "data/"
                    + dir_name
                    + "/"
                    + key
                    + "/"
                    + key
                    + "_lematbulk_MLIP_full_"
                    + str(i)
                    + ".pkl"
                )
            else:
                pass
