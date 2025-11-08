import json
import os
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition, Element
from scipy import sparse
from tqdm import tqdm

CURRENT_FOLDER = os.path.dirname(Path(__file__).resolve())


def build_formation_energy_reference_file():
    ds_pbe = load_dataset("LeMaterial/LeMat-Bulk", "compatible_pbe")
    # ds_pbesol = load_dataset("LeMaterial/LeMat-Bulk", "compatible_pbesol")
    # ds_scan = load_dataset("LeMaterial/LeMat-Bulk", "compatible_scan")

    data = {
        "energy": [
            *[
                x / z
                for x, y, z in zip(
                    ds_pbe["train"]["energy"],
                    ds_pbe["train"]["nelements"],
                    ds_pbe["train"]["nsites"],
                )
                if y == 1
            ],
        ],
        "composition": [
            *[
                [x for x in Composition(Counter(y)).chemical_system_set][0]
                for x, y in zip(
                    ds_pbe["train"]["nelements"], ds_pbe["train"]["species_at_sites"]
                )
                if x == 1
            ],
        ],
    }

    element_chem_pot = {}
    for element, energy in (
        pd.DataFrame(data).groupby("composition").min().to_dict()["energy"].items()
    ):
        if element not in element_chem_pot:
            element_chem_pot[element] = {}
        element_chem_pot[element]["pbe"] = energy

    json.dump(
        element_chem_pot,
        open(os.path.join(CURRENT_FOLDER, "element_chem_pot.json"), "w"),
    )


def get_formation_energy_from_composition_energy(
    total_energy: float, composition: str | Composition, functional="pbe"
):
    element_chem_pot_file = os.path.join(CURRENT_FOLDER, "element_chem_pot.json")

    if not os.path.exists(element_chem_pot_file):
        raise FileNotFoundError(
            f"Reference energy file not found: {element_chem_pot_file}. "
            "Run build_formation_energy_reference_file() first."
        )

    with open(element_chem_pot_file) as f:
        element_chem_pot = json.load(f)

    try:
        res = 0
        # Handle charged species by converting to neutral elements
        neutral_composition_dict = {}
        missing_elements = []

        for element, count in composition.as_dict().items():
            # Handle charged species by extracting the base element
            # element can be a string like 'Cs+' or a Species object
            if isinstance(element, str):
                # For string-based charged species like 'Cs+', 'Ni2+', extract the base element
                if "+" in element or "-" in element:
                    # Extract the base element (everything before the charge)
                    base_element = element.rstrip("+-0123456789")
                else:
                    base_element = element
            elif hasattr(element, "element"):
                # For Species objects, extract the base element
                base_element = element.element
            else:
                # For neutral elements
                base_element = element

            # Check if reference energy is available
            if base_element not in element_chem_pot:
                missing_elements.append(base_element)
                continue

            if functional not in element_chem_pot[base_element]:
                raise ValueError(
                    f"Functional '{functional}' not available for element '{base_element}'"
                )

            # Use neutral element for chemical potential lookup
            if base_element not in neutral_composition_dict:
                neutral_composition_dict[base_element] = 0
            neutral_composition_dict[base_element] += count

        if missing_elements:
            raise ValueError(
                f"Missing reference energies for elements: {missing_elements}"
            )

        res = total_energy - sum(
            [
                element_chem_pot[k][functional] * v
                for k, v in neutral_composition_dict.items()
            ]
        )
        return res
    except Exception as e:
        print("Error in get_formation_energy_from_composition_energy: ", e)
        return None


def get_formation_energy_per_atom_from_composition_energy(
    total_energy, composition, functional="pbe"
):
    """Calculate formation energy per atom from total energy and composition.

    This function properly handles charged species by converting them
    to neutral elements before looking up reference chemical potentials.

    Parameters
    ----------
    total_energy : float
        Total energy in eV
    composition : Composition
        Pymatgen composition object (may contain charged species)
    functional : str, optional
        DFT functional to use for reference energies (default is "pbe")

    Returns
    -------
    float
        Formation energy per atom in eV/atom (intensive property, normalized per atom)

    Notes
    -----
    This follows Materials Project conventions for formation energy.
    """

    try:
        formation_energy = get_formation_energy_from_composition_energy(
            total_energy, composition, functional=functional
        )
        composition = Composition(composition)  # Ensure it's a Composition object
        formation_energy = formation_energy / composition.num_atoms
        if formation_energy is None:
            raise ValueError("Formation energy calculation returned None")
        return formation_energy
    except Exception as e:
        raise ValueError(
            f"Failed to compute formation energy for {composition.formula}: {str(e)}"
        ) from e


def one_hot_encode_composition(elements):
    one_hot = np.zeros(
        119
    )  # 119 to use direct Z indexing (0 unused, 1-118 for elements)
    for element in elements:
        try:
            # Handle charged species by extracting the base element
            if isinstance(element, str):
                # For string-based charged species like 'Cs+', 'Ni2+', extract the base element
                if "+" in element or "-" in element:
                    # Extract the base element (everything before the charge)
                    base_element = element.rstrip("+-0123456789")
                else:
                    base_element = element
            elif hasattr(element, "element"):
                # For Species objects, extract the base element
                base_element = element.element
            else:
                # For neutral elements
                base_element = element

            # Validate element and get atomic number
            element_obj = Element(base_element)
            one_hot[int(element_obj.number)] = 1
        except ValueError as e:
            print(f"Warning: Invalid element '{element}': {e}")
            continue
    return one_hot


def process_chunk(chunk):
    one_hot_compositions = []
    for elements in tqdm(chunk):
        one_hot_compositions.append(one_hot_encode_composition(elements))
    return one_hot_compositions


def _retrieve_df_mp20(hull_type="mace", threshold=0.025):
    """Retrieve MP-20 dataset for hull computations.

    Parameters
    ----------
    hull_type : str, optional
        Type of hull to use for MP-20 ('total_energy', 'mace', 'orb', 'uma', 'ensemble')
        Default is 'mace'
    threshold : float, optional
        Energy above hull threshold in eV/atom (default 0.025)

    Returns
    -------
    pd.DataFrame
        Dataset of MP-20 materials close to the hull
    """
    try:
        # Try local MP-20 hull-specific path first
        # Try .parquet first (main dataset), then .npz (metadata backup)
        for filename_pattern in [
            f"{hull_type}_above_hull_dataset.parquet",  # Main dataset file
            f"{hull_type}_above_hull_metadata.npz",     # Backup metadata file
        ]:
            local_hull_path = os.path.join(
                CURRENT_FOLDER,
                "..",
                "..",
                "..",
                "data",
                "mp20_hulls",
                filename_pattern,
            )
            
            if os.path.exists(local_hull_path):
                if filename_pattern.endswith(".parquet"):
                    dataset = pd.read_parquet(local_hull_path)
                else:  # .npz
                    import numpy as np
                    data = np.load(local_hull_path, allow_pickle=True)
                    # Convert npz to DataFrame
                    dataset = pd.DataFrame({k: data[k] for k in data.files})
                
                if "species_at_sites" in dataset.columns:
                    dataset["species_at_sites"] = dataset["species_at_sites"].apply(
                        lambda x: x.tolist() if hasattr(x, "tolist") else x
                    )
                return dataset

        # Try to load from HuggingFace as dataset splits
        try:
            from datasets import load_dataset

            dataset_dict = load_dataset("LeMaterial/mp20-hull-reference", token=True)

            if hull_type in dataset_dict:
                dataset = dataset_dict[hull_type].to_pandas()
                # Ensure species_at_sites is properly formatted
                if "species_at_sites" in dataset.columns:
                    dataset["species_at_sites"] = dataset["species_at_sites"].apply(
                        lambda x: x.tolist() if hasattr(x, "tolist") else x
                    )
                return dataset
        except Exception:
            pass

        # Fall back to file-based download from HuggingFace Hub
        from huggingface_hub import hf_hub_download

        # Try to download parquet dataset file
        file_path = hf_hub_download(
            repo_id="LeMaterial/mp20-hull-reference",
            filename=f"{hull_type}_above_hull_dataset.parquet",
            repo_type="dataset",
            token=True,  # Use cached HuggingFace token
            cache_dir=os.path.join(CURRENT_FOLDER, "..", "..", "..", "data", ".cache"),
        )
        
        # Load parquet file as DataFrame
        dataset = pd.read_parquet(file_path)
        
        if "species_at_sites" in dataset.columns:
            dataset["species_at_sites"] = dataset["species_at_sites"].apply(
                lambda x: x.tolist() if hasattr(x, "tolist") else x
            )
        return dataset
    except Exception as e:
        raise RuntimeError(
            f"MP-20 hull dataset not found for hull type '{hull_type}' and threshold {threshold}. "
            "Tried local path and HuggingFace Hub."
            f" Error: {e}"
        )


@lru_cache(maxsize=None)
def _retrieve_df(hull_type="dft", threshold=0.001):
    """Retrieve dataset for hull computations.

    Parameters
    ----------
    hull_type : str, optional
        Type of hull to use ('dft', 'orb', 'uma', 'mace_mp', 'mace_omat')
        For MP-20 dataset, use 'model_mp20' prefix (e.g., 'mace_mp20', 'orb_mp20', 'uma_mp20', 'ensemble_mp20')
        Default is 'dft' for backward compatibility
    threshold : float, optional
        Energy above hull threshold in eV/atom (default 0.001)

    Returns
    -------
    pd.DataFrame
        Dataset of materials close to the hull
    """
    # Check if this is an MP-20 hull type
    if hull_type.endswith("_mp20"):
        # Extract the model type (e.g., 'mace_mp20' -> 'mace')
        model_type = hull_type.replace("_mp20", "")
        return _retrieve_df_mp20(hull_type=model_type, threshold=threshold)

    try:
        threshold_str = f"{threshold:.3f}".replace(".", "_")

        # Try local hull-specific path first
        local_hull_path = os.path.join(
            CURRENT_FOLDER,
            "..",
            "..",
            "..",
            "data",
            "convex_hulls",
            f"{hull_type}_above_hull_dataset.parquet",
        )

        if os.path.exists(local_hull_path):
            dataset = pd.read_parquet(local_hull_path)
            if "elements" in dataset.columns:
                dataset["elements"] = dataset["elements"].apply(
                    lambda x: x.tolist() if hasattr(x, "tolist") else x
                )
            if "species_at_sites" in dataset.columns:
                dataset["species_at_sites"] = dataset["species_at_sites"].apply(
                    lambda x: x.tolist() if hasattr(x, "tolist") else x
                )
            return dataset

        try:
            from datasets import load_dataset

            dataset_dict = load_dataset("LeMaterial/LeMat-Bulk-MLIP-Hull")

            if hull_type in dataset_dict:
                dataset = dataset_dict[hull_type].to_pandas()
                # species_at_sites should already be lists in the new format
                # but we'll still check just in case
                if "species_at_sites" in dataset.columns:
                    dataset["species_at_sites"] = dataset["species_at_sites"].apply(
                        lambda x: x.tolist() if hasattr(x, "tolist") else x
                    )
                return dataset
        except Exception:
            pass

        # Fall back to file-based approach
        from huggingface_hub import hf_hub_download

        file_path = hf_hub_download(
            repo_id="LeMaterial/LeMat-Bulk-MLIP-Hull",
            filename=f"threshold_{threshold_str}/{hull_type}_above_hull_dataset.parquet",
            repo_type="dataset",
            cache_dir=os.path.join(CURRENT_FOLDER, "..", "..", "..", "data", ".cache"),
        )
        dataset = pd.read_parquet(file_path)
        if "elements" in dataset.columns:
            dataset["elements"] = dataset["elements"].apply(
                lambda x: x.tolist() if hasattr(x, "tolist") else x
            )
        if "species_at_sites" in dataset.columns:
            dataset["species_at_sites"] = dataset["species_at_sites"].apply(
                lambda x: x.tolist() if hasattr(x, "tolist") else x
            )
        return dataset
    except Exception as e:
        raise RuntimeError(
            f"Hull-specific dataset not found for hull type '{hull_type}' and threshold {threshold}. "
            "Tried local path and HuggingFace Hub."
            f" Error: {e}"
        )


def _retrieve_matrix_mp20(hull_type="mace", threshold=0.025):
    """Retrieve MP-20 composition matrix for hull computations.

    Parameters
    ----------
    hull_type : str, optional
        Type of hull to use for MP-20 ('total_energy', 'mace', 'orb', 'uma', 'ensemble')
        Default is 'mace'
    threshold : float, optional
        Energy above hull threshold in eV/atom (default 0.025)

    Returns
    -------
    np.ndarray
        Composition matrix
    """
    try:
        threshold_str = f"{threshold:.3f}".replace(".", "_")

        # Try local MP-20 composition matrix path first
        local_matrix_path = os.path.join(
            CURRENT_FOLDER,
            "..",
            "..",
            "..",
            "data",
            "mp20_hulls",
            f"{hull_type}_above_hull_composition_matrix.npz",
        )

        if os.path.exists(local_matrix_path):
            composition_array = sparse.load_npz(local_matrix_path).toarray()
            return composition_array

        # Try to fetch from HuggingFace Hub
        from huggingface_hub import hf_hub_download

        file_path = hf_hub_download(
            repo_id="LeMaterial/mp20-hull-reference",
            filename=f"threshold_{threshold_str}/{hull_type}_above_hull_composition_matrix.npz",
            repo_type="dataset",
            token=True,  # Use cached HuggingFace token
            cache_dir=os.path.join(CURRENT_FOLDER, "..", "..", "..", "data", ".cache"),
        )
        composition_array = sparse.load_npz(file_path).toarray()
        return composition_array
    except Exception as e:
        raise RuntimeError(
            f"MP-20 composition matrix not found for hull type '{hull_type}' and threshold {threshold}. "
            f"Error: {e}"
        ) from e


@lru_cache(maxsize=None)
def _retrieve_matrix(hull_type="dft", threshold=0.001):
    """Retrieve composition matrix for hull computations.

    Parameters
    ----------
    hull_type : str, optional
        Type of hull to use ('dft', 'orb', 'uma', 'mace_mp', 'mace_omat')
        For MP-20 dataset, use 'model_mp20' suffix (e.g., 'mace_mp20', 'orb_mp20', 'uma_mp20', 'ensemble_mp20')
        Default is 'dft' for backward compatibility
    threshold : float, optional
        Energy above hull threshold in eV/atom (default 0.001)

    Returns
    -------
    np.ndarray
        Composition matrix
    """
    # Check if this is an MP-20 hull type
    if hull_type.endswith("_mp20"):
        # Extract the model type (e.g., 'mace_mp20' -> 'mace')
        model_type = hull_type.replace("_mp20", "")
        return _retrieve_matrix_mp20(hull_type=model_type, threshold=threshold)

    try:
        threshold_str = f"{threshold:.3f}".replace(".", "_")

        local_hull_path = os.path.join(
            CURRENT_FOLDER,
            "..",
            "..",
            "..",
            "data",
            "convex_hulls",
            f"{hull_type}_above_hull_composition_matrix.npz",
        )

        if os.path.exists(local_hull_path):
            composition_array = sparse.load_npz(local_hull_path).toarray()
            return composition_array

        # otherwise fetch from HuggingFace Hub with hull type
        try:
            from huggingface_hub import hf_hub_download

            file_path = hf_hub_download(
                repo_id="LeMaterial/LeMat-Bulk-MLIP-Hull",
                filename=f"threshold_{threshold_str}/{hull_type}_above_hull_composition_matrix.npz",
                repo_type="dataset",
                cache_dir=os.path.join(
                    CURRENT_FOLDER, "..", "..", "..", "data", ".cache"
                ),
            )
            composition_array = sparse.load_npz(file_path).toarray()
            return composition_array
        except Exception:
            pass
    except Exception as e:
        raise RuntimeError(f"Failed to load composition matrix: {e}") from e


def filter_df(df, matrix, composition):
    try:
        structure_vector = one_hot_encode_composition(composition.elements).reshape(
            -1, 1
        )
        forbidden_elements = 1 - structure_vector
        intersection_elements = df.loc[(matrix @ forbidden_elements) == 0]

        if intersection_elements.empty:
            print(
                f"Warning: No reference structures found for composition {composition.formula}"
            )

        return intersection_elements
    except Exception as e:
        raise ValueError(f"Failed to filter reference dataset: {e}") from e


def get_energy_above_hull(total_energy, composition, hull_type="ensemble_mp20", threshold=0.025):
    """Calculate energy above hull from total energy and composition.

    This function properly handles charged species by converting them
    to neutral elements before creating PDEntry objects.

    Parameters
    ----------
    total_energy : float
        Total energy in eV
    composition : Composition
        Pymatgen composition object (may contain charged species)
    hull_type : str, optional
        Type of hull to use:
        - For LeMat-Bulk: 'dft', 'orb', 'uma', 'mace_mp', 'mace_omat'
        - For MP-20: 'mace_mp20', 'orb_mp20', 'uma_mp20', 'ensemble_mp20'
        Default is 'dft' for backward compatibility
    threshold : float, optional
        Energy above hull threshold in eV/atom for reference dataset
        Default 0.001 for LeMat-Bulk, 0.025 for MP-20

    Returns
    -------
    float
        Energy above hull in eV/atom (intensive property, normalized per atom)

    Notes
    -----
    Pymatgen's get_decomp_and_e_above_hull() always returns eV/atom,
    making this an intensive property like formation energy.
    This follows Materials Project conventions.

    Examples
    --------
    >>> # Using LeMat-Bulk DFT hull
    >>> e_hull = get_energy_above_hull(-10.5, Composition("Fe2O3"), hull_type="dft")

    >>> # Using MP-20 MACE hull
    >>> e_hull = get_energy_above_hull(-10.5, Composition("Fe2O3"), hull_type="mace_mp20", threshold=0.025)
    """

    try:
        intersection_elements = filter_df(
            _retrieve_df(hull_type, threshold),
            _retrieve_matrix(hull_type, threshold),
            composition,
        )

        # Create PDEntries from the filtered DataFrame
        # using species_at_sites for composition because no ambiguity there
        pd_entries = [
            PDEntry(Composition(Counter(row["species_at_sites"])), row["energy"])
            for _, row in intersection_elements.iterrows()
        ]

        if not pd_entries:
            raise ValueError(
                f"No entries found in dataset containing any of the elements in: {composition.elements}"
            )

        # Construct phase diagram
        pd = PhaseDiagram(pd_entries)

        # Convert charged species to neutral composition for PDEntry creation
        # This follows the same logic as get_formation_energy_from_composition_energy
        neutral_composition_dict = {}
        for element, count in composition.as_dict().items():
            # Handle charged species by extracting the base element
            if isinstance(element, str):
                # For string-based charged species like 'Cs+', 'Ni2+', extract the base element
                if "+" in element or "-" in element:
                    # Extract the base element (everything before the charge)
                    base_element = element.rstrip("+-0123456789")
                else:
                    base_element = element
            elif hasattr(element, "element"):
                # For Species objects, extract the base element
                base_element = element.element
            else:
                # For neutral elements
                base_element = element

            # Use neutral element for PDEntry creation
            if base_element not in neutral_composition_dict:
                neutral_composition_dict[base_element] = 0
            neutral_composition_dict[base_element] += count

        neutral_composition = Composition(neutral_composition_dict)

        entry = PDEntry(neutral_composition, total_energy)
        e_above_hull = pd.get_decomp_and_e_above_hull(entry, allow_negative=True)[1]

        return e_above_hull

    except Exception as e:
        raise ValueError(
            f"Failed to compute energy above hull for {composition.formula}: {str(e)}"
        ) from e
