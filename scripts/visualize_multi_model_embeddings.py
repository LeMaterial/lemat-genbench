"""
Script to visualize embeddings from multiple models/runs in a shared space.
Allows setting one dataset as a reference for dimensionality reduction alignment.
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from embedding_utils import get_dimensionality_reducer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_embeddings(
    file_paths: List[str], labels: Optional[List[str]] = None
) -> Dict[str, Dict[str, np.ndarray]]:
    """Load embedding pickle files."""
    data = {}

    if labels and len(labels) != len(file_paths):
        logger.error("Number of labels must match number of file paths")
        sys.exit(1)

    for i, path_str in enumerate(file_paths):
        path = Path(path_str)
        if not path.exists():
            logger.warning(f"File not found: {path}")
            continue

        label = (
            labels[i]
            if labels
            else path.stem.replace("embeddings_", "").replace(".pkl", "")
        )

        try:
            with open(path, "rb") as f:
                embeddings = pickle.load(f)

            if not isinstance(embeddings, dict):
                logger.warning(f"Unexpected format in {path}. Expected dictionary.")
                continue

            data[label] = embeddings
            logger.info(
                f"Loaded {label} from {path} ({len(embeddings)} embedding types)"
            )

        except Exception as e:
            logger.error(f"Error loading {path}: {e}")

    return data


def get_common_embedding_types(data: Dict[str, Dict[str, np.ndarray]]) -> List[str]:
    """Find embedding types present in at least one loaded dataset."""
    types = set()
    for dataset in data.values():
        types.update(dataset.keys())
    return sorted(list(types))


def align_and_plot_embeddings(
    data: Dict[str, Dict[str, np.ndarray]],
    reference_label: str,
    output_dir: Path,
    methods: List[str] = ["pca", "umap", "tsne"],
    mode: str = "ref_transform",
):
    """
    Align embeddings from multiple sources using a reference dataset and plot them.

    Args:
        data: Dictionary mapping dataset labels to their embedding dictionaries.
        reference_label: The label of the dataset to use as reference.
        output_dir: Directory to save plots.
        methods: List of reduction methods ('pca', 'umap', 'tsne').
        mode: 'ref_transform' (fit on ref, transform others) or 'joint' (fit on all).
    """
    if mode == "ref_transform" and reference_label not in data:
        logger.error(
            f"Reference label '{reference_label}' not found in loaded data: {list(data.keys())}"
        )
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_types = get_common_embedding_types(data)

    for emb_type in embedding_types:
        logger.info(f"Processing embedding type: {emb_type}")

        target_datasets = {}
        all_data = []
        all_labels = []

        # For ref_transform, we need ref data specifically
        ref_data = None
        if reference_label in data and emb_type in data[reference_label]:
            ref_data = data[reference_label][emb_type]

        for label, dataset in data.items():
            if emb_type in dataset:
                arr = dataset[emb_type]
                target_datasets[label] = arr
                all_data.append(arr)
                all_labels.extend([label] * len(arr))

        if not target_datasets:
            logger.warning(f"No data found for {emb_type}")
            continue

        # If mode is ref_transform but ref is missing for this type, warn and skip
        if mode == "ref_transform" and ref_data is None:
            logger.warning(
                f"Reference '{reference_label}' missing {emb_type}. Skipping."
            )
            continue

        for method in methods:
            logger.info(f"  Computing {method.upper()}...")

            # Special handling for t-SNE which doesn't support transform() in sklearn
            current_mode = mode
            if method == "tsne" and mode == "ref_transform":
                logger.warning(
                    "t-SNE does not support 'ref_transform' mode (no transform method). Switching to 'joint' mode for t-SNE."
                )
                current_mode = "joint"

            try:
                reduced_data = {}

                if current_mode == "joint":
                    combined_data = np.vstack(all_data)
                    reducer = get_dimensionality_reducer(method, len(combined_data))
                    reduced_combined = reducer.fit_transform(combined_data)

                    current_idx = 0
                    # We must iterate in the same order we constructed combined_data
                    # combined_data was built by iterating data.items()
                    for label, dataset in data.items():
                        if emb_type in dataset:
                            n = len(dataset[emb_type])
                            reduced_data[label] = reduced_combined[
                                current_idx : current_idx + n
                            ]
                            current_idx += n

                else:
                    reducer = get_dimensionality_reducer(method, len(ref_data))
                    reducer.fit(ref_data)

                    for label, emb_array in target_datasets.items():
                        try:
                            reduced_data[label] = reducer.transform(emb_array)
                        except Exception as e:
                            logger.warning(
                                f"Failed to transform {label} with {method}: {e}"
                            )

                plt.figure(figsize=(12, 10))
                sorted_labels = sorted(reduced_data.keys())

                for label in sorted_labels:
                    coords = reduced_data[label]
                    is_ref = label == reference_label

                    alpha = 0.6
                    size = 30
                    if current_mode == "ref_transform":
                        if is_ref:
                            alpha = 0.4
                            size = 20

                    plt.scatter(
                        coords[:, 0],
                        coords[:, 1],
                        alpha=alpha,
                        s=size,
                        label=f"{label} {'(Ref)' if is_ref and current_mode == 'ref_transform' else ''}",
                        edgecolor="none",
                    )

                plt.xlabel(f"{method.upper()} Component 1")
                plt.ylabel(f"{method.upper()} Component 2")

                title_suffix = (
                    f"Ref: {reference_label}"
                    if current_mode == "ref_transform"
                    else "Joint Embedding"
                )
                plt.title(f"Combined {emb_type} ({method.upper()})\n{title_suffix}")
                plt.legend()
                plt.grid(True, alpha=0.3)

                plot_file = (
                    output_dir / f"combined_{method}_{emb_type}_{current_mode}.png"
                )
                plt.savefig(plot_file, dpi=300, bbox_inches="tight")
                plt.close()
                logger.info(f"  Saved {plot_file}")

            except Exception as e:
                logger.error(f"Error calculating/plotting {method} for {emb_type}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize embeddings from multiple models."
    )
    parser.add_argument("files", nargs="+", help="Path to embedding .pkl files")
    parser.add_argument(
        "--labels", nargs="+", help="Custom labels for each file (must match count)"
    )
    parser.add_argument(
        "--reference",
        required=False,
        help="Label of the reference dataset (defaults to first file's label)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="results_final/combined_plots",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--mode",
        choices=["ref_transform", "joint"],
        default="ref_transform",
        help="Visualization mode: 'ref_transform' (map to reference) or 'joint' (fit on all)",
    )

    args = parser.parse_args()

    data = load_embeddings(args.files, args.labels)

    if not data:
        logger.error("No data loaded. Exiting.")
        return

    if args.reference:
        reference_label = args.reference
    else:
        reference_label = list(data.keys())[0]
        logger.info(f"No reference specified. Using '{reference_label}' as reference.")

    output_dir = Path(args.output)

    align_and_plot_embeddings(data, reference_label, output_dir, mode=args.mode)


if __name__ == "__main__":
    main()
