"""
Script to visualize embeddings from multiple models/runs in a shared space.
Allows setting one dataset as a reference for dimensionality reduction alignment.
Supports filtering by SUN/MSUN status from benchmark results.
"""

import argparse
import glob
import json
import logging
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

import matplotlib.pyplot as plt
import numpy as np
from embedding_utils import get_dimensionality_reducer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def expand_file_paths(file_paths: List[str]) -> List[Path]:
    """Expand file paths, supporting glob patterns and directories."""
    expanded = []
    for path_str in file_paths:
        path = Path(path_str)

        if path.is_dir():
            pkl_files = list(path.rglob("*.pkl"))
            if pkl_files:
                expanded.extend(pkl_files)
                logger.info(f"Found {len(pkl_files)} .pkl files in directory: {path}")
            else:
                logger.warning(f"No .pkl files found in directory: {path}")
        elif "*" in path_str or "?" in path_str:
            globbed = glob.glob(path_str, recursive=True)
            if globbed:
                expanded.extend([Path(f) for f in globbed if Path(f).suffix == ".pkl"])
                logger.info(
                    f"Glob pattern '{path_str}' matched {len(globbed)} .pkl files"
                )
            else:
                logger.warning(f"Glob pattern '{path_str}' matched no files")
        else:
            if path.exists():
                if path.suffix == ".pkl":
                    expanded.append(path)
                else:
                    logger.warning(f"File {path} is not a .pkl file, skipping")
            else:
                logger.warning(f"File not found: {path}")

    return expanded


def load_embeddings(
    file_paths: List[str],
    labels: Optional[List[str]] = None,
    embedding_types_filter: Optional[List[str]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Load embedding pickle files.

    Args:
        file_paths: List of file paths, glob patterns, or directories
        labels: Optional custom labels for each file
        embedding_types_filter: Optional list of embedding types to keep (e.g., ['orb_graph', 'mace_graph'])
    """
    expanded_paths = expand_file_paths(file_paths)

    if not expanded_paths:
        logger.error("No .pkl files found from provided paths")
        return {}

    data = {}

    if labels and len(labels) != len(expanded_paths):
        logger.warning(
            f"Number of labels ({len(labels)}) doesn't match number of files ({len(expanded_paths)}). Using auto-generated labels."
        )
        labels = None

    for i, path in enumerate(expanded_paths):
        if labels:
            label = labels[i]
        else:
            # Extract model name from parent directory (e.g., embeddings_plaid_pp_21102025 -> plaid_pp)
            parent_dir = path.parent.name
            if parent_dir.startswith("embeddings_"):
                # Remove 'embeddings_' prefix and date suffix (numbers at the end)
                model_part = parent_dir.replace("embeddings_", "")
                # Remove trailing date pattern (numbers, possibly with underscores)
                model_name = re.sub(r"_\d+$", "", model_part)
                label = (
                    model_name
                    if model_name
                    else path.stem.replace("embeddings_", "").replace(".pkl", "")
                )
            else:
                # Fallback to filename if parent doesn't match pattern
                label = path.stem.replace("embeddings_", "").replace(".pkl", "")

        try:
            with open(path, "rb") as f:
                embeddings = pickle.load(f)

            if not isinstance(embeddings, dict):
                logger.warning(f"Unexpected format in {path}. Expected dictionary.")
                continue

            if embedding_types_filter:
                filtered_embeddings = {
                    k: v
                    for k, v in embeddings.items()
                    if any(filter_type in k for filter_type in embedding_types_filter)
                }
                if not filtered_embeddings:
                    logger.warning(
                        f"No matching embedding types in {path} (filter: {embedding_types_filter})"
                    )
                    continue
                embeddings = filtered_embeddings

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


def clean_label_for_display(label: str) -> str:
    """Clean up label for display in plots.

    Removes common prefixes like 'dataset-mp20-prerelax-' and suffixes like '_2500.csv'.

    Args:
        label: Original label

    Returns:
        Cleaned label for display
    """
    # Remove "dataset-mp20-prerelax-" or similar prefixes
    cleaned = re.sub(r"^dataset-[^-]+-prerelax-", "", label)
    # Remove "_2500.csv" or similar suffixes
    cleaned = re.sub(r"_\d+\.csv$", "", cleaned)
    # Remove just ".csv" if present
    cleaned = cleaned.replace(".csv", "")
    return cleaned


def extract_sun_msun_indices(benchmark_json_path: Path) -> Dict[str, Set[int]]:
    """Extract SUN and MSUN structure indices from benchmark JSON file.

    Args:
        benchmark_json_path: Path to benchmark JSON file

    Returns:
        Dictionary with 'sun_indices' and 'msun_indices' as sets of integers
    """
    with open(benchmark_json_path, "r") as f:
        data = json.load(f)

    sun_result_str = data.get("results", {}).get("sun", "")
    if not sun_result_str:
        logger.warning(f"No SUN results found in {benchmark_json_path}")
        return {"sun_indices": set(), "msun_indices": set()}

    # Extract indices using regex
    sun_match = re.search(r"'sun_indices':\s*\[([^\]]+)\]", sun_result_str)
    msun_match = re.search(r"'msun_indices':\s*\[([^\]]+)\]", sun_result_str)

    def parse_indices(match):
        if not match:
            return set()
        indices_str = match.group(1)
        # Handle empty list
        if not indices_str.strip():
            return set()
        # Parse comma-separated integers
        try:
            return set(int(x.strip()) for x in indices_str.split(",") if x.strip())
        except ValueError as e:
            logger.warning(f"Error parsing indices: {e}")
            return set()

    sun_indices = parse_indices(sun_match)
    msun_indices = parse_indices(msun_match)

    logger.info(
        f"Extracted from {benchmark_json_path.name}: "
        f"{len(sun_indices)} SUN indices, {len(msun_indices)} MSUN indices"
    )

    return {"sun_indices": sun_indices, "msun_indices": msun_indices}


def load_benchmark_indices(
    benchmark_dir: Path, model_label: str
) -> Dict[str, Set[int]]:
    """Load SUN/MSUN indices for a specific model from benchmark directory.

    Args:
        benchmark_dir: Directory containing benchmark JSON files
        model_label: Model label from embeddings (e.g., 'diffcsp', 'mattergen', 'plaid_pp')

    Returns:
        Dictionary with 'sun_indices' and 'msun_indices'
    """
    # Find all benchmark JSON files
    all_benchmark_files = list(benchmark_dir.glob("*_*_comprehensive_*.json"))

    # Normalize model label for matching
    model_normalized = model_label.lower().replace("_", "")

    matching_files = []
    for benchmark_file in all_benchmark_files:
        filename = benchmark_file.stem
        parts = filename.split("_")
        if len(parts) >= 2:
            file_model = parts[0].lower()
            file_model_combined = (
                "_".join(parts[:2]).lower() if len(parts) >= 2 else file_model
            )

            # Match if model name is in filename or vice versa
            if (
                model_normalized in file_model
                or file_model in model_normalized
                or model_normalized in file_model_combined
                or file_model_combined in model_normalized
            ):
                matching_files.append(benchmark_file)

    if not matching_files:
        logger.warning(
            f"No benchmark file found for model '{model_label}' in {benchmark_dir}. "
            f"Available files: {[f.name for f in all_benchmark_files[:5]]}"
        )
        return {"sun_indices": set(), "msun_indices": set()}

    if len(matching_files) > 1:
        logger.warning(
            f"Multiple benchmark files found for '{model_label}', using first: {matching_files[0].name}"
        )

    benchmark_file = matching_files[0]
    indices = extract_sun_msun_indices(benchmark_file)

    return {
        **indices,
        "benchmark_file": benchmark_file,
    }


def filter_embeddings_by_indices(
    embeddings: Dict[str, np.ndarray],
    valid_indices: Set[int],
) -> Dict[str, np.ndarray]:
    """Filter embeddings to only include structures at specified indices.

    Args:
        embeddings: Dictionary mapping embedding type names to numpy arrays
        valid_indices: Set of structure indices to keep (relative to valid structures)

    Returns:
        Filtered embeddings dictionary
    """
    filtered = {}

    for emb_type, emb_array in embeddings.items():
        if len(emb_array) == 0:
            continue

        embedding_indices = sorted(
            [idx for idx in valid_indices if idx < len(emb_array)]
        )

        if embedding_indices:
            filtered[emb_type] = emb_array[embedding_indices]
            logger.debug(
                f"Filtered {emb_type}: {len(embedding_indices)}/{len(emb_array)} structures"
            )
        else:
            logger.warning(
                f"No matching indices for {emb_type} "
                f"(indices: {sorted(valid_indices)[:10]}..., array size: {len(emb_array)})"
            )

    return filtered


def align_and_plot_embeddings(
    data: Dict[str, Dict[str, np.ndarray]],
    reference_label: str,
    output_dir: Path,
    methods: List[str] = ["umap", "tsne"],
    mode: str = "ref_transform",
    filter_label: Optional[str] = None,
):
    """
    Align embeddings from multiple sources using a reference dataset and plot them.

    Args:
        data: Dictionary mapping dataset labels to their embedding dictionaries.
        reference_label: The label of the dataset to use as reference.
        output_dir: Directory to save plots.
        methods: List of reduction methods ('pca', 'umap', 'tsne').
        mode: 'ref_transform' (fit on ref, transform others) or 'joint' (fit on all).
        filter_label: Optional label to add to plot titles (e.g., 'SUN', 'MSUN').
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

                    # Clean label for display
                    display_label = clean_label_for_display(label)

                    plt.scatter(
                        coords[:, 0],
                        coords[:, 1],
                        alpha=alpha,
                        s=size,
                        label=f"{display_label} {'(Ref)' if is_ref and current_mode == 'ref_transform' else ''}",
                        edgecolor="none",
                    )

                plt.xlabel(f"{method.upper()} Component 1")
                plt.ylabel(f"{method.upper()} Component 2")

                title_suffix = (
                    f"Ref: {reference_label}"
                    if current_mode == "ref_transform"
                    else "Joint Embedding"
                )
                if filter_label:
                    title_suffix = f"{filter_label} - {title_suffix}"
                plt.title(f"Combined {emb_type} ({method.upper()})\n{title_suffix}")
                plt.legend()
                plt.grid(True, alpha=0.3)

                filter_suffix = f"_{filter_label.lower()}" if filter_label else ""
                plot_file = (
                    output_dir
                    / f"combined_{method}_{emb_type}_{current_mode}{filter_suffix}.png"
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
    parser.add_argument(
        "files",
        nargs="+",
        help="Path to embedding .pkl files, glob patterns (e.g., 'results_final/embeddings_*/*.pkl'), or directories",
    )
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
    parser.add_argument(
        "--embedding-types",
        nargs="+",
        help="Filter to specific embedding types (e.g., 'graph' to only plot *_graph embeddings, or 'orb_graph mace_graph' for specific types)",
    )
    parser.add_argument(
        "--filter",
        nargs="+",
        choices=["sun", "msun"],
        help="Filter embeddings by SUN (Stable, Unique, Novel) and/or MSUN (Metastable, Unique, Novel) structures. Can specify both (e.g., --filter sun msun). Requires --benchmark-dir.",
    )
    parser.add_argument(
        "--benchmark-dir",
        help="Directory containing benchmark JSON files (required when using --filter)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["pca", "umap", "tsne"],
        choices=["pca", "umap", "tsne"],
        help="Dimensionality reduction methods to use",
    )

    args = parser.parse_args()

    # Validate filtering arguments
    if args.filter and not args.benchmark_dir:
        logger.error("--benchmark-dir is required when using --filter")
        return

    data = load_embeddings(args.files, args.labels, args.embedding_types)

    if not data:
        logger.error("No data loaded. Exiting.")
        return

    # Determine reference label
    if args.reference:
        reference_label = args.reference
    else:
        reference_label = list(data.keys())[0]
        logger.info(f"No reference specified. Using '{reference_label}' as reference.")

    output_dir = Path(args.output)

    # Apply SUN/MSUN filtering if requested
    if args.filter:
        benchmark_dir = Path(args.benchmark_dir)
        if not benchmark_dir.exists():
            logger.error(f"Benchmark directory not found: {benchmark_dir}")
            return

        # Normalize filter types (handle duplicates)
        filter_types = list(set([f.lower() for f in args.filter]))
        filter_types.sort()  # Consistent ordering: msun, sun

        logger.info(
            f"\n{'=' * 60}\n"
            f"Filtering embeddings for {', '.join([f.upper() for f in filter_types])} structures...\n"
            f"{'=' * 60}"
        )

        # Load benchmark indices for each model
        model_indices = {}
        for label in data.keys():
            indices_data = load_benchmark_indices(benchmark_dir, label)
            if indices_data.get("sun_indices") or indices_data.get("msun_indices"):
                model_indices[label] = indices_data

        if not model_indices:
            logger.error("No SUN/MSUN indices found for any models. Exiting.")
            return

        # Process each filter type separately
        for filter_type in filter_types:
            filter_label = filter_type.upper()
            index_key = f"{filter_type}_indices"

            logger.info(f"\nProcessing {filter_label} structures...")

            # Filter embeddings for this type
            filtered_data = {}

            for label, embeddings in data.items():
                if label not in model_indices:
                    logger.warning(f"Skipping {label} - no benchmark indices found")
                    continue

                indices = model_indices[label].get(index_key, set())
                if indices:
                    filtered_embeddings = filter_embeddings_by_indices(
                        embeddings, indices
                    )
                    if filtered_embeddings:
                        filtered_data[label] = filtered_embeddings
                        logger.info(
                            f"Filtered {label}: {len(indices)} {filter_label} structures "
                            f"(embedding size: {len(list(embeddings.values())[0]) if embeddings else 0})"
                        )
                else:
                    logger.warning(f"No {filter_label} indices found for {label}")

            if not filtered_data:
                logger.warning(
                    f"No {filter_label} structures found for any models. Skipping {filter_label} plots."
                )
                continue

            # Create subdirectory for this filter type
            filter_output_dir = output_dir / filter_type.lower()

            # Generate plots for this filter type
            align_and_plot_embeddings(
                filtered_data,
                reference_label,
                filter_output_dir,
                methods=args.methods,
                mode=args.mode,
                filter_label=filter_label,
            )
    else:
        # No filtering - plot all data
        align_and_plot_embeddings(
            data,
            reference_label,
            output_dir,
            methods=args.methods,
            mode=args.mode,
            filter_label=None,
        )


if __name__ == "__main__":
    main()
