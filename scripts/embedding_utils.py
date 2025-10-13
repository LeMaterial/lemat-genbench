#!/usr/bin/env python3
"""
Utilities for handling and visualizing embeddings from Multi-MLIP preprocessors.

This module provides functions to extract, save, and visualize embeddings
generated during materials structure evaluation.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def save_embeddings_from_structures(
    structures: List,
    config: Dict[str, Any],
    run_name: str,
    generate_plots: bool = False,
    logger=None,
) -> None:
    """Save embeddings extracted from structures to pickle files.

    This function extracts embeddings from structures that have been processed
    by the Multi-MLIP preprocessor and saves them for later analysis.

    Parameters
    ----------
    structures : List
        List of processed structures with embedding properties
    config : Dict[str, Any]
        Configuration dictionary (used for metadata)
    run_name: str
        Name of the run for the embeddings
    generate_plots : bool, default=False
        Whether to automatically generate embedding analysis plots
    logger : Logger, optional
        Logger instance for output messages
    """
    import pickle

    if logger:
        logger.info(f"Saving embeddings from {len(structures)} structures...")

    # Create embeddings directory
    embeddings_dir = (
        Path(__file__).parent.parent / "results_final" / f"embeddings_{run_name}"
    )
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Extract embeddings
    embeddings = {}
    mlip_names = ["orb", "mace", "uma"]

    for mlip_name in mlip_names:
        graph_embeddings = []
        node_embeddings = []

        for structure in structures:
            # Extract graph embeddings
            graph_emb_key = f"graph_embedding_{mlip_name}"
            if (
                graph_emb_key in structure.properties
                and structure.properties[graph_emb_key] is not None
            ):
                graph_embeddings.append(structure.properties[graph_emb_key])

            # Extract node embeddings (aggregated)
            node_emb_key = f"node_embeddings_{mlip_name}"
            if (
                node_emb_key in structure.properties
                and structure.properties[node_emb_key] is not None
            ):
                node_emb = structure.properties[node_emb_key]
                if isinstance(node_emb, np.ndarray) and len(node_emb.shape) > 1:
                    # Aggregate node embeddings (mean across nodes)
                    aggregated = np.mean(node_emb, axis=0)
                    node_embeddings.append(aggregated)
                else:
                    node_embeddings.append(node_emb)

        if graph_embeddings:
            embeddings[f"{mlip_name}_graph"] = np.array(graph_embeddings)
            if logger:
                logger.info(
                    f"Extracted {len(graph_embeddings)} graph embeddings for {mlip_name}"
                )

        if node_embeddings:
            embeddings[f"{mlip_name}_node"] = np.array(node_embeddings)
            if logger:
                logger.info(
                    f"Extracted {len(node_embeddings)} node embeddings for {mlip_name}"
                )

    if embeddings:
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mlip_names_str = "_".join(mlip_names)
        filename = f"embeddings_{mlip_names_str}_{timestamp}.pkl"
        filepath = embeddings_dir / filename

        # Save embeddings
        with open(filepath, "wb") as f:
            pickle.dump(embeddings, f)

        # Save metadata
        metadata = {
            "timestamp": timestamp,
            "n_structures": len(structures),
            "embedding_types": list(embeddings.keys()),
            "mlip_names": mlip_names,
            "config_info": {
                "type": config.get("type", "unknown"),
                "fingerprint_method": config.get("fingerprint_method", "unknown"),
            },
        }

        metadata_file = embeddings_dir / f"embeddings_metadata_{timestamp}.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        if logger:
            logger.info(f"Embeddings saved to: {filepath}")
            logger.info(f"Metadata saved to: {metadata_file}")
            logger.info(f"Total embedding types: {len(embeddings)}")

        # Generate plots if requested
        if generate_plots:
            try:
                if logger:
                    logger.info("Generating embedding analysis plots...")
                generate_embedding_plots(
                    embeddings, embeddings_dir / f"plots_{timestamp}", logger=logger
                )
                if logger:
                    logger.info("📊 Embedding plots generated successfully")
            except Exception as e:
                if logger:
                    logger.warning(f"📊 Failed to generate embedding plots: {e}")
                    logger.info(
                        "📊 Plots can be generated later using scripts/embedding_postprocess.py"
                    )

    else:
        if logger:
            logger.warning("No embeddings found in structures to save")


def generate_embedding_plots(
    embeddings: Dict[str, np.ndarray],
    output_dir: Path,
    logger=None,
    methods: List[str] = None,
) -> None:
    """Generate basic embedding visualization plots.

    Parameters
    ----------
    embeddings : Dict[str, np.ndarray]
        Dictionary mapping embedding names to arrays
    output_dir : Path
        Directory to save plots
    logger : Logger, optional
        Logger instance for output messages
    methods : List[str], optional
        List of dimensionality reduction methods to use.
        Options: "pca", "tsne", "umap". Defaults to all three.
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from umap import UMAP

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up plotting style
    plt.style.use("default")

    if methods is None:
        methods = ["pca", "tsne", "umap"]

    for emb_name, emb_array in embeddings.items():
        if logger:
            logger.info(f"📊 Creating plots for {emb_name} ({emb_array.shape})")

        for method in methods:
            try:
                # Perform dimensionality reduction
                if method == "pca":
                    reducer = PCA(n_components=2, random_state=42)
                elif method == "tsne":
                    reducer = TSNE(
                        n_components=2,
                        random_state=42,
                        perplexity=min(30, len(emb_array) // 4),
                    )
                elif method == "umap":
                    # Reasonable defaults; n_neighbors scales mildly with dataset size
                    n_neighbors = max(5, min(30, len(emb_array) // 20))
                    reducer = UMAP(
                        n_components=2,
                        random_state=42,
                        n_neighbors=n_neighbors,
                        min_dist=0.1,
                        metric="euclidean",
                    )
                else:
                    raise ValueError(
                        f"Unknown dimensionality reduction method: {method}"
                    )

                reduced = reducer.fit_transform(emb_array)

                # Create plot
                plt.figure(figsize=(10, 8))
                plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6, s=30)
                plt.xlabel(f"{method.upper()} Component 1")
                plt.ylabel(f"{method.upper()} Component 2")
                plt.title(
                    f"{emb_name} Embeddings ({method.upper()})\n{len(emb_array)} structures"
                )
                plt.grid(True, alpha=0.3)

                # Save plot
                plot_file = output_dir / f"{method}_{emb_name}_plot.png"
                plt.savefig(plot_file, dpi=300, bbox_inches="tight")
                plt.close()

                if logger:
                    logger.info(f"📊 Saved plot: {plot_file}")

            except Exception as e:
                if logger:
                    logger.warning(
                        f"📊 Failed to create {method} plot for {emb_name}: {e}"
                    )

    if logger:
        logger.info(f"📊 All plots saved to: {output_dir}")
