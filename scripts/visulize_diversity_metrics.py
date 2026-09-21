"""Diveristy Metric Visulizer. The script should call the results of the divseristy
and create a Radar Plot. The next TODO would be to  also integrate with HF to create a heatmap table of best performers

Usage:
    python scripts/visulize_diversity_metrics.py --results_file path/to/benchmark_results.json

"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def read_json(path: str | Path) -> dict:
    """Read a JSON file and return it as a Python dictionary.

    Args:
        path (str | Path): Path to the JSON file.

    Returns:
        dict: Parsed JSON content as a Python dictionary.

    Raises:
        FileNotFoundError: If the JSON file does not exist at the specified path.

    This function is used to read the diversity benchmark JSON file containing
    results that will be further processed and visualized.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_diversity_results(results_dict: Dict) -> str:
    """Extract the diversity results string from the benchmark results dictionary.

    Args:
        results_dict (Dict): The dictionary containing benchmark results.

    Returns:
        str: The diversity results string extracted from the benchmark results.

    This function isolates the 'diversity' section from the benchmark JSON data,
    which contains raw diversity metrics as a string to be parsed later.
    """
    benchmark_results = results_dict["results"]
    diversity = benchmark_results["diversity"]
    return diversity


def extract_metric_result_section(
    raw_text: str, section_key: str = "physical_size_diversity"
):
    """Extract the MetricResult block for a given diversity metric section from raw text.

    Args:
        raw_text (str): Raw string containing diversity metrics with MetricResult(...) blocks.
        section_key (str, optional): The key identifying the diversity metric section to extract.
            Defaults to "physical_size_diversity".

    Returns:
        dict: A dictionary containing:
            - 'metrics' (dict): Metrics dictionary with metric names and values.
            - 'primary_metric' (str): The name of the primary metric.
            - 'uncertainties' (dict): Uncertainties associated with metrics.

    Raises:
        ValueError: If the MetricResult block for the specified section_key cannot be found,
                    or if evaluation of metrics/uncertainties fails.

    This function parses the raw diversity string to extract and safely evaluate the
    MetricResult structure for a specific diversity metric, converting string representations
    of numpy floats into actual float values.
    """
    # Regex to capture the MetricResult(...)
    pattern = rf"""
        '{re.escape(section_key)}'\s*:\s*MetricResult\(\s*
        metrics\s*=\s*\{{(?P<metrics>.*?)\}}\s*,\s*
        primary_metric\s*=\s*'(?P<primary_metric>[^']+)'\s*,\s*
        uncertainties\s*=\s*\{{(?P<uncertainties>.*?)\}}\s*,\s*
        config\s*=
    """

    match = re.search(pattern, raw_text, re.DOTALL | re.VERBOSE)
    if not match:
        raise ValueError(f"Could not locate MetricResult for section '{section_key}'")

    # Extract string fragments
    metrics_str = "{" + match.group("metrics") + "}"
    uncertainties_str = "{" + match.group("uncertainties") + "}"
    primary_metric = match.group("primary_metric")

    # Controlled evaluation environment with disabled builtins
    safe_env = {"np": np, "__builtins__": {}}
    try:
        metrics_dict = eval(metrics_str, safe_env)
        uncertainties_dict = eval(uncertainties_str, safe_env)
    except Exception as e:
        raise ValueError(f"Failed to evaluate metrics/uncertainties: {e}")

    return {
        "metrics": metrics_dict,
        "primary_metric": primary_metric,
        "uncertainties": uncertainties_dict,
    }


def convert_diversity_string_to_cleaned_dict(diversity_str: str) -> Dict:
    """Convert the raw diversity string into a cleaned dictionary of diversity metric results.

    Args:
        diversity_str (str): Raw string containing multiple diversity MetricResult blocks.

    Returns:
        Dict: A dictionary with keys as diversity metric names and values as parsed MetricResult dicts.

    This function extracts and parses all relevant diversity metric sections from the raw string,
    returning a structured dictionary ready for visualization.
    """
    diversity_metrics = [
        "element_diversity",
        "space_group_diversity",
        "site_number_diversity",
        "physical_size_diversity",
    ]
    compilied_diversity_metrics = {
        metric: extract_metric_result_section(diversity_str, metric)
        for metric in diversity_metrics
    }
    return compilied_diversity_metrics


def create_plotting_values(cleaned_dict: Dict, n_strucutres: int = 1):
    """Prepare metric values and corresponding bounds for plotting the radar chart.

    Args:
        cleaned_dict (Dict): Dictionary containing cleaned diversity metrics.
        n_strucutres (int, optional): Number of structures used to define bounds. Defaults to 1.

    Returns:
        list: A list of tuples each containing:
            - Label (str): Name of the metric.
            - Value (float): Metric value to plot.
            - Bounds (tuple): (min, max) range for normalization.

    This function extracts primary metric values from the cleaned dictionary and defines
    appropriate bounds for normalization on the radar plot.
    """
    default_metrics = [
        "element_diversity",
        "space_group_diversity",
        "site_number_diversity",
        "physical_size_diversity",
    ]
    primary_metric_labels = [
        cleaned_dict[metric]["primary_metric"] for metric in default_metrics
    ]
    primary_metric_values = [
        cleaned_dict[default_metrics[metric]]["metrics"][primary_metric_labels[metric]]
        for metric in range(len(default_metrics))
    ]

    packing_factor_value = cleaned_dict["physical_size_diversity"]["metrics"][
        "packing_factor_diversity_shannon_entropy"
    ]
    primary_metric_values.append(packing_factor_value)

    bounds = [
        (0, 1),
        (0, 1),
        (1, n_strucutres),
        (1, np.log(n_strucutres)),
        (1, np.log(n_strucutres)),
    ]

    plotting_headings = [
        "Element Coverage",
        "Space Group Coverage",
        "Site Number Diversity",
        "Physical Diversity",
        "Packing_factor",
    ]

    plotting_metrics_collected = list(
        zip(plotting_headings, primary_metric_values, bounds)
    )

    return plotting_metrics_collected


def plot_radar_zipped(
    metric_tuples,
    title: str = "Radar Plot",
    show: bool = True,
):
    """Plot a radar chart for diversity metrics given their labels, values, and bounds.

    Args:
        metric_tuples (list of tuple): List of tuples where each tuple contains:
            - label (str): Name of the metric.
            - value (float): Raw numeric metric value.
            - (min, max) (tuple of float): Normalization bounds for the metric.
        title (str, optional): Title for the radar chart. Defaults to "Radar Plot".
        show (bool, optional): Whether to display the plot immediately. Defaults to True.

    Returns:
        tuple: Matplotlib Figure and Axes objects of the radar plot.

    Raises:
        ValueError: If less than 3 metric tuples are provided or if any bounds are invalid.

    This function normalizes the metric values according to their bounds and plots them on a radar chart,
    providing a visual representation of diversity metrics from the benchmark.
    """
    if len(metric_tuples) < 3:
        raise ValueError(
            "Radar plot needs at least 3 (label, value, (min,max)) tuples."
        )

    # Unpack
    labels = [label for label, _, _ in metric_tuples]
    raw_values = np.asarray([val for _, val, _ in metric_tuples], dtype=float)
    bounds_array = np.asarray([b for _, _, b in metric_tuples], dtype=float)

    # Validate bounds
    mins = bounds_array[:, 0]
    maxs = bounds_array[:, 1]
    if np.any(maxs <= mins):
        bad = np.where(maxs <= mins)[0].tolist()
        raise ValueError(f"Invalid bounds (min >= max) for indices: {bad}")

    # Normalize to [0, 1]
    normalized = np.clip((raw_values - mins) / (maxs - mins), 0.0, 1.0)

    # Close polygon + angles
    normalized = np.append(normalized, normalized[0])
    n_axes = len(labels)
    angles = np.linspace(0.0, 2.0 * np.pi, n_axes, endpoint=False)
    angles = np.append(angles, angles[0])

    # Figure/Axes (polar)
    fig, ax = plt.subplots(subplot_kw={"polar": True}, figsize=(6.5, 6.5))
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)

    # Tick labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=9)
    ax.set_ylim(0.0, 1.0)

    # Plot polygon
    ax.plot(angles, normalized, linewidth=2)
    ax.fill(angles, normalized, alpha=0.15)
    ax.set_title(title, pad=18, fontsize=13)

    if show:
        plt.show()
    return fig, ax


def main():
    """Main function to parse arguments, read benchmark data, extract diversity metrics,
    and generate a radar plot visualization.

    This script reads a diversity benchmark JSON file, extracts the relevant diversity
    metrics, processes them for plotting, and displays a radar chart summarizing the diversity.

    Raises:
        SystemExit: If the results file is not found or is not a file.
    """
    parser = argparse.ArgumentParser(
        description="Run visualization to create a Radar Plot of Diversity Metrics."
    )
    parser.add_argument(
        "--results_file",
        required=True,
        type=Path,
        help="Path to JSON of benchmark results",
    )
    args = parser.parse_args()

    results_path: Path = args.results_file
    if not results_path.exists():
        parser.error(f"File not found: {results_path}")
    if not results_path.is_file():
        parser.error(f"Not a file: {results_path}")

    benchmark_dict = read_json(results_path)
    n_structures = benchmark_dict["run_info"]["n_structures"]

    diversity_dict = extract_diversity_results(benchmark_dict)
    cleaned_dict = convert_diversity_string_to_cleaned_dict(diversity_dict)

    plotting_metrics_zipped = create_plotting_values(cleaned_dict, n_structures)
    plot_radar_zipped(plotting_metrics_zipped, title="Diversity Radar Plot", show=True)


if __name__ == "__main__":
    main()
