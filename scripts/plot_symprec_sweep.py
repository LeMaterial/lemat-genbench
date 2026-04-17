"""
Plot bar charts from a symprec sweep results JSON.

Produces one subplot per symprec value, each with 4 bars:
overall validity, overall novelty, overall uniqueness, and space group diversity.
Bar colors are consistent across subplots.

Usage:
    uv run python scripts/plot_symprec_sweep.py results_final/diffcsp_sweep_symprec_sweep_20260417_230813.json
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Metrics to extract and their regex patterns within final_scores
METRICS = {
    "Validity": r"'overall_validity_ratio':\s*([\d.]+)",
    "Novelty": r"'novelty_score':\s*([\d.]+)",
    "Uniqueness": r"'uniqueness_score':\s*([\d.]+)",
    "SG Diversity": r"'space_group_diversity':\s*np\.float64\(([\d.]+)\)",
}

# Consistent colours across all subplots
COLORS = {
    "Validity": "#4C72B0",
    "Novelty": "#DD8452",
    "Uniqueness": "#55A868",
    "SG Diversity": "#C44E52",
}


def extract_score(result_str: str, pattern: str) -> float | None:
    """Pull a numeric value from the stringified BenchmarkResult.

    Parameters
    ----------
    result_str : str
        String representation of a ``BenchmarkResult``.
    pattern : str
        Regex with one capture group for the numeric value.

    Returns
    -------
    float | None
        Extracted value, or ``None`` if not found.
    """
    m = re.search(pattern, result_str)
    return float(m.group(1)) if m else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot symprec sweep results.")
    parser.add_argument("results_json", type=Path, help="Path to sweep results JSON")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output image path (default: same name as input with .png)",
    )
    args = parser.parse_args()

    with open(args.results_json) as f:
        data = json.load(f)

    symprec_results = data["results_by_symprec"]
    symprec_keys = list(symprec_results.keys())
    n = len(symprec_keys)

    metric_names = list(METRICS.keys())

    # Build score matrix: rows = symprec values, cols = metrics
    scores = np.full((n, len(metric_names)), np.nan)
    for i, sp_key in enumerate(symprec_keys):
        entry = symprec_results[sp_key]
        bench_map = {
            "Validity": entry.get("validity", ""),
            "Novelty": entry.get("novelty", ""),
            "Uniqueness": entry.get("uniqueness", ""),
            "SG Diversity": entry.get("diversity", ""),
        }
        for j, metric in enumerate(metric_names):
            val = extract_score(bench_map[metric], METRICS[metric])
            if val is not None:
                scores[i, j] = val

    # Plot
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    x = np.arange(len(metric_names))
    bar_width = 0.6

    for i, (ax, sp_key) in enumerate(zip(axes, symprec_keys)):
        bars = ax.bar(
            x,
            scores[i],
            width=bar_width,
            color=[COLORS[m] for m in metric_names],
            edgecolor="white",
            linewidth=0.5,
        )
        # Value labels on bars
        for bar, val in zip(bars, scores[i]):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.set_title(f"symprec {sp_key}", fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Score")
    fig.suptitle(
        f"Benchmark scores across symprec values — {data['run_info']['run_name']}",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()

    output_path = args.output or args.results_json.with_suffix(".png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
