"""
Plot fine-grained breakdown of novelty and uniqueness metrics.

Produces two bar charts side by side:
  - Novelty:    total novel, novel composition, novel spacegroup, novel structure only
  - Uniqueness: total unique, unique composition, unique spacegroup, unique structure only

Usage:
    uv run python scripts/plot_novelty_uniqueness_breakdown.py results_final/diffcsp_2500_all_validity_20260417_221544.json
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# (label, regex pattern) for each breakdown bar
NOVELTY_METRICS = [
    ("Total Novel", r"'novel_structures_count':\s*(\d+)"),
    ("Novel\nComposition", r"'novel_composition_count':\s*(\d+)"),
    ("Novel\nSpacegroup", r"'novel_spacegroup_count':\s*(\d+)"),
    ("Novel\nStructure Only", r"'novel_structure_only_count':\s*(\d+)"),
]

UNIQUENESS_METRICS = [
    ("Total Unique", r"'unique_structures_count':\s*(\d+)"),
    ("Unique\nComposition", r"'unique_composition_count':\s*(\d+)"),
    ("Unique\nSpacegroup", r"'unique_spacegroup_count':\s*(\d+)"),
    ("Unique\nStructure Only", r"'unique_structure_only_count':\s*(\d+)"),
]

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def extract_int(result_str: str, pattern: str) -> int | None:
    """Pull an integer value from the stringified BenchmarkResult.

    Parameters
    ----------
    result_str : str
        String representation of a ``BenchmarkResult``.
    pattern : str
        Regex with one capture group for the integer value.

    Returns
    -------
    int | None
        Extracted value, or ``None`` if not found.
    """
    m = re.search(pattern, result_str)
    return int(m.group(1)) if m else None


def extract_total(result_str: str, bench_type: str) -> int | None:
    """Extract total_structures_evaluated from the result string.

    Parameters
    ----------
    result_str : str
        String representation of a ``BenchmarkResult``.
    bench_type : str
        Unused, kept for interface consistency.

    Returns
    -------
    int | None
        Total structures evaluated.
    """
    m = re.search(r"'total_structures_evaluated':\s*(\d+)", result_str)
    return int(m.group(1)) if m else None


def plot_breakdown(
    ax: plt.Axes,
    result_str: str,
    metrics: list[tuple[str, str]],
    title: str,
    total: int | None,
) -> None:
    """Draw a bar chart for one benchmark's breakdown.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to draw on.
    result_str : str
        String representation of the ``BenchmarkResult``.
    metrics : list[tuple[str, str]]
        List of ``(label, regex_pattern)`` pairs.
    title : str
        Subplot title.
    total : int | None
        Total structures evaluated (for annotation).
    """
    labels = [m[0] for m in metrics]
    values = [extract_int(result_str, m[1]) or 0 for m in metrics]

    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=COLORS, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, values):
        pct = f"({100 * val / total:.1f}%)" if total else ""
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            f"{val}\n{pct}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.15)
    ax.grid(axis="y", alpha=0.3)

    if total is not None:
        ax.axhline(total, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.text(
            len(labels) - 0.5,
            total + max(values) * 0.02,
            f"total evaluated: {total}",
            ha="right",
            va="bottom",
            fontsize=8,
            color="grey",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot novelty/uniqueness breakdown from benchmark results."
    )
    parser.add_argument("results_json", type=Path, help="Path to benchmark results JSON")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output image path (default: <input>_breakdown.png)",
    )
    args = parser.parse_args()

    with open(args.results_json) as f:
        data = json.load(f)

    results = data.get("results", data.get("results_by_symprec", {}))
    novelty_str = str(results.get("novelty", ""))
    uniqueness_str = str(results.get("uniqueness", ""))

    novelty_total = extract_total(novelty_str, "novelty")
    uniqueness_total = extract_total(uniqueness_str, "uniqueness")

    run_name = data.get("run_info", {}).get("run_name", args.results_json.stem)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    plot_breakdown(ax1, novelty_str, NOVELTY_METRICS, "Novelty Breakdown", novelty_total)
    plot_breakdown(ax2, uniqueness_str, UNIQUENESS_METRICS, "Uniqueness Breakdown", uniqueness_total)

    fig.suptitle(f"Novelty & Uniqueness Breakdown — {run_name}", fontsize=13, fontweight="bold")
    fig.tight_layout()

    output_path = args.output or args.results_json.with_name(
        args.results_json.stem + "_breakdown.png"
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
