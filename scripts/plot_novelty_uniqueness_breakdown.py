"""
Plot fine-grained breakdown of novelty and uniqueness metrics.

Produces two bar charts side by side:
  - Novelty:    total novel, novel composition, novel spacegroup, novel structure only
  - Uniqueness: total unique, unique composition, unique spacegroup, unique structure only

Usage:
    uv run python scripts/plot_novelty_uniqueness_breakdown.py results_final/diffcsp_2500_all_validity_20260417_221544.json
    uv run python scripts/plot_novelty_uniqueness_breakdown.py results_final/
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Fixed y-axis height so all plots are directly comparable
Y_MAX = 2650

# Threshold: bars shorter than this fraction of Y_MAX get labels above
INSIDE_LABEL_THRESHOLD = 0.12

# (label, regex pattern) for each breakdown bar
NOVELTY_METRICS = [
    ("Total", r"'novel_structures_count':\s*(\d+)"),
    ("Comp.", r"'novel_composition_count':\s*(\d+)"),
    ("SG", r"'novel_spacegroup_count':\s*(\d+)"),
    ("Struct.", r"'novel_structure_only_count':\s*(\d+)"),
]

UNIQUENESS_METRICS = [
    ("Total", r"'unique_structures_count':\s*(\d+)"),
    ("Comp.", r"'unique_composition_count':\s*(\d+)"),
    ("SG", r"'unique_spacegroup_count':\s*(\d+)"),
    ("Struct.", r"'unique_structure_only_count':\s*(\d+)"),
]

COLORS = ["#40619C", "#CB7446", "#497259", "#B14245"]

# Display names for run names that contain underscores or non-standard casing
DISPLAY_NAMES = {
    "ADiT": "ADiT",
    "aflow": "AFLOW",
    "alexandria": "Alexandria",
    "crystal_gfn": "Crystal-GFN",
    "crystalformer": "CrystalFormer",
    "diffcsp": "DiffCSP",
    "diffcsp_pp": "DiffCSP++",
    "llamat2": "LLaMat-2",
    "llamat3": "LLaMat-3",
    "mattergen": "MatterGen",
    "mp": "Materials Project",
    "oqmd": "OQMD",
    "plaid_pp": "PLAID++",
    "symmcd": "SymmCD",
    "wang2021_stable_cifs": "Wang 2021",
    "wyformer_diffcsppp": "WyFormer-DiffCSP++",
    "wyformer_diffcsppp_dft": "WyFormer-DiffCSP++ (DFT)",
    "Randomly-Enumerated": "Randomly Enumerated",
}


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
    total_evaluated: int | None,
    total_submitted: int | None,
    show_labels: bool = True,
) -> None:
    """Draw a bar chart for one benchmark's breakdown.

    Labels are placed inside bars (white text near bar top) to avoid collisions
    with the reference lines. For very short bars the label is placed just above.

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
    total_evaluated : int | None
        Total structures that passed validity (evaluated by this benchmark).
    total_submitted : int | None
        Total structures submitted before validity filtering.
    show_labels : bool
        If True, draw count and percentage text on/above bars.
    """
    labels = [m[0] for m in metrics]
    values = [extract_int(result_str, m[1]) or 0 for m in metrics]

    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=COLORS, edgecolor="white", linewidth=0.5)

    ref_total = total_evaluated or 1
    threshold = Y_MAX * INSIDE_LABEL_THRESHOLD

    if show_labels:
        # Approximate height of a two-line label in data coords (generous to avoid clipping)
        label_height = Y_MAX * 0.10

        # Reference lines that labels must not overlap
        ref_lines = []
        if total_submitted is not None:
            ref_lines.append(total_submitted)
        if total_evaluated is not None:
            ref_lines.append(total_evaluated)

        for bar, val in zip(bars, values):
            pct = f"({100 * val / ref_total:.1f}%)"
            label_text = f"{val}\n{pct}"

            if val >= threshold:
                # Place inside the bar, near the top
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() - Y_MAX * 0.01,
                    label_text,
                    ha="center",
                    va="top",
                    fontsize=20,
                    fontweight="bold",
                    color="white",
                )
            else:
                # Short bar — place just above, but check for line collisions
                label_bottom = bar.get_height() + Y_MAX * 0.01
                label_top = label_bottom + label_height

                # If the label would overlap a reference line, push it above
                for line_y in ref_lines:
                    if label_bottom < line_y < label_top:
                        label_bottom = line_y + Y_MAX * 0.01

                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    label_bottom,
                    label_text,
                    ha="center",
                    va="bottom",
                    fontsize=20,
                    color="black",
                    fontweight="bold",
                )

    # X-tick labels: append the subplot category as a second line (e.g. "Total\nNovel")
    category_map = {"Novelty": "Novel", "Uniqueness": "Unique"}
    category = category_map.get(title, title)
    tick_labels = [f"{lbl}\n{category}" for lbl in labels]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=32)
    ax.set_ylabel("")
    ax.set_title(title, fontsize=40, fontweight="bold")
    ax.set_ylim(0, Y_MAX)
    ax.set_yticks(np.arange(0, Y_MAX + 1, 500))
    ax.tick_params(axis="y", labelsize=32)
    ax.grid(axis="y", alpha=0.3)

    # Reference lines
    legend_handles = []
    if total_submitted is not None:
        line_sub = ax.axhline(
            total_submitted, color="black", linestyle="-", linewidth=1.2,
        )
        legend_handles.append((line_sub, f"Submitted: {total_submitted}"))
    if total_evaluated is not None:
        line_eval = ax.axhline(
            total_evaluated, color="black", linestyle="--", linewidth=1.2,
        )
        legend_handles.append((line_eval, f"Valid: {total_evaluated}"))

    if legend_handles:
        ax.legend(
            [h[0] for h in legend_handles],
            [h[1] for h in legend_handles],
            loc="upper right",
            bbox_to_anchor=(1.0, 1.0),
            fontsize=30,
            framealpha=0.95,
            edgecolor="0.7",
        )


def plot_single_file(results_json: Path, output_path: Path, show_labels: bool = True) -> None:
    """Generate a breakdown plot for a single results JSON file.

    Parameters
    ----------
    results_json : Path
        Path to the benchmark results JSON.
    output_path : Path
        Where to save the output PNG.
    show_labels : bool
        If True, draw count and percentage text on/above bars.
    """
    with open(results_json) as f:
        data = json.load(f)

    results = data.get("results", data.get("results_by_symprec", {}))
    novelty_str = str(results.get("novelty", ""))
    uniqueness_str = str(results.get("uniqueness", ""))

    novelty_total = extract_total(novelty_str, "novelty")
    uniqueness_total = extract_total(uniqueness_str, "uniqueness")

    # Total submitted before validity filtering
    total_submitted = data.get("run_info", {}).get("n_structures")
    if total_submitted is None:
        total_submitted = data.get("validity_filtering", {}).get("total_input_structures")

    SKIP_NAMES = {"alexandria", "oqmd", "mp"}

    run_name = data.get("run_info", {}).get("run_name", results_json.stem)
    if run_name in SKIP_NAMES:
        print(f"Skipping {run_name} (excluded)")
        return
    run_name = run_name.replace("_2500", "").replace("2500", "")
    run_name = DISPLAY_NAMES.get(run_name, run_name)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

    plot_breakdown(ax1, novelty_str, NOVELTY_METRICS, "Novelty", novelty_total, total_submitted, show_labels=show_labels)
    plot_breakdown(ax2, uniqueness_str, UNIQUENESS_METRICS, "Uniqueness", uniqueness_total, total_submitted, show_labels=show_labels)

    fig.suptitle(f"{run_name}", fontsize=50, fontweight="bold")
    fig.tight_layout()

    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot novelty/uniqueness breakdown from benchmark results."
    )
    parser.add_argument(
        "input", type=Path,
        help="Path to a single results JSON or a directory of JSON files",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output image path (only used when input is a single file)",
    )
    parser.add_argument(
        "--no-labels", action="store_true",
        help="Hide count/percentage text on bars",
    )
    args = parser.parse_args()

    show_labels = not args.no_labels

    if args.input.is_dir():
        json_files = sorted(args.input.glob("*.json"))
        if not json_files:
            print(f"No JSON files found in {args.input}")
            return

        output_dir = args.input.parent / f"summary_plots_{args.input.name}"
        output_dir.mkdir(exist_ok=True)

        for json_file in json_files:
            output_path = output_dir / (json_file.stem + "_breakdown.png")
            try:
                plot_single_file(json_file, output_path, show_labels=show_labels)
            except Exception as e:
                print(f"Skipping {json_file.name}: {e}")

        print(f"\nAll plots saved to {output_dir}")
    else:
        output_path = args.output or args.input.with_name(
            args.input.stem + "_breakdown.png"
        )
        plot_single_file(args.input, output_path, show_labels=show_labels)


if __name__ == "__main__":
    main()
