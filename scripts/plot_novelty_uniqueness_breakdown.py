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

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

SKIP_NAMES = {"alexandria", "oqmd", "mp"}

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
    font_scale: float = 1.0,
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
    total_evaluated : int | None
        Total structures that passed validity (evaluated by this benchmark).
    total_submitted : int | None
        Total structures submitted before validity filtering.
    show_labels : bool
        If True, draw count and percentage text inside bars.
    font_scale : float
        Multiplier for all font sizes (default 1.0 for standalone plots).
    """
    labels = [m[0] for m in metrics]
    values = [extract_int(result_str, m[1]) or 0 for m in metrics]

    s = font_scale  # shorthand

    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=COLORS, edgecolor="white", linewidth=0.5)

    # X-tick labels
    category_map = {"Novelty": "Novel", "Uniqueness": "Unique"}
    category = category_map.get(title, title)
    tick_labels = [f"{lbl}\n{category}" for lbl in labels]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=35.2 * s)
    ax.set_ylabel("")
    ax.set_title(title, fontsize=40 * s, fontweight="bold")
    ax.set_ylim(0, Y_MAX)
    ax.set_yticks(np.arange(0, Y_MAX + 1, 500))
    ax.tick_params(axis="y", labelsize=38.4 * s)
    ax.grid(axis="y", alpha=0.3)

    # Reference lines
    legend_handles = []
    if total_submitted is not None:
        line_sub = ax.axhline(
            total_submitted, color="black", linestyle="-", linewidth=1.2,
        )
        legend_handles.append((line_sub, "Submitted"))
    if total_evaluated is not None:
        line_eval = ax.axhline(
            total_evaluated, color="black", linestyle="--", linewidth=1.2,
        )
        legend_handles.append((line_eval, "Valid"))

    if legend_handles:
        ax.legend(
            [h[0] for h in legend_handles],
            [h[1] for h in legend_handles],
            loc="upper right",
            bbox_to_anchor=(1.0, 1.0),
            fontsize=45 * s,
            framealpha=0.95,
            edgecolor="0.7",
        )


def _parse_result(results_json: Path) -> dict | None:
    """Parse a results JSON and return data needed for plotting.

    Supports two formats:
      - ``data["results"]["novelty"]`` as a stringified BenchmarkResult
      - ``data["novelty"]`` as a plain dict with metric keys

    Returns
    -------
    dict | None
        Dict with keys ``novelty_str``, ``uniqueness_str``, ``novelty_total``,
        ``uniqueness_total``, ``total_submitted``, ``run_name``.
        Returns ``None`` if the run should be skipped.
    """
    with open(results_json) as f:
        data = json.load(f)

    # Try format 1: data["results"]["novelty"] as stringified BenchmarkResult
    results = data.get("results", data.get("results_by_symprec", {}))
    novelty_str = str(results.get("novelty", ""))
    uniqueness_str = str(results.get("uniqueness", ""))

    # Try format 2: data["novelty"] / data["uniqueness"] as plain dicts
    if not novelty_str.strip() and isinstance(data.get("novelty"), dict):
        novelty_str = str(data["novelty"])
    if not uniqueness_str.strip() and isinstance(data.get("uniqueness"), dict):
        uniqueness_str = str(data["uniqueness"])

    novelty_total = extract_total(novelty_str, "novelty")
    uniqueness_total = extract_total(uniqueness_str, "uniqueness")

    total_submitted = data.get("run_info", {}).get("n_structures")
    if total_submitted is None:
        total_submitted = data.get("validity_filtering", {}).get("total_input_structures")

    run_name = data.get("run_info", {}).get("run_name", results_json.stem)
    if run_name in SKIP_NAMES:
        print(f"Skipping {run_name} (excluded)")
        return None
    run_name = run_name.replace("_2500", "").replace("2500", "")
    run_name = DISPLAY_NAMES.get(run_name, run_name)

    return {
        "novelty_str": novelty_str,
        "uniqueness_str": uniqueness_str,
        "novelty_total": novelty_total,
        "uniqueness_total": uniqueness_total,
        "total_submitted": total_submitted,
        "run_name": run_name,
    }


def plot_single_panel(
    ax1: plt.Axes,
    ax2: plt.Axes,
    parsed: dict,
    show_labels: bool = True,
    font_scale: float = 1.0,
) -> None:
    """Draw novelty + uniqueness breakdown on a pair of axes.

    Parameters
    ----------
    ax1 : plt.Axes
        Axes for the novelty subplot.
    ax2 : plt.Axes
        Axes for the uniqueness subplot.
    parsed : dict
        Output of ``_parse_result``.
    show_labels : bool
        If True, draw count and percentage text inside bars.
    font_scale : float
        Multiplier for font sizes (< 1.0 for multi-panel figures).
    """
    plot_breakdown(
        ax1, parsed["novelty_str"], NOVELTY_METRICS, "Novelty",
        parsed["novelty_total"], parsed["total_submitted"],
        show_labels=show_labels, font_scale=font_scale,
    )
    plot_breakdown(
        ax2, parsed["uniqueness_str"], UNIQUENESS_METRICS, "Uniqueness",
        parsed["uniqueness_total"], parsed["total_submitted"],
        show_labels=show_labels, font_scale=font_scale,
    )


def plot_single_file(results_json: Path, output_path: Path, show_labels: bool = True) -> None:
    """Generate a standalone breakdown plot for a single results JSON file.

    Parameters
    ----------
    results_json : Path
        Path to the benchmark results JSON.
    output_path : Path
        Where to save the output PNG.
    show_labels : bool
        If True, draw count and percentage text inside bars.
    """
    parsed = _parse_result(results_json)
    if parsed is None:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), sharey=True)

    plot_single_panel(ax1, ax2, parsed, show_labels=show_labels)

    fig.suptitle(parsed["run_name"], fontsize=50, fontweight="bold")
    fig.tight_layout()

    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def plot_assembled_figure(
    json_files: list[Path],
    output_path: Path,
    ncols: int = 3,
    show_labels: bool = True,
) -> None:
    """Generate a single multi-panel figure from multiple results JSONs.

    Each panel shows novelty + uniqueness side by side, with a run-name title.
    Panel labels (a), (b), ... are added in the top-left corner.

    Parameters
    ----------
    json_files : list[Path]
        Results JSON files to include.
    output_path : Path
        Where to save the assembled PNG.
    ncols : int
        Number of columns in the panel grid.
    show_labels : bool
        If True, draw count and percentage text inside bars.
    """
    import string

    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    # Parse all files, filtering out skipped runs
    panels = []
    for jf in json_files:
        try:
            parsed = _parse_result(jf)
            if parsed is not None:
                panels.append(parsed)
        except Exception as e:
            print(f"Skipping {jf.name}: {e}")

    if not panels:
        print("No valid panels to plot")
        return

    n = len(panels)
    nrows = (n + ncols - 1) // ncols

    panel_w = 12.5
    panel_h = 6
    fig_w = panel_w * ncols
    fig_h = panel_h * nrows

    fig = plt.figure(figsize=(fig_w, fig_h))

    # Outer grid: one cell per panel block, with wide gaps between blocks
    outer_gs = GridSpec(
        nrows, ncols,
        figure=fig,
        wspace=0.2,
        hspace=0.5,
    )

    font_scale = 0.5

    for idx, parsed in enumerate(panels):
        row = idx // ncols
        col = idx % ncols

        # Inner grid: 2 columns (novelty + uniqueness) tight together
        inner_gs = GridSpecFromSubplotSpec(
            1, 2,
            subplot_spec=outer_gs[row, col],
            wspace=0.08,
        )

        ax1 = fig.add_subplot(inner_gs[0, 0])
        ax2 = fig.add_subplot(inner_gs[0, 1], sharey=ax1)

        plot_single_panel(ax1, ax2, parsed, show_labels=show_labels, font_scale=font_scale)

        # Hide y-tick labels on the uniqueness (right) axis
        ax2.tick_params(axis="y", labelleft=False)

        # "Novelty" / "Uniqueness" subtitles above each axis
        ax1.set_title("Novelty", fontsize=21, fontweight="bold", pad=8)
        ax2.set_title("Uniqueness", fontsize=21, fontweight="bold", pad=8)

        # Run name centered above the pair, using fig.text after layout
        # We use the outer_gs bounding box via ax positions after drawing
        fig.canvas.draw()
        pos1 = ax1.get_position()
        pos2 = ax2.get_position()
        mid_x = (pos1.x0 + pos2.x1) / 2
        top_y = max(pos1.y1, pos2.y1)
        fig.text(
            mid_x, top_y + 0.01,
            parsed["run_name"],
            ha="center", va="bottom",
            fontsize=22, fontweight="bold",
        )

        # Panel label
        if idx < len(string.ascii_lowercase):
            label = f"({string.ascii_lowercase[idx]})"
            fig.text(
                pos1.x0 - 0.01, top_y + 0.01,
                label,
                ha="right", va="bottom",
                fontsize=28, fontweight="bold",
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved assembled figure ({n} panels, {nrows}x{ncols}) to {output_path}")


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
        help="Output image path",
    )
    parser.add_argument(
        "--no-labels", action="store_true",
        help="Hide count/percentage text on bars",
    )
    parser.add_argument(
        "--ncols", type=int, default=3,
        help="Number of columns for assembled figure (default: 3)",
    )
    args = parser.parse_args()

    show_labels = not args.no_labels

    if args.input.is_dir():
        json_files = sorted(args.input.glob("*.json"))
        if not json_files:
            print(f"No JSON files found in {args.input}")
            return

        figures_dir = Path(__file__).parent.parent / "figures"
        output_path = args.output or figures_dir / "novelty_and_uniqueness_breakdown.png"

        plot_assembled_figure(
            json_files, output_path,
            ncols=args.ncols, show_labels=show_labels,
        )
    else:
        output_path = args.output or args.input.with_name(
            args.input.stem + "_breakdown.png"
        )
        plot_single_file(args.input, output_path, show_labels=show_labels)


if __name__ == "__main__":
    main()
