"""
Plot framework novelty breakdown.

Supports two input formats:
  - Standalone explore_framework_novelty.py output (``counts`` / ``total_structures``)
  - Benchmark results JSON with framework counts in the novelty result string

Produces a bar chart for each results JSON showing the three classification
categories: existing anon + known SG, existing anon + novel SG, novel anon.

Usage:
    uv run python scripts/plot_framework_novelty.py results_final/diffcsp_pp_framework_validity_20260419_172021.json
    uv run python scripts/plot_framework_novelty.py results_final/framework_novelty/
    uv run python scripts/plot_framework_novelty.py results_final/
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

# Categories in display order
CATEGORIES = [
    ("existing_anon_known_sg", "Known Anon.\nKnown SG"),
    ("existing_anon_novel_sg", "Known Anon.\nNovel SG"),
    ("novel_anon", "Novel Anon.\nFormula"),
]

COLORS = ["#40619C", "#CB7446", "#497259"]

DISPLAY_NAMES = {
    "ADiT": "ADiT",
    "aflow": "AFLOW",
    "alexandria": "Alexandria",
    "cifs_chemeleon2_alex_mp": "CeLLM-2",
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


FRAMEWORK_METRICS = {
    "existing_anon_known_sg": r"'framework_existing_anon_known_sg':\s*(\d+)",
    "existing_anon_novel_sg": r"'framework_existing_anon_novel_sg':\s*(\d+)",
    "novel_anon": r"'framework_novel_anon':\s*(\d+)",
}

SKIP_NAMES = {"alexandria", "oqmd", "mp"}


def _extract_int(text: str, pattern: str) -> int | None:
    """Pull an integer from a stringified BenchmarkResult."""
    m = re.search(pattern, text)
    return int(m.group(1)) if m else None


def _extract_framework_data(data: dict) -> tuple[dict[str, int], int, str]:
    """Extract framework counts, total, and run name from either JSON format.

    Parameters
    ----------
    data : dict
        Parsed JSON data.

    Returns
    -------
    tuple[dict[str, int], int, str]
        ``(counts, total_evaluated, run_name)``.

    Raises
    ------
    ValueError
        If no framework novelty data found in the file.
    """
    # Format 1: standalone explore_framework_novelty.py output
    if "counts" in data and "total_structures" in data:
        return data["counts"], data["total_structures"], ""

    # Format 2: benchmark results JSON with stringified novelty result
    results = data.get("results", data.get("results_by_symprec", {}))
    novelty_str = str(results.get("novelty", ""))

    counts = {}
    for key, pattern in FRAMEWORK_METRICS.items():
        val = _extract_int(novelty_str, pattern)
        if val is not None:
            counts[key] = val

    if not counts:
        raise ValueError("No framework novelty data found")

    # Total evaluated from the novelty result
    total_m = re.search(r"'total_structures_evaluated':\s*(\d+)", novelty_str)
    total = int(total_m.group(1)) if total_m else sum(counts.values())

    run_name = data.get("run_info", {}).get("run_name", "")
    return counts, total, run_name


def plot_single_file(
    results_json: Path, output_path: Path, show_labels: bool = True
) -> None:
    """Generate a framework novelty plot for a single results JSON.

    Parameters
    ----------
    results_json : Path
        Path to a framework novelty results JSON.
    output_path : Path
        Where to save the output PNG.
    show_labels : bool
        If True, draw count and percentage text on/above bars.
    """
    with open(results_json) as f:
        data = json.load(f)

    counts, total, run_name = _extract_framework_data(data)

    # Derive run name from filename if not in data
    if not run_name:
        run_name = results_json.stem.replace("_framework_novelty", "")
    run_name = re.sub(r"_(?:validity|all_validity)_\d{8}_\d{6}$", "", run_name)
    run_name = run_name.replace("_framework", "")
    run_name = run_name.replace("_2500", "").replace("2500", "")
    run_name = DISPLAY_NAMES.get(run_name, run_name)

    if run_name.lower() in SKIP_NAMES:
        print(f"Skipping {run_name} (excluded)")
        return

    labels = [cat[1] for cat in CATEGORIES]
    values = [counts.get(cat[0], 0) for cat in CATEGORIES]

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.bar(x, values, color=COLORS, edgecolor="white", linewidth=0.5)

    threshold = Y_MAX * INSIDE_LABEL_THRESHOLD

    # Reference lines
    total_submitted = data.get("run_info", {}).get("n_structures")
    if total_submitted is None:
        total_submitted = data.get("validity_filtering", {}).get("total_input_structures")

    legend_handles = []
    if total_submitted is not None and total_submitted != total:
        line_sub = ax.axhline(total_submitted, color="black", linestyle="-", linewidth=1.2)
        legend_handles.append((line_sub, f"Submitted: {total_submitted}"))
    line_eval = ax.axhline(total, color="black", linestyle="--", linewidth=1.2)
    legend_handles.append((line_eval, f"Valid: {total}"))

    if show_labels:
        label_height = Y_MAX * 0.10

        ref_lines = [total]
        if total_submitted is not None:
            ref_lines.append(total_submitted)

        for bar, val in zip(bars, values):
            pct = f"({100 * val / total:.1f}%)"
            label_text = f"{val}\n{pct}"

            if val >= threshold:
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
                label_bottom = bar.get_height() + Y_MAX * 0.01
                label_top = label_bottom + label_height

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
                    fontweight="bold",
                    color="black",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=32)
    ax.set_ylim(0, Y_MAX)
    ax.set_yticks(np.arange(0, Y_MAX + 1, 500))
    ax.tick_params(axis="y", labelsize=32)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Framework Novelty", fontsize=40, fontweight="bold")

    if legend_handles:
        ax.legend(
            [h[0] for h in legend_handles],
            [h[1] for h in legend_handles],
            loc="upper right",
            fontsize=30,
            framealpha=0.95,
            edgecolor="0.7",
        )

    fig.suptitle(run_name, fontsize=50, fontweight="bold")
    fig.tight_layout()

    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot framework novelty breakdown from results JSON."
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
