"""
Plot framework novelty breakdown.

Shows a "Total Novel" bar followed by the framework classification of those
novel structures (known anon + known SG, known anon + novel SG, novel anon).

Framework counts are derived from the per-structure CSV (cross-tabulating
novelty_category and framework_category for novel-only structures) and cached
back into the JSON for future runs.

Supports multiple input formats:
  - Standalone explore_framework_novelty.py output (``counts`` / ``total_structures``)
  - Benchmark results JSON with per-structure CSV companion
  - Benchmark results JSON with pre-cached ``novel_framework_*`` keys

When given a directory, produces a single assembled multi-panel figure.

Usage:
    uv run python scripts/plot_framework_novelty.py results_final/rebuttal_novelty_framework/aflow_vnf_comprehensive_multi_mlip_hull_20260420_102907.json
    uv run python scripts/plot_framework_novelty.py results_final/rebuttal_novelty_framework/
"""

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Fixed y-axis height so all plots are directly comparable
Y_MAX = 2650

# Bar categories: total novel + framework breakdown of novel structures
CATEGORIES = [
    ("novel_total", "Total\nNovel"),
    ("existing_anon_known_sg", "Known Anon.\nKnown SG"),
    ("existing_anon_novel_sg", "Known Anon.\nNovel SG"),
    ("novel_anon", "Novel Anon.\nFormula"),
]

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

SKIP_NAMES = {"alexandria", "oqmd", "mp"}

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

FRAMEWORK_REGEX = {
    "existing_anon_known_sg": r"'framework_existing_anon_known_sg':\s*(\d+)",
    "existing_anon_novel_sg": r"'framework_existing_anon_novel_sg':\s*(\d+)",
    "novel_anon": r"'framework_novel_anon':\s*(\d+)",
}

NOVEL_FRAMEWORK_KEYS = [
    "novel_framework_existing_anon_known_sg",
    "novel_framework_existing_anon_novel_sg",
    "novel_framework_novel_anon",
]


def _extract_int(text: str, pattern: str) -> int | None:
    """Pull an integer from a stringified BenchmarkResult."""
    m = re.search(pattern, text)
    return int(m.group(1)) if m else None


def _compute_novel_framework_counts(csv_path: Path) -> dict[str, int]:
    """Cross-tabulate framework_category for novel-only structures from CSV.

    Parameters
    ----------
    csv_path : Path
        Path to the per-structure CSV.

    Returns
    -------
    dict[str, int]
        Framework category counts for novel structures only.
    """
    counts: dict[str, int] = {
        "existing_anon_known_sg": 0,
        "existing_anon_novel_sg": 0,
        "novel_anon": 0,
    }
    non_novel = {"not_novel", "invalid"}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["novelty_category"] not in non_novel:
                fw = row["framework_category"]
                if fw in counts:
                    counts[fw] += 1
    return counts


def _parse_result(results_json: Path) -> dict | None:
    """Parse a results JSON and return data needed for plotting.

    If novel-only framework counts aren't cached in the JSON yet, computes
    them from the companion per-structure CSV and writes them back.

    Returns
    -------
    dict | None
        Dict with keys ``counts``, ``novel_count``, ``total``,
        ``total_submitted``, ``run_name``.
        Returns ``None`` if the run should be skipped.
    """
    with open(results_json) as f:
        data = json.load(f)

    novel_count = None
    novel_fw_counts = None
    total = None

    # --- Extract novelty dict (try multiple locations) ---
    novelty_dict = data.get("novelty", {})
    if not isinstance(novelty_dict, dict):
        novelty_dict = {}
    if not novelty_dict:
        results = data.get("results", data.get("results_by_symprec", {}))
        novelty_str = str(results.get("novelty", ""))
        # Build a pseudo-dict from regex extraction
        if novelty_str.strip():
            novelty_dict = {}
            for key in ["novel_structures_count", "total_structures_evaluated"]:
                m = re.search(rf"'{key}':\s*(\d+)", novelty_str)
                if m:
                    novelty_dict[key] = int(m.group(1))
            for key in NOVEL_FRAMEWORK_KEYS:
                m = re.search(rf"'{key}':\s*(\d+)", novelty_str)
                if m:
                    novelty_dict[key] = int(m.group(1))

    # --- Check for pre-cached novel framework counts ---
    if all(k in novelty_dict for k in NOVEL_FRAMEWORK_KEYS):
        novel_fw_counts = {
            "existing_anon_known_sg": novelty_dict["novel_framework_existing_anon_known_sg"],
            "existing_anon_novel_sg": novelty_dict["novel_framework_existing_anon_novel_sg"],
            "novel_anon": novelty_dict["novel_framework_novel_anon"],
        }
        novel_count = novelty_dict.get("novel_structures_count")
        total = novelty_dict.get("total_structures_evaluated")

    # --- Compute from CSV if not cached ---
    if novel_fw_counts is None:
        csv_name = data.get("run_info", {}).get("per_structure_csv")
        if csv_name:
            csv_path = results_json.parent / csv_name
        else:
            # Try convention: same stem + _per_structure.csv
            csv_path = results_json.parent / (results_json.stem + "_per_structure.csv")

        if csv_path.exists():
            novel_fw_counts = _compute_novel_framework_counts(csv_path)
            novel_count = novelty_dict.get("novel_structures_count")
            total = novelty_dict.get("total_structures_evaluated")

            # Cache back into the JSON
            if isinstance(data.get("novelty"), dict):
                data["novelty"]["novel_framework_existing_anon_known_sg"] = novel_fw_counts["existing_anon_known_sg"]
                data["novelty"]["novel_framework_existing_anon_novel_sg"] = novel_fw_counts["existing_anon_novel_sg"]
                data["novelty"]["novel_framework_novel_anon"] = novel_fw_counts["novel_anon"]
                with open(results_json, "w") as f:
                    json.dump(data, f, indent=2)
                print(f"Cached novel framework counts into {results_json.name}")

    if novel_fw_counts is None:
        raise ValueError("No framework novelty data found (no cached counts or CSV)")

    # Fall back for novel_count if not in novelty dict
    if novel_count is None:
        novel_count = sum(novel_fw_counts.values())

    total_submitted = data.get("run_info", {}).get("n_structures")
    if total_submitted is None:
        total_submitted = data.get("validity_filtering", {}).get("total_input_structures")
    if total is None:
        total = data.get("validity_filtering", {}).get("valid_structures")

    run_name = data.get("run_info", {}).get("run_name", "")
    if not run_name:
        run_name = results_json.stem.replace("_framework_novelty", "")
    run_name = re.sub(r"_(?:validity|all_validity)_\d{8}_\d{6}$", "", run_name)
    run_name = run_name.replace("_framework", "")
    run_name = run_name.replace("_2500", "").replace("2500", "")
    run_name = run_name.replace("_vnf_comprehensive_multi_mlip_hull", "")
    if run_name in SKIP_NAMES:
        print(f"Skipping {run_name} (excluded)")
        return None
    run_name = DISPLAY_NAMES.get(run_name, run_name)

    # Build the 4-bar counts dict
    counts = {
        "novel_total": novel_count,
        **novel_fw_counts,
    }

    return {
        "counts": counts,
        "total": total,
        "total_submitted": total_submitted,
        "run_name": run_name,
    }


def plot_framework_bars(
    ax: plt.Axes,
    parsed: dict,
    font_scale: float = 1.0,
) -> None:
    """Draw a framework novelty bar chart on the given axes.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to draw on.
    parsed : dict
        Output of ``_parse_result``.
    font_scale : float
        Multiplier for all font sizes.
    """
    s = font_scale
    counts = parsed["counts"]
    total = parsed["total"]
    total_submitted = parsed["total_submitted"]

    labels = [cat[1] for cat in CATEGORIES]
    values = [counts.get(cat[0], 0) for cat in CATEGORIES]

    x = np.arange(len(labels))
    ax.bar(x, values, color=COLORS, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=35.2 * s)
    ax.set_ylabel("")
    ax.set_title("Framework Novelty", fontsize=40 * s, fontweight="bold")
    ax.set_ylim(0, Y_MAX)
    ax.set_yticks(np.arange(0, Y_MAX + 1, 500))
    ax.tick_params(axis="y", labelsize=38.4 * s)
    ax.grid(axis="y", alpha=0.3)

    # Reference lines
    legend_handles = []
    if total_submitted is not None and total_submitted != total:
        line_sub = ax.axhline(total_submitted, color="black", linestyle="-", linewidth=1.2)
        legend_handles.append((line_sub, "Submitted"))
    if total is not None:
        line_eval = ax.axhline(total, color="black", linestyle="--", linewidth=1.2)
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


def plot_single_file(results_json: Path, output_path: Path) -> None:
    """Generate a standalone framework novelty plot for a single results JSON.

    Parameters
    ----------
    results_json : Path
        Path to a framework novelty results JSON.
    output_path : Path
        Where to save the output PNG.
    """
    parsed = _parse_result(results_json)
    if parsed is None:
        return

    fig, ax = plt.subplots(figsize=(10, 10))
    plot_framework_bars(ax, parsed)

    fig.suptitle(parsed["run_name"], fontsize=50, fontweight="bold")
    fig.tight_layout()

    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def plot_assembled_figure(
    json_files: list[Path],
    output_path: Path,
    ncols: int = 3,
) -> None:
    """Generate a single multi-panel figure from multiple results JSONs.

    Each panel shows a framework novelty bar chart with a run-name title.
    Panel labels (a), (b), ... are added in the top-left corner.

    Parameters
    ----------
    json_files : list[Path]
        Results JSON files to include.
    output_path : Path
        Where to save the assembled PNG.
    ncols : int
        Number of columns in the panel grid.
    """
    import string

    from matplotlib.gridspec import GridSpec

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

    panel_w = 10.0625
    panel_h = 6
    fig_w = panel_w * ncols
    fig_h = panel_h * nrows

    fig = plt.figure(figsize=(fig_w, fig_h))

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

        ax = fig.add_subplot(outer_gs[row, col])
        plot_framework_bars(ax, parsed, font_scale=font_scale)

        ax.set_title("Framework Novelty", fontsize=21, fontweight="bold", pad=8)

        fig.canvas.draw()
        pos = ax.get_position()
        mid_x = (pos.x0 + pos.x1) / 2
        top_y = pos.y1
        fig.text(
            mid_x, top_y + 0.01,
            parsed["run_name"],
            ha="center", va="bottom",
            fontsize=22, fontweight="bold",
        )

        if idx < len(string.ascii_lowercase):
            label = f"({string.ascii_lowercase[idx]})"
            fig.text(
                pos.x0 - 0.01, top_y + 0.01,
                label,
                ha="right", va="bottom",
                fontsize=28, fontweight="bold",
            )

    # Turn off unused axes
    for idx in range(n, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        fig.add_subplot(outer_gs[row, col]).axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved assembled figure ({n} panels, {nrows}x{ncols}) to {output_path}")


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
        help="Output image path",
    )
    parser.add_argument(
        "--ncols", type=int, default=3,
        help="Number of columns for assembled figure (default: 3)",
    )
    args = parser.parse_args()

    if args.input.is_dir():
        json_files = sorted(args.input.glob("*.json"))
        if not json_files:
            print(f"No JSON files found in {args.input}")
            return

        figures_dir = Path(__file__).parent.parent / "figures"
        output_path = args.output or figures_dir / "framework_novelty_breakdown.png"

        plot_assembled_figure(json_files, output_path, ncols=args.ncols)
    else:
        output_path = args.output or args.input.with_name(
            args.input.stem + "_breakdown.png"
        )
        plot_single_file(args.input, output_path)


if __name__ == "__main__":
    main()
