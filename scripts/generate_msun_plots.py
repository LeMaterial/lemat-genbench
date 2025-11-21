import matplotlib.pyplot as plt
from matplotlib import rcParams

# Global styling: prefer Inter (if installed), with clean sans-serif fallbacks
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Inter", "DejaVu Sans", "Arial", "Helvetica"]

# Data for models
models_data = {
    # Inference times are for 2.5k samples
    "SymmCD": {"params": 61_010_350, "inference_time": 1400, "gpu": "RTX 8000"},
    "Crystalformer": {"params": 4_840_295, "inference_time": 100, "gpu": "V100"},
    "ADiT": {"params": 32_000_000, "inference_time": 300, "gpu": "V100"},
    "PLaID++": {"params": 8_000_000_000, "inference_time": 345, "gpu": "H100"},
    "MatterGen": {"params": 46_800_000, "inference_time": 18_000, "gpu": "V100"},
    "DiffCSP": {"params": 12_300_000, "inference_time": 2335, "gpu": "A40"},
    "DiffCSP++": {"params": 12_300_000, "inference_time": 6150, "gpu": "A40"},
    "WyFormer": {"params": 150_000, "inference_time": 0.625, "gpu": "RTX 6000"},
    "LLaMat2": {"params": 7_000_000_000, "inference_time": None, "gpu": None},
    "LLaMat3": {"params": 8_000_000_000, "inference_time": None, "gpu": None},
}

# GPU color mapping
gpu_colors = {
    "RTX 8000": "#D4AF37",  # Gold
    "V100": "#87CEEB",  # Sky blue
    "H100": "#6495ED",  # Cornflower blue
    "A40": "#90EE90",  # Light green
    "RTX 6000": "#FF8C00",  # Dark orange
}

# Toggle whether inference-time plots use a logarithmic x-axis
INFERENCE_USE_LOG_SCALE = False

# MSUN% from LeMat-Bulk
msun_lemat_bulk = {
    "MatterGen": 15.0,
    "PLaID++": 7.6,
    "WyFormer": 1.9,
    "ADiT": 1.0,
    "Crystalformer": 3.1,
    "DiffCSP": 8.5,
    "DiffCSP++": 5.0,
    "LLaMat2": 2.1,
    "LLaMat3": 0.2,
    "SymmCD": 2.4,
}

# MSUN% from MP-20
msun_mp20 = {
    "MatterGen": 22.7,
    "PLaID++": 30.2,  # Note: table shows "PLaID" but we'll use PLaID++
    "WyFormer": 6.1,
    "ADiT": 2.8,
    "Crystalformer": 10.4,
    "DiffCSP": 18.3,
    "DiffCSP++": 13.6,
    "LLaMat2": 10.7,
    "LLaMat3": 0.9,
    "SymmCD": 8.9,
}


def find_pareto_frontier(points):
    """Find Pareto frontier for minimizing x and maximizing y"""
    # Sort by x (ascending)
    sorted_points = sorted(points, key=lambda p: p[0])
    pareto = []
    max_y = -float("inf")

    for point in sorted_points:
        if point[1] > max_y:
            pareto.append(point)
            max_y = point[1]

    return pareto


def plot_params_vs_msun(msun_data, title_suffix, filename):
    """Plot Model Parameters vs MSUN%"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Prepare data
    models = []
    params_list = []
    msun_list = []

    for model in msun_data.keys():
        if model in models_data:
            models.append(model)
            params_list.append(models_data[model]["params"])
            msun_list.append(msun_data[model])

    # Find Pareto frontier (minimize params, maximize MSUN)
    points = list(zip(params_list, msun_list))
    pareto_points = find_pareto_frontier(points)
    pareto_params = [p[0] for p in pareto_points]
    pareto_msun = [p[1] for p in pareto_points]

    # Identify which models are on Pareto frontier
    pareto_models = []
    other_models = []

    for i, model in enumerate(models):
        if (params_list[i], msun_list[i]) in pareto_points:
            pareto_models.append(i)
        else:
            other_models.append(i)

    # Plot non-Pareto models
    if other_models:
        ax.scatter(
            [params_list[i] for i in other_models],
            [msun_list[i] for i in other_models],
            s=150,
            color="#CCCCCC",
            alpha=0.7,
            label="Other models",
            edgecolors="#666666",
            linewidth=1.5,
            zorder=2,
        )

    # Plot Pareto models
    if pareto_models:
        ax.scatter(
            [params_list[i] for i in pareto_models],
            [msun_list[i] for i in pareto_models],
            s=150,
            color="#7B68EE",
            alpha=0.8,
            label="Pareto frontier models",
            edgecolors="#4B0082",
            linewidth=1.5,
            zorder=3,
        )

    # Draw Pareto frontier line
    if len(pareto_points) > 1:
        pareto_sorted = sorted(pareto_points, key=lambda p: p[0])
        pareto_params_sorted = [p[0] for p in pareto_sorted]
        pareto_msun_sorted = [p[1] for p in pareto_sorted]
        ax.plot(
            pareto_params_sorted,
            pareto_msun_sorted,
            color="#7B68EE",
            linewidth=2.5,
            alpha=0.7,
            zorder=1,
        )

    # Add labels for all models
    for i, model in enumerate(models):
        # Default placement: above/below the point
        offset_x = 0
        offset_y = (
            12.0
            if model not in ["LLaMat2", "Crystalformer", "PLaID++", "DiffCSP++"]
            else -22.0
        )
        ha = "center"

        # Special positioning tweaks to improve readability
        if model == "MatterGen":
            # Place label to the right of the point instead of above
            offset_x = 18
            offset_y = -3
            ha = "left"

        is_pareto = (params_list[i], msun_list[i]) in pareto_points
        fontweight = "bold" if is_pareto else "normal"

        ax.annotate(
            model,
            (params_list[i], msun_list[i]),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            ha=ha,
            fontsize=12,
            fontweight=fontweight,
        )

    # Styling
    ax.set_xlabel("Parameters (log scale)", fontsize=14)
    ax.set_ylabel("MSUN%", fontsize=14)
    ax.set_title(
        f"Model Parameters vs M.S.U.N. ({title_suffix})",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xscale("log")
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.legend(fontsize=12, loc="best", framealpha=0.9)

    # Set background color
    ax.set_facecolor("#F8F8F8")
    fig.patch.set_facecolor("white")

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close()


def plot_inference_vs_msun(msun_data, title_suffix, filename, log_x=False):
    """Plot Inference Time vs MSUN%"""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Prepare data (exclude models without inference time)
    models = []
    inference_list = []
    msun_list = []
    gpu_list = []

    for model in msun_data.keys():
        if model in models_data and models_data[model]["inference_time"] is not None:
            models.append(model)
            inference_list.append(models_data[model]["inference_time"])
            msun_list.append(msun_data[model])
            gpu_list.append(models_data[model]["gpu"])

    # Find Pareto frontier (minimize inference time, maximize MSUN)
    points = list(zip(inference_list, msun_list))
    pareto_points = find_pareto_frontier(points)
    pareto_inference = [p[0] for p in pareto_points]
    pareto_msun = [p[1] for p in pareto_points]

    # Identify which models are on Pareto frontier
    pareto_models = []
    other_models = []

    for i, model in enumerate(models):
        if (inference_list[i], msun_list[i]) in pareto_points:
            pareto_models.append(i)
        else:
            other_models.append(i)

    # Plot models with GPU color coding
    # Group by GPU for legend (collect scatter objects for clean legend)
    gpu_scatter_handles = {}

    for i, model in enumerate(models):
        is_pareto = (inference_list[i], msun_list[i]) in pareto_points
        gpu = gpu_list[i]
        color = gpu_colors.get(gpu, "#CCCCCC")

        if is_pareto:
            # Pareto frontier with purple outline similar to parameter plots
            scatter = ax.scatter(
                inference_list[i],
                msun_list[i],
                s=200,
                color=color,
                alpha=0.9,
                edgecolors="#4B0082",
                linewidth=1.5,
                zorder=3,
            )
        else:
            # Other models with thin border
            scatter = ax.scatter(
                inference_list[i],
                msun_list[i],
                s=150,
                color=color,
                alpha=0.7,
                edgecolors="#666666",
                linewidth=1.5,
                zorder=2,
            )

        # Store one scatter object per GPU for legend (without borders)
        if gpu and gpu not in gpu_scatter_handles:
            # Create a clean marker for legend without borders
            gpu_scatter_handles[gpu] = ax.scatter(
                [], [], s=100, color=color, alpha=0.8, edgecolors="none"
            )

    # Draw Pareto frontier line
    if len(pareto_points) > 1:
        pareto_sorted = sorted(pareto_points, key=lambda p: p[0])
        pareto_inference_sorted = [p[0] for p in pareto_sorted]
        pareto_msun_sorted = [p[1] for p in pareto_sorted]
        ax.plot(
            pareto_inference_sorted,
            pareto_msun_sorted,
            color="#7B68EE",
            linewidth=2.5,
            alpha=0.7,
            zorder=1,
        )

    # Add labels for all models with manual positioning
    for i, model in enumerate(models):
        offset_y = 8.0
        offset_x = 0
        ha = "center"
        va = "bottom"

        # Check if model is on Pareto frontier
        is_pareto = (inference_list[i], msun_list[i]) in pareto_points
        fontweight = "bold" if is_pareto else "normal"

        # Manual adjustments for specific models to avoid overlap
        if model == "PLaID++":
            offset_y = 8.5
            ha = "center"
        elif model == "MatterGen":
            offset_y = 8.5
            ha = "center"
        elif model == "DiffCSP":
            offset_y = -22.5
            ha = "center"
        elif model == "DiffCSP++":
            offset_y = -8.5
            va = "top"
        elif model == "WyFormer":
            offset_y = -8.5
            va = "top"
        elif model == "Crystalformer":
            offset_x = 14
            offset_y = -6
            ha = "left"
        elif model == "SymmCD":
            offset_y = -8.5
            va = "top"
        elif model == "ADiT":
            offset_y = -8.5
            va = "top"

        ax.annotate(
            model,
            (inference_list[i], msun_list[i]),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=12,
            fontweight=fontweight,
        )

    # Styling
    if log_x:
        ax.set_xscale("log")
        x_label = "Inference time for 2.5k samples in seconds (log scale)"
    else:
        x_label = "Inference time for 2.5k samples (seconds)"

    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel("MSUN%", fontsize=14)
    ax.set_title(
        f"Inference Time vs M.S.U.N. ({title_suffix})",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    # Create legend with only GPU types (no borders, no pareto/other distinction)
    legend_handles = []
    legend_labels = []
    for gpu in sorted(gpu_scatter_handles.keys()):
        legend_handles.append(gpu_scatter_handles[gpu])
        legend_labels.append(gpu)

    if legend_handles:
        ax.legend(
            handles=legend_handles,
            labels=legend_labels,
            fontsize=12,
            loc="best",
            framealpha=0.9,
            title="GPU",
        )

    # Set background color
    ax.set_facecolor("#F8F8F8")
    fig.patch.set_facecolor("white")

    # Add some padding to x and y limits
    x_min = min(inference_list)
    x_max = max(inference_list)
    y_margin = (max(msun_list) - min(msun_list)) * 0.1

    if log_x:
        # Use multiplicative padding when in log space to keep edge labels visible
        ax.set_xlim(x_min * 0.6, x_max * 1.6)
    else:
        x_margin = (x_max - x_min) * 0.05
        ax.set_xlim(x_min - x_margin, x_max + x_margin)

    ax.set_ylim(min(msun_list) - y_margin, max(msun_list) + y_margin)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close()


def main():
    print("Generating MSUN plots...")

    # LeMat-Bulk plots
    plot_params_vs_msun(
        msun_lemat_bulk,
        "LeMat-Bulk",
        "/Users/axu/lemat-genbench/model_params_vs_msun_lemat_bulk.png",
    )
    plot_inference_vs_msun(
        msun_lemat_bulk,
        "LeMat-Bulk",
        "/Users/axu/lemat-genbench/inference_time_vs_msun_lemat_bulk.png",
        log_x=INFERENCE_USE_LOG_SCALE,
    )

    # MP-20 plots
    plot_params_vs_msun(
        msun_mp20,
        "MP-20",
        "/Users/axu/lemat-genbench/model_params_vs_msun_mp20.png",
    )
    plot_inference_vs_msun(
        msun_mp20,
        "MP-20",
        "/Users/axu/lemat-genbench/inference_time_vs_msun_mp20.png",
        log_x=INFERENCE_USE_LOG_SCALE,
    )

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
