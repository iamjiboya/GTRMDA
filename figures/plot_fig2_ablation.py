"""Generate the single-column GTRMDA ablation figure."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from font_utils import publication_serif


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "fig2_ablation_data.csv"
OUTPUT_STEM = ROOT / "fig2"

DATASETS = ["HMDD v3.2", "HMDD v4.0", "miR2Disease"]
VARIANTS = [
    "GTRMDA",
    "w/o relation semantics",
    "w/o in-context demos",
    "w/o test-time verifier",
    "w/o counterfactual loss",
    "single trajectory (B=1)",
]

# Colors follow the blue, green, and orange vocabulary of Figure 1.
COLORS = {
    "HMDD v3.2": "#78A9D1",
    "HMDD v4.0": "#76C893",
    "miR2Disease": "#F2A166",
}
EDGES = {
    "HMDD v3.2": "#2F618A",
    "HMDD v4.0": "#347A55",
    "miR2Disease": "#A95F2F",
}
MARKERS = {"HMDD v3.2": "o", "HMDD v4.0": "s", "miR2Disease": "D"}
OFFSETS = {"HMDD v3.2": -0.16, "HMDD v4.0": 0.0, "miR2Disease": 0.16}
PALE_GREEN = "#DDF1E5"
GRID = "#D8DEE3"
TEXT = "#202020"


def configure_style() -> None:
    family = publication_serif(ROOT)

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = [family]
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = family
    plt.rcParams["mathtext.it"] = f"{family}:italic"
    plt.rcParams["mathtext.bf"] = f"{family}:bold"
    plt.rcParams["mathtext.bfit"] = f"{family}:italic:bold"
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 8.5
    plt.rcParams["axes.titlesize"] = 10
    plt.rcParams["axes.labelsize"] = 9
    plt.rcParams["xtick.labelsize"] = 8.3
    plt.rcParams["ytick.labelsize"] = 8.1
    plt.rcParams["axes.linewidth"] = 0.85
    plt.rcParams["savefig.facecolor"] = "white"


def draw_metric_panel(ax, panel, metric, data, show_method_labels) -> None:
    y = list(range(len(VARIANTS)))
    ax.axhspan(-0.46, 0.46, color=PALE_GREEN, zorder=0)

    for dataset in DATASETS:
        subset = data[data["dataset"] == dataset].set_index("variant").loc[VARIANTS]
        values = subset[metric].to_numpy()
        errors = subset[f"{metric}_sd"].to_numpy()
        positions = [value + OFFSETS[dataset] for value in y]
        ax.errorbar(
            values,
            positions,
            xerr=errors,
            fmt=MARKERS[dataset],
            ms=4.5,
            color=COLORS[dataset],
            markeredgecolor=EDGES[dataset],
            markeredgewidth=0.65,
            ecolor=EDGES[dataset],
            elinewidth=0.75,
            capsize=1.5,
            zorder=3,
        )
    if metric == "auc":
        ax.set_xlim(0.855, 0.972)
        ax.set_xticks([0.86, 0.90, 0.94])
        title = "AUC"
    else:
        ax.set_xlim(0.575, 0.785)
        ax.set_xticks([0.60, 0.68, 0.75])
        title = "AUPR"

    ax.set_ylim(len(VARIANTS) - 0.5, -0.5)
    ax.set_yticks(y)
    if show_method_labels:
        ax.set_yticklabels(VARIANTS)
        ax.get_yticklabels()[0].set_fontweight("bold")
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.set_title(title, pad=3, fontweight="bold", color=TEXT)
    ax.grid(axis="x", color=GRID, linewidth=0.5, alpha=0.9, zorder=0)
    ax.tick_params(axis="both", length=2.1, width=0.6, colors=TEXT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8C969C")
    ax.spines["bottom"].set_color("#8C969C")
    ax.text(
        -0.12,
        1.05,
        panel,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
        color=TEXT,
    )


def main() -> None:
    configure_style()
    data = pd.read_csv(DATA_FILE)

    # 89 x 75 mm: compact horizontal two-panel layout for one AAAI column.
    fig, axes = plt.subplots(1, 2, figsize=(3.50, 2.95), sharey=True)
    draw_metric_panel(axes[0], "a", "auc", data, show_method_labels=True)
    draw_metric_panel(axes[1], "b", "aupr", data, show_method_labels=False)

    legend = [
        Line2D(
            [0], [0], marker=MARKERS[name], linestyle="none", markersize=4.7,
            markerfacecolor=COLORS[name], markeredgecolor=EDGES[name], label=name,
        )
        for name in DATASETS
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.995),
        ncol=3,
        frameon=False,
        fontsize=7.2,
        handletextpad=0.25,
        columnspacing=0.60,
    )
    fig.supxlabel("Performance", x=0.72, y=0.055, fontsize=9)
    fig.subplots_adjust(left=0.47, right=0.985, top=0.81, bottom=0.20, wspace=0.22)

    metadata = {
        "Title": "Ablation analysis of GTRMDA",
        "Subject": "",
    }
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), dpi=1200, metadata=metadata)
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"), dpi=1200)
    fig.savefig(
        OUTPUT_STEM.with_suffix(".tiff"),
        dpi=1200,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(ROOT / "fig2_preview.png", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
