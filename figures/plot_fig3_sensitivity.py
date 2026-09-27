"""Generate the single-column GTRMDA parameter-sensitivity figure."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from font_utils import publication_serif


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "fig3_sensitivity_data.csv"
OUTPUT_STEM = ROOT / "fig3"

BLUE = "#78A9D1"
BLUE_EDGE = "#2F618A"
ORANGE = "#F2A166"
ORANGE_EDGE = "#A95F2F"
PALE_GREEN = "#DDF1E5"
TEAL = "#8FC9B5"
TEAL_EDGE = "#3D806E"
MAUVE = "#C9A4C5"
MAUVE_EDGE = "#805678"
GRID = "#D8DEE3"
TEXT = "#202020"

PANELS = [
    {
        "key": "demonstrations",
        "title": "Demonstrations",
        "xlabel": r"Demonstrations $K$",
        "tag": "a",
        "band": (6.7, 9.3),
        "ticks": [0, 2, 4, 8, 16],
        "ticklabels": ["0", "2", "4", "8", "16"],
    },
    {
        "key": "budget",
        "title": "Test-time Budget",
        "xlabel": r"Budget $B$",
        "tag": "b",
        "band": (6.7, 9.3),
        "ticks": [1, 2, 4, 8, 16],
        "ticklabels": ["1", "2", "4", "8", "16"],
    },
    {
        "key": "hops",
        "title": "Subgraph Depth",
        "xlabel": "Subgraph hops",
        "tag": "c",
        "band": (2.72, 3.28),
        "ticks": [1, 2, 3, 4, 5],
        "ticklabels": ["1", "2", "3", "4", "5"],
    },
    {
        "key": "cf_weight",
        "title": "CF Loss Weight",
        "xlabel": r"$\lambda_{\mathrm{cf}}$",
        "tag": "d",
        "band": (0.16, 0.24),
        "ticks": [0.0, 0.1, 0.2, 0.3, 0.5],
        "ticklabels": ["0", ".1", ".2", ".3", ".5"],
    },
]


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
    plt.rcParams["ytick.labelsize"] = 8.3
    plt.rcParams["axes.linewidth"] = 0.85
    plt.rcParams["savefig.facecolor"] = "white"


def draw_panel(ax, spec, data, show_ylabel) -> None:
    subset = data[data["parameter"] == spec["key"]].sort_values("value")
    x = subset["value"].to_numpy()

    ax.axvspan(*spec["band"], color=PALE_GREEN, zorder=0)
    ax.errorbar(
        x, subset["auc"], yerr=subset["auc_sd"], color=BLUE_EDGE,
        marker="o", markerfacecolor=BLUE, markeredgecolor=BLUE_EDGE,
        markersize=4.7, linewidth=1.35, elinewidth=0.8, capsize=1.6,
        label="AUC", zorder=3,
    )
    ax.errorbar(
        x, subset["aupr"], yerr=subset["aupr_sd"], color=ORANGE_EDGE,
        marker="s", markerfacecolor=ORANGE, markeredgecolor=ORANGE_EDGE,
        markersize=4.4, linewidth=1.35, elinewidth=0.8, capsize=1.6,
        label="AUPR", zorder=3,
    )
    ax.errorbar(
        x, subset["mrr"], yerr=subset["mrr_sd"], color=TEAL_EDGE,
        marker="D", markerfacecolor=TEAL, markeredgecolor=TEAL_EDGE,
        markersize=4.0, linewidth=1.25, linestyle="--",
        elinewidth=0.75, capsize=1.5, label="MRR", zorder=3,
    )
    ax.errorbar(
        x, subset["hits10"], yerr=subset["hits10_sd"], color=MAUVE_EDGE,
        marker="^", markerfacecolor=MAUVE, markeredgecolor=MAUVE_EDGE,
        markersize=4.3, linewidth=1.25, linestyle=(0, (1.2, 1.2)),
        elinewidth=0.75, capsize=1.5, label="Hits@10", zorder=3,
    )

    ax.set_ylim(0.49, 0.975)
    ax.set_yticks([0.50, 0.60, 0.70, 0.80, 0.90])
    ax.set_xticks(spec["ticks"])
    ax.set_xticklabels(spec["ticklabels"])
    ax.set_title(spec["title"], pad=3, fontweight="bold", color=TEXT)
    ax.set_xlabel(spec["xlabel"], labelpad=2)
    if show_ylabel:
        ax.set_ylabel("Performance", labelpad=3)
    else:
        ax.tick_params(axis="y", labelleft=False)

    ax.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.9, zorder=0)
    ax.tick_params(axis="both", length=2.2, width=0.65, colors=TEXT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8C969C")
    ax.spines["bottom"].set_color("#8C969C")
    ax.text(
        -0.18, 1.05, spec["tag"], transform=ax.transAxes,
        fontsize=10, fontweight="bold", va="top", ha="left", color=TEXT,
    )


def main() -> None:
    configure_style()
    data = pd.read_csv(DATA_FILE)

    # 89 x 97 mm: compact two-by-two layout for one AAAI column.
    fig, axes = plt.subplots(2, 2, figsize=(3.50, 3.80), sharey=True)
    for index, (ax, spec) in enumerate(zip(axes.flat, PANELS)):
        draw_panel(ax, spec, data, show_ylabel=index % 2 == 0)

    legend = [
        Line2D([0], [0], color=BLUE_EDGE, marker="o", markersize=4.7,
               markerfacecolor=BLUE, label="AUC"),
        Line2D([0], [0], color=ORANGE_EDGE, marker="s", markersize=4.4,
               markerfacecolor=ORANGE, label="AUPR"),
        Line2D([0], [0], color=TEAL_EDGE, marker="D", markersize=4.0,
               markerfacecolor=TEAL, linestyle="--", label="MRR"),
        Line2D([0], [0], color=MAUVE_EDGE, marker="^", markersize=4.3,
               markerfacecolor=MAUVE, linestyle=(0, (1.2, 1.2)), label="Hits@10"),
    ]
    fig.legend(
        handles=legend, loc="upper center", bbox_to_anchor=(0.56, 0.995),
        ncol=4, frameon=False, fontsize=7.2, handlelength=1.35,
        handletextpad=0.3, columnspacing=0.65,
    )
    fig.subplots_adjust(left=0.145, right=0.99, top=0.875, bottom=0.115,
                        wspace=0.14, hspace=0.46)

    metadata = {
        "Title": "Parameter sensitivity analysis of GTRMDA",
        "Subject": "",
    }
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), dpi=1200, metadata=metadata)
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"), dpi=1200)
    fig.savefig(
        OUTPUT_STEM.with_suffix(".tiff"), dpi=1200,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(ROOT / "fig3_preview.png", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
