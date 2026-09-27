"""Reproduce the reference Figure 5 layout for two GTRMDA disease cases."""

from pathlib import Path
from math import cos, pi, sin

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch

from font_utils import publication_serif


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "fig5_contribution_data.csv"

BLUE = "#78A9D1"
BLUE_EDGE = "#2F618A"
ORANGE = "#F2A166"
ORANGE_EDGE = "#A95F2F"
TEAL = "#8FC9B5"
TEAL_EDGE = "#3D806E"
MAUVE = "#C9A4C5"
MAUVE_EDGE = "#805678"
PALE_BLUE = "#D9F0F3"
PALE_PEACH = "#FBE8DD"
PALE_CORE = "#C8DFC1"
TEXT = "#202020"
EDGE = "#9AA3A8"

CASES = {
    "Breast Neoplasms": {
        "mirna": "hsa-miR-21-5p",
        "disease": "Breast Neoplasms",
        "core": "PI3K-Akt /\nApoptosis",
        "core_size": 7.1,
        "score": 0.948,
        "left_edge": 0.241,
        "right_edge": 0.219,
        "neighbor_values": [0.184, 0.156, 0.219],
    },
    "Alzheimer's Disease": {
        "mirna": "hsa-miR-146a-5p",
        "disease": "Alzheimer's Disease",
        "core": "NF-kB signaling /\nNeuroinflammation",
        "core_size": 6.3,
        "score": 0.941,
        "left_edge": 0.252,
        "right_edge": 0.236,
        "neighbor_values": [0.173, 0.149, 0.236],
    },
}


def configure_style() -> None:
    family = publication_serif(ROOT)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [family],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8.5,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.3,
        "ytick.labelsize": 8.3,
        "legend.fontsize": 7.5,
        "savefig.facecolor": "white",
    })


def contribution(data, case, node) -> float:
    value = data[(data["case"] == case) & (data["node"] == node)]["contribution"]
    return float(value.iloc[0])


def wrap_entity(label: str) -> str:
    replacements = {
        "hsa-miR-21-5p": "hsa-miR-21-5p",
        "hsa-miR-146a-5p": "hsa-miR-146a-5p",
        "Breast Neoplasms": "Breast\nNeoplasms",
        "Alzheimer's Disease": "Alzheimer's\nDisease",
    }
    return replacements.get(label, label)


def draw_subgraph(
    ax,
    center,
    radius,
    facecolor,
    bordercolor,
    main_color,
    main_edge,
    main_label,
    main_value,
    neighbor_colors,
    side,
    neighbor_values=None,
) -> None:
    background = Circle(
        center, radius, facecolor=facecolor, edgecolor=bordercolor,
        linewidth=0.9, linestyle=(0, (4, 2.5)), zorder=0,
    )
    ax.add_patch(background)

    angles = [38, 78, 148, 190, 238, 302]
    neighbor_positions = []
    for index, angle in enumerate(angles):
        rad = angle * pi / 180
        position = (
            center[0] + radius * 0.73 * cos(rad),
            center[1] + radius * 0.73 * sin(rad),
        )
        neighbor_positions.append(position)
        ax.plot(
            [center[0], position[0]], [center[1], position[1]],
            color=EDGE, linewidth=0.46, zorder=1,
        )
        node = Circle(
            position, radius * 0.125,
            facecolor=neighbor_colors[index % len(neighbor_colors)],
            edgecolor=TEXT, linewidth=0.45, zorder=2,
        )
        ax.add_patch(node)

    main = Circle(
        center, radius * 0.145, facecolor=main_color,
        edgecolor=main_edge, linewidth=0.65, zorder=4,
    )
    ax.add_patch(main)

    if side == "left":
        ax.text(
            center[0], center[1] - radius * 0.34,
            f"{wrap_entity(main_label)}\n({main_value:.3f})",
            ha="center", va="center", fontsize=7.0,
            fontweight="bold", color=TEXT, linespacing=0.95, zorder=5,
        )
    else:
        ax.text(
            center[0], center[1] - radius * 0.34,
            f"{wrap_entity(main_label)}\n({main_value:.3f})",
            ha="center", va="center", fontsize=6.9,
            fontweight="bold", color=TEXT, linespacing=0.92, zorder=5,
        )
        values = neighbor_values or []
        value_positions = [
            (center[0] + radius * 0.60, center[1] + radius * 0.45),
            (center[0] + radius * 0.60, center[1] + radius * 0.05),
            (center[0] + radius * 0.60, center[1] - radius * 0.70),
        ]
        for value, position in zip(values, value_positions):
            ax.text(
                position[0], position[1], f"({value:.3f})",
                ha="left", va="center", fontsize=6.5, color=TEXT, zorder=5,
            )


def draw_case(ax, case, spec, data, tag) -> None:
    left_center = (0.37, 0.46)
    core_center = (1.10, 0.46)
    right_center = (1.83, 0.46)
    sub_radius = 0.285
    core_radius = 0.245

    mirna_value = contribution(data, case, spec["mirna"])
    disease_value = contribution(data, case, spec["disease"])

    ax.text(0.37, 0.900, "Subgraph(m_g_m)", ha="center", va="center",
            fontsize=8.2, color=TEXT)
    ax.text(1.83, 0.900, "Subgraph(d_g_d)", ha="center", va="center",
            fontsize=8.2, color=TEXT)
    ax.text(1.10, 0.850, f"Score: {spec['score']:.3f}", ha="center",
            va="center", fontsize=8.2, color=TEXT)

    draw_subgraph(
        ax, left_center, sub_radius, PALE_BLUE, "#2E9FB1",
        BLUE, BLUE_EDGE, spec["mirna"], mirna_value,
        [PALE_CORE, TEAL, ORANGE], "left",
    )
    draw_subgraph(
        ax, right_center, sub_radius, PALE_PEACH, "#E49A70",
        MAUVE, MAUVE_EDGE, spec["disease"], disease_value,
        [PALE_CORE, ORANGE, TEAL], "right", spec["neighbor_values"],
    )

    core = Circle(
        core_center, core_radius, facecolor=PALE_CORE,
        edgecolor=TEXT, linewidth=0.55, zorder=2,
    )
    ax.add_patch(core)
    ax.text(
        core_center[0], core_center[1], spec["core"],
        ha="center", va="center", fontsize=spec["core_size"],
        color=TEXT, linespacing=0.95, zorder=3,
    )

    # Solid structural evidence links and their contribution/provenance labels.
    ax.plot(
        [left_center[0] + sub_radius * 0.14, core_center[0] - core_radius],
        [left_center[1], core_center[1]], color=TEXT, linewidth=0.62, zorder=1,
    )
    ax.plot(
        [core_center[0] + core_radius, right_center[0] - sub_radius * 0.14],
        [core_center[1], right_center[1]], color=TEXT, linewidth=0.62, zorder=1,
    )
    ax.text(0.70, 0.535, f"({spec['left_edge']:.3f})", ha="center",
            va="center", fontsize=6.5, color=TEXT)
    ax.text(1.51, 0.395, f"({spec['right_edge']:.3f})", ha="center",
            va="center", fontsize=6.5, color=TEXT)

    # Curved dashed query link matches the reference figure's predicted edge.
    query_arc = FancyArrowPatch(
        left_center, right_center, arrowstyle="-",
        connectionstyle="arc3,rad=-0.31", linewidth=0.72,
        linestyle=(0, (3.0, 2.3)), color=ORANGE_EDGE,
        shrinkA=10, shrinkB=10, zorder=1,
    )
    ax.add_patch(query_arc)

    ax.text(0.005, 0.98, tag, transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", va="top", color=TEXT)
    ax.set_xlim(0, 2.2)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def save_bundle(fig) -> None:
    metadata = {
        "Title": "Molecular contribution subgraphs for representative MDAs",
        "Subject": "",
    }
    fig.savefig(ROOT / "fig5.pdf", dpi=1200, metadata=metadata)
    fig.savefig(ROOT / "fig5.svg", dpi=1200)
    fig.savefig(ROOT / "fig5.tiff", dpi=1200,
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(ROOT / "fig5_preview.png", dpi=600)
    plt.close(fig)


def main() -> None:
    configure_style()
    data = pd.read_csv(DATA_FILE)
    fig, axes = plt.subplots(2, 1, figsize=(3.50, 3.20))
    for index, (ax, (case, spec)) in enumerate(zip(axes, CASES.items())):
        draw_case(ax, case, spec, data, chr(ord("a") + index))
    fig.subplots_adjust(left=0.018, right=0.988, top=0.985, bottom=0.025, hspace=0.06)
    save_bundle(fig)


if __name__ == "__main__":
    main()
