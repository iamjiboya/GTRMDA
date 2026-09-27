"""Draw reference-style clustered expression dot maps for two MDA cases."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import leaves_list, linkage

from font_utils import publication_serif


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "fig6_expression_data.csv"

BLUE = "#78A9D1"
BLUE_EDGE = "#2F618A"
ORANGE = "#F2A166"
TEAL = "#8FC9B5"
MAUVE = "#C9A4C5"
TEXT = "#202020"
TREE = "#7F898F"
GRID = "#D7DEE1"
LABEL_BG = "#F1F2F2"
LABEL_EDGE = "#AEB6BA"

CMAP = LinearSegmentedColormap.from_list(
    "gtrmda_expression", [MAUVE, "#F8F5F3", TEAL], N=256
)
NORM = Normalize(vmin=-1.8, vmax=1.8)

CASES = [
    {
        "disease": "Breast Neoplasms",
        "title": "Breast Neoplasms",
        "samples": ["B1", "B2", "B3", "N1", "N2", "N3"],
        "target": "hsa-miR-21-5p",
        "accent": BLUE_EDGE,
    },
    {
        "disease": "Alzheimer's Disease",
        "title": "Alzheimer's Disease",
        "samples": ["A1", "A2", "A3", "C1", "C2", "C3"],
        "target": "hsa-miR-146a-5p",
        "accent": "#A95F2F",
    },
]


def configure_style() -> None:
    family = publication_serif(ROOT)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [family],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8.3,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.3,
        "ytick.labelsize": 8.3,
        "savefig.facecolor": "white",
    })


def draw_dendrogram(ax, matrix, order, y_positions, x_leaf, width) -> None:
    z = linkage(matrix, method="average", metric="euclidean", optimal_ordering=True)
    max_distance = max(float(z[:, 2].max()), 1e-8)
    node_xy = {int(leaf): (x_leaf, y_positions[int(leaf)]) for leaf in order}

    for merge_index, (left, right, distance, _) in enumerate(z):
        left = int(left)
        right = int(right)
        lx, ly = node_xy[left]
        rx, ry = node_xy[right]
        parent_x = x_leaf - width * float(distance) / max_distance
        parent_y = (ly + ry) / 2
        ax.plot([lx, parent_x], [ly, ly], color=TREE, linewidth=0.55, zorder=1)
        ax.plot([rx, parent_x], [ry, ry], color=TREE, linewidth=0.55, zorder=1)
        ax.plot([parent_x, parent_x], [ly, ry], color=TREE, linewidth=0.55, zorder=1)
        node_xy[len(matrix) + merge_index] = (parent_x, parent_y)


def draw_colorbar(ax, x0, x1, y0, y1) -> None:
    gradient = np.linspace(-1.8, 1.8, 256).reshape(1, -1)
    ax.imshow(
        gradient, extent=(x0, x1, y0, y1), origin="lower",
        cmap=CMAP, norm=NORM, aspect="auto", interpolation="bicubic", zorder=0,
    )
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                           fill=False, edgecolor=TEXT, linewidth=0.55, zorder=2))
    ax.text(x0, y0 - 0.12, "min", ha="center", va="top", fontsize=6.6, color=TEXT)
    ax.text(x1, y0 - 0.12, "max", ha="center", va="top", fontsize=6.6, color=TEXT)


def draw_case(ax, frame, spec, panel_label) -> None:
    values = frame.iloc[:, 2:].to_numpy(dtype=float)
    labels = frame["mirna"].tolist()
    z = linkage(values, method="average", metric="euclidean", optimal_ordering=True)
    order = leaves_list(z).tolist()

    # Reference-style geometry: dendrogram, circular heat map, boxed row labels.
    n_rows, n_cols = values.shape
    y_by_leaf = {leaf: n_rows - 1 - rank for rank, leaf in enumerate(order)}
    heat_x = np.linspace(0.35, 0.69, n_cols)
    x_leaf = 0.305
    draw_dendrogram(ax, values, order, y_by_leaf, x_leaf=x_leaf, width=0.205)

    for col, x in enumerate(heat_x):
        for leaf in order:
            y = y_by_leaf[leaf]
            value = values[leaf, col]
            size = 50 + 24 * min(abs(value) / 1.8, 1.0)
            ax.scatter(x, y, s=size, facecolor=CMAP(NORM(value)),
                       edgecolor="white", linewidth=0.28, zorder=3)

    label_x0, label_x1 = 0.735, 1.125
    ax.add_patch(Rectangle(
        (label_x0, -0.47), label_x1 - label_x0, n_rows - 0.06,
        facecolor=LABEL_BG, edgecolor=LABEL_EDGE, linewidth=0.55, zorder=0,
    ))
    for leaf in order:
        y = y_by_leaf[leaf]
        label = labels[leaf].replace("hsa-", "")
        is_target = labels[leaf] == spec["target"]
        ax.text(label_x0 + 0.018, y, label, ha="left", va="center",
                fontsize=6.45, color=TEXT,
                fontweight="bold" if is_target else "normal", zorder=4)
        if is_target:
            ax.plot([label_x0 + 0.006, label_x0 + 0.006], [y - 0.36, y + 0.36],
                    color=spec["accent"], linewidth=1.45, zorder=4)

    for x, sample in zip(heat_x, spec["samples"]):
        ax.text(x, n_rows + 0.02, sample, ha="center", va="bottom",
                fontsize=6.65, color=TEXT)

    draw_colorbar(ax, 0.09, 1.09, n_rows + 1.42, n_rows + 1.76)
    ax.text(0.59, n_rows + 2.14, spec["title"], ha="center", va="center",
            fontsize=8.6, fontweight="bold", color=TEXT)
    ax.text(0.00, n_rows + 2.34, panel_label, ha="left", va="top",
            fontsize=10, fontweight="bold", color=TEXT)

    # Thin separation between case and control samples.
    split_x = (heat_x[2] + heat_x[3]) / 2
    ax.plot([split_x, split_x], [-0.44, n_rows - 0.56], color=GRID,
            linewidth=0.55, linestyle=(0, (2, 2)), zorder=1)
    ax.set_xlim(-0.01, 1.16)
    ax.set_ylim(-0.72, n_rows + 2.46)
    ax.axis("off")


def save_bundle(fig) -> None:
    metadata = {
        "Title": "Differential expression patterns for representative MDA cases",
        "Subject": "",
    }
    fig.savefig(ROOT / "fig6.pdf", dpi=1200, metadata=metadata)
    fig.savefig(ROOT / "fig6.svg", dpi=1200)
    fig.savefig(ROOT / "fig6.tiff", dpi=1200,
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(ROOT / "fig6_preview.png", dpi=600)
    plt.close(fig)


def main() -> None:
    configure_style()
    data = pd.read_csv(DATA_FILE)
    fig, axes = plt.subplots(1, 2, figsize=(3.50, 2.25))
    for index, (ax, spec) in enumerate(zip(axes, CASES)):
        frame = data[data["disease"] == spec["disease"]].reset_index(drop=True)
        draw_case(ax, frame, spec, chr(ord("a") + index))
    fig.subplots_adjust(left=0.018, right=0.992, top=0.985, bottom=0.025, wspace=0.035)
    save_bundle(fig)


if __name__ == "__main__":
    main()
