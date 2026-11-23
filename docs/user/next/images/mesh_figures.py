# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Generate the figures of the small unstructured mesh used in the documentation.

The mesh, its connectivity tables and the values of the fields are the same as
in the pages using the figures (`python/grids-and-fields.md` and
`dsl/shifts.md`), so the figures show what the code of those pages computes.

Run it from the root of the repository after changing anything here:

    uv run --group docs python docs/user/next/images/mesh_figures.py
"""

from __future__ import annotations

import math
import pathlib

import matplotlib


matplotlib.use("svg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors


OUTPUT_DIR = pathlib.Path(__file__).parent

# -- Mesh --
# Five triangles around a central vertex, plus one more triangle (cell 2)
_ring = [
    (math.cos(math.radians(angle)), math.sin(math.radians(angle)))
    for angle in (198, 270, 342, 54, 126)
]
VERTICES = np.array(
    [
        _ring[0],  # 0
        _ring[1],  # 1
        _ring[2],  # 2
        _ring[3],  # 3
        _ring[4],  # 4
        (0.0, 0.0),  # 5: central vertex
        (1.05, -1.15),  # 6: outer vertex of cell 2
    ]
)
EDGE_TO_VERTEX = np.array(
    [[0, 1], [1, 6], [6, 2], [2, 3], [3, 4], [4, 0],
     [0, 5], [1, 5], [1, 2], [5, 2], [5, 3], [5, 4]]
)  # fmt: skip
EDGE_TO_CELL = np.array(
    [[0, -1], [2, -1], [2, -1], [3, -1], [4, -1], [5, -1],
     [0, 5], [0, 1], [1, 2], [1, 3], [3, 4], [4, 5]]
)  # fmt: skip
CELL_TO_EDGE = np.array([[0, 6, 7], [7, 8, 9], [1, 2, 8], [3, 9, 10], [4, 10, 11], [5, 6, 11]])

CELL_VALUES = np.array([1.0, 1.0, 2.0, 3.0, 5.0, 8.0])

# -- Style --
CELL_COLOR = "#d6336c"  # indices of the cells
EDGE_COLOR = "#1c7ed6"  # indices of the edges
LINE_COLOR = "#495057"
CELL_FILL = "#f1f3f5"
MUTED_COLOR = "#868e96"
FONT = {"family": "DejaVu Sans"}
CELL_CMAP = mcolors.LinearSegmentedColormap.from_list("cells", ["#fff4e6", "#ffa94d"])
EDGE_CMAP = mcolors.LinearSegmentedColormap.from_list("edges", ["#a5d8ff", "#1864ab"])


def cell_vertices(cell: int) -> np.ndarray:
    return VERTICES[sorted(set(EDGE_TO_VERTEX[CELL_TO_EDGE[cell]].ravel()))]


def cell_center(cell: int) -> np.ndarray:
    return cell_vertices(cell).mean(axis=0)


def edge_center(edge: int) -> np.ndarray:
    return VERTICES[EDGE_TO_VERTEX[edge]].mean(axis=0)


def format_value(value: float) -> str:
    return f"{value:g}"


def draw_mesh(
    ax: plt.Axes,
    *,
    cell_fill: list[str] | None = None,
    edge_colors: list[str] | None = None,
    edge_width: float = 1.6,
) -> None:
    for cell in range(len(CELL_TO_EDGE)):
        ax.fill(
            *cell_vertices(cell).T,
            facecolor=cell_fill[cell] if cell_fill else CELL_FILL,
            edgecolor="none",
            zorder=1,
        )
    for edge, vertices in enumerate(EDGE_TO_VERTEX):
        ax.plot(
            *VERTICES[vertices].T,
            color=edge_colors[edge] if edge_colors else LINE_COLOR,
            linewidth=edge_width,
            solid_capstyle="round",
            zorder=2,
        )
    ax.scatter(*VERTICES.T, s=22, color=LINE_COLOR, zorder=3)


def edge_label(ax: plt.Axes, edge: int, text: str, *, color: str, weight: str = "normal") -> None:
    ax.text(
        *edge_center(edge),
        text,
        color=color,
        fontsize=10,
        fontweight=weight,
        ha="center",
        va="center",
        zorder=4,
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": color,
            "linewidth": 0.9,
        },
        **FONT,
    )


def cell_label(ax: plt.Axes, cell: int, text: str, *, color: str, size: float = 15) -> None:
    ax.text(
        *cell_center(cell),
        text,
        color=color,
        fontsize=size,
        fontweight="bold",
        ha="center",
        va="center",
        zorder=4,
        **FONT,
    )


def new_figure() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.15, 1.25)
    ax.set_ylim(-1.3, 1.0)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig, ax


def save(fig: plt.Figure, name: str) -> None:
    # Fixed metadata and ids, so regenerating the figures gives identical files
    plt.rcParams["svg.hashsalt"] = "gt4py"
    fig.savefig(OUTPUT_DIR / name, format="svg", facecolor="white", metadata={"Date": None})
    plt.close(fig)


def numbered_grid() -> None:
    fig, ax = new_figure()
    draw_mesh(ax)
    for cell in range(len(CELL_TO_EDGE)):
        cell_label(ax, cell, str(cell), color=CELL_COLOR)
    for edge in range(len(EDGE_TO_VERTEX)):
        edge_label(ax, edge, str(edge), color=EDGE_COLOR)
    save(fig, "connectivity_numbered_grid.svg")


def cell_field() -> None:
    fig, ax = new_figure()
    norm = mcolors.Normalize(vmin=0.0, vmax=CELL_VALUES.max())
    draw_mesh(ax, cell_fill=[mcolors.to_hex(CELL_CMAP(norm(value))) for value in CELL_VALUES])
    for cell, value in enumerate(CELL_VALUES):
        cell_label(ax, cell, format_value(value), color="#212529")
    save(fig, "connectivity_cell_field.svg")


def edge_field(values: np.ndarray, name: str) -> None:
    fig, ax = new_figure()
    norm = mcolors.Normalize(vmin=0.0, vmax=values.max())
    edge_colors = [mcolors.to_hex(EDGE_CMAP(norm(value))) for value in values]
    draw_mesh(ax, edge_colors=edge_colors, edge_width=3.0)
    for cell, value in enumerate(CELL_VALUES):
        cell_label(ax, cell, format_value(value), color=MUTED_COLOR, size=12)
    for edge, value in enumerate(values):
        edge_label(ax, edge, format_value(value), color="#1864ab", weight="bold")
    save(fig, name)


def main() -> None:
    numbered_grid()
    cell_field()

    # cells(E2C[0]): the value of the first cell of every edge
    edge_field(CELL_VALUES[EDGE_TO_CELL[:, 0]], "connectivity_edge_0th_cell.svg")

    # neighbor_sum(cells(E2C), axis=E2CDim): the missing neighbors (-1) are ignored
    neighbor_values = np.where(EDGE_TO_CELL >= 0, CELL_VALUES[EDGE_TO_CELL], 0.0)
    edge_field(neighbor_values.sum(axis=1), "connectivity_edge_cell_sum.svg")


if __name__ == "__main__":
    main()
