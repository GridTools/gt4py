---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Operators on an unstructured mesh

This example builds the classic differential operators of a finite volume
discretization, gradient, divergence and Laplacian, on an unstructured mesh. It
shows:

- The definition of a mesh with its connectivities.
- Neighbor accesses in both directions, from edges to cells and from cells to
  edges.
- A reduction with geometric factors stored in a field with a local dimension.
- Operators which do not depend on the mesh they run on.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np

import gt4py.next as gtx
from gt4py.next import neighbor_sum
```

## The operators

The fields live on the cells and on the edges of the mesh. Every edge has two
cells, and every cell has a fixed number of edges:

```{code-cell} ipython3
Cell = gtx.Dimension("Cell")
Edge = gtx.Dimension("Edge")
E2CDim = gtx.Dimension("E2C", kind=gtx.DimensionKind.LOCAL)
C2EDim = gtx.Dimension("C2E", kind=gtx.DimensionKind.LOCAL)

E2C = gtx.FieldOffset("E2C", source=Cell, target=(Edge, E2CDim))
C2E = gtx.FieldOffset("C2E", source=Edge, target=(Cell, C2EDim))

CellField = gtx.Field[gtx.Dims[Cell], gtx.float64]
EdgeField = gtx.Field[gtx.Dims[Edge], gtx.float64]
CellEdgesField = gtx.Field[gtx.Dims[Cell, C2EDim], gtx.float64]
```

The **gradient** of a cell quantity is computed at the edges, as the difference
between the two cells of the edge divided by the distance between their
centers. It is the component of the gradient normal to the edge, pointing from
the first cell to the second one:

```{code-cell} ipython3
@gtx.field_operator
def gradient(phi: CellField, inverse_distance: EdgeField) -> EdgeField:
    return (phi(E2C[1]) - phi(E2C[0])) * inverse_distance
```

The **divergence** of a flux given at the edges is computed at the cells: the
sum of the fluxes through the edges of the cell, each one multiplied by the
length of the edge, divided by the area of the cell. A flux pointing out of one
cell points into its neighbor, so every term needs a sign. These geometric
factors depend on the cell _and_ on which of its edges is considered, so they
are stored in a field with the local dimension `C2EDim`:

```{code-cell} ipython3
@gtx.field_operator
def divergence(flux: EdgeField, geometric_factor: CellEdgesField) -> CellField:
    return neighbor_sum(flux(C2E) * geometric_factor, axis=C2EDim)
```

The **Laplacian** is the divergence of the gradient, and a diffusion step uses
it to update the field:

```{code-cell} ipython3
@gtx.field_operator
def laplacian(
    phi: CellField, inverse_distance: EdgeField, geometric_factor: CellEdgesField
) -> CellField:
    return divergence(gradient(phi, inverse_distance), geometric_factor)


@gtx.field_operator
def diffusion_step(
    phi: CellField,
    coefficient: gtx.float64,
    inverse_distance: EdgeField,
    geometric_factor: CellEdgesField,
) -> CellField:
    return phi + coefficient * laplacian(phi, inverse_distance, geometric_factor)
```

Nothing in these definitions says how many cells the mesh has, which shape they
have, or how they are connected. All of that is provided when calling them.

## A mesh

To keep the example self-contained, the mesh is generated in the code: a
periodic arrangement of `nx` × `ny` square cells of side `dx`. It is described
as an unstructured mesh, with numbered cells and edges and connectivity tables.
The edge at the west side of the cell `c` gets the index `c`, and the edge at
its south side gets the index `n_cells + c`.

```{code-cell} ipython3
nx, ny, dx = 48, 32, 1.0
n_cells, n_edges = nx * ny, 2 * nx * ny

cell_index = np.arange(n_cells).reshape(nx, ny)
west_cell = np.roll(cell_index, 1, axis=0)
east_cell = np.roll(cell_index, -1, axis=0)
south_cell = np.roll(cell_index, 1, axis=1)
north_cell = np.roll(cell_index, -1, axis=1)

# Cells of every edge: first the cell before the edge, then the cell after it
edge_to_cell_table = np.concatenate(
    [
        np.stack([west_cell.ravel(), cell_index.ravel()], axis=1),  # west edges
        np.stack([south_cell.ravel(), cell_index.ravel()], axis=1),  # south edges
    ]
)

# Edges of every cell: west, east, south, north
cell_to_edge_table = np.stack(
    [
        cell_index.ravel(),
        east_cell.ravel(),
        n_cells + cell_index.ravel(),
        n_cells + north_cell.ravel(),
    ],
    axis=1,
)

offset_provider = {
    "E2C": gtx.as_connectivity([Edge, E2CDim], codomain=Cell, data=edge_to_cell_table),
    "C2E": gtx.as_connectivity([Cell, C2EDim], codomain=Edge, data=cell_to_edge_table),
}
```

The geometric fields complete the description of the mesh. The gradient points
from the first cell of an edge to the second one, which is the outward
direction for the east and north edges of a cell, and the inward one for its
west and south edges:

```{code-cell} ipython3
inverse_distance = gtx.full({Edge: n_edges}, 1.0 / dx)

# sign * edge length / cell area, for the west, east, south and north edges
factors = np.tile(np.array([-1.0, 1.0, -1.0, 1.0]) * dx / dx**2, (n_cells, 1))
geometric_factor = gtx.as_field([Cell, C2EDim], factors)
```

## Running it

The initial state is a square block of value one:

```{code-cell} ipython3
initial = np.zeros((nx, ny))
initial[16:32, 10:22] = 1.0

phi = gtx.as_field([Cell], initial.ravel())
new_phi = gtx.zeros({Cell: n_cells})

for _ in range(100):
    diffusion_step(
        phi, 0.2, inverse_distance, geometric_factor, out=new_phi, offset_provider=offset_provider
    )
    phi, new_phi = new_phi, phi
```

The fields over the cells are one-dimensional. For this mesh the cell index can
be turned back into a position to plot them:

```{code-cell} ipython3
final = phi.asnumpy().reshape(nx, ny)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True)
for ax, data, title in zip(axes, (initial, final), ("initial", "after 100 steps")):
    image = ax.imshow(data.T, origin="lower", cmap="magma", vmin=0.0, vmax=1.0)
    ax.set(title=title)
fig.colorbar(image, ax=axes)
plt.show()
```

## Checking the result

On this mesh, the operator must be the five-point Laplacian of a periodic
Cartesian grid:

```{code-cell} ipython3
def numpy_laplacian(phi):
    return (
        np.roll(phi, 1, axis=0)
        + np.roll(phi, -1, axis=0)
        + np.roll(phi, 1, axis=1)
        + np.roll(phi, -1, axis=1)
        - 4.0 * phi
    ) / dx**2


result = gtx.zeros({Cell: n_cells})
laplacian(
    gtx.as_field([Cell], initial.ravel()),
    inverse_distance,
    geometric_factor,
    out=result,
    offset_provider=offset_provider,
)

np.allclose(result.asnumpy().reshape(nx, ny), numpy_laplacian(initial))
```

The scheme is written in flux form, so the total amount of `phi` is conserved:

```{code-cell} ipython3
bool(np.isclose(final.sum(), initial.sum()))
```

## Other meshes

To run the same operators on a triangular or a hexagonal mesh, only the data
changes: the connectivity tables, with three or six edges per cell, and the
geometric fields. The definitions of `gradient`, `divergence` and `laplacian`
stay as they are.
