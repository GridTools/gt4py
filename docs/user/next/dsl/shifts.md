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

# Shifts and neighbors

Stencil computations combine the value at a grid point with the values at
neighboring points. In GT4Py there are no indices to do that. Instead, a field
is **called** with the description of where to read from, and the result is a
new field holding the values of the neighbors.

This page covers the two cases: Cartesian grids, where the neighbors are found
by adding an offset to the index, and unstructured meshes, where they are listed
in connectivity tables.

```{code-cell} ipython3
import numpy as np

import gt4py.next as gtx
```

## Cartesian shifts

A Cartesian shift is written as a dimension plus or minus an integer.
`a(I + 1)` is the field whose value at the point `i` is the value of `a` at
`i + 1`:

```{code-cell} ipython3
I = gtx.Dimension("I")
IField = gtx.Field[gtx.Dims[I], gtx.float64]

a = gtx.as_field({I: 6}, np.array([0.0, 1.0, 4.0, 9.0, 16.0, 25.0]))


@gtx.field_operator
def forward_difference(a: IField) -> IField:
    return a(I + 1) - a
```

The last point has no neighbor to the right, so the result cannot be computed
there. The `domain` argument restricts the call to the points where it can:

```{code-cell} ipython3
result = gtx.zeros({I: 6})

forward_difference(a, out=result, domain={I: (0, 5)})
print(result.asnumpy())
```

Shifts can be applied to any expression and chained, and fields with several
dimensions can be shifted along any of them:

```python
a(I + 1)(J - 1)  # diagonal neighbor
(a * b)(K - 1)  # value of the product in the previous level
```

The offset must be a number written in the code: `a(I + n)` with a variable `n`
is not supported.

### Shifts and domains

Running the same expression in plain Python shows what a shift does to the
domain of a field:

```{code-cell} ipython3
print(a.domain)
print(a(I + 1).domain)
print((a(I + 1) - a(I - 1)).domain)
```

`a(I + 1)` contains the same values as `a`, relabeled one position to the left.
An operation between fields is defined in the points which all its operands have
in common, so every shift in a stencil removes points at the boundaries.

```{warning}
Embedded execution checks the domains, and it reports an error when a result is
requested in points where it cannot be computed. The compiled backends do not
check them: they read outside of the fields and give wrong values without any
error. Test new operators in embedded execution first.
```

### Staggered dimensions

```{note}
Staggered dimensions are an experimental feature. The interface may still
change.
```

Many discretizations place some quantities at the centers of the cells and
others, like fluxes, at the interfaces between them. Such interfaces are
described by the **staggered** counterpart of a dimension:

```{code-cell} ipython3
IHalf = gtx.flip_staggered(I)

IHalfField = gtx.Field[gtx.Dims[IHalf], gtx.float64]
```

The point `i` of `IHalf` is located half a cell _before_ the point `i` of `I`,
that is, between the cells `i - 1` and `i`. Shifts by half a point move from
one dimension to the other. The dimension written in the shift is the one of
the **result**:

```{code-cell} ipython3
@gtx.field_operator
def gradient(a: IField) -> IHalfField:
    # at every interface: the cell after it minus the cell before it
    return a(IHalf + 0.5) - a(IHalf - 0.5)


@gtx.field_operator
def divergence(flux: IHalfField) -> IField:
    # at every cell: the interface after it minus the interface before it
    return flux(I + 0.5) - flux(I - 0.5)


@gtx.field_operator
def laplacian(a: IField) -> IField:
    return divergence(gradient(a))


result = gtx.zeros({I: 6})

laplacian(a, out=result, domain={I: (1, 5)})
print(result.asnumpy())
```

Using two different dimensions lets GT4Py check that cell and interface
quantities are not mixed by mistake. A field cannot have a dimension and its
staggered counterpart at the same time.

## Unstructured meshes

In an unstructured mesh the fields live on the locations of the mesh (cells,
edges, vertices), and the neighbors are given by connectivity tables. The
examples use this mesh with six cells and twelve edges:

```{figure} ../images/connectivity_numbered_grid.svg
---
width: 40%
align: center
---
The mesh, with the indices of the cells (red) and the edges (blue)
```

The following cell defines its dimensions and connectivities.
[Grids and fields](../python/grids-and-fields.md#unstructured-meshes) explains
these definitions in detail.

```{code-cell} ipython3
Cell = gtx.Dimension("Cell")
Edge = gtx.Dimension("Edge")
E2CDim = gtx.Dimension("E2C", kind=gtx.DimensionKind.LOCAL)
C2EDim = gtx.Dimension("C2E", kind=gtx.DimensionKind.LOCAL)

# Offsets: the names used in the operators to go from one location to another
E2C = gtx.FieldOffset("E2C", source=Cell, target=(Edge, E2CDim))
C2E = gtx.FieldOffset("C2E", source=Edge, target=(Cell, C2EDim))

# Tables: the cells of every edge (-1: no cell) and the edges of every cell
edge_to_cell_table = np.array(
    [[0, -1], [2, -1], [2, -1], [3, -1], [4, -1], [5, -1],
     [0, 5], [0, 1], [1, 2], [1, 3], [3, 4], [4, 5]]
)  # fmt: skip
cell_to_edge_table = np.array(
    [[0, 6, 7], [7, 8, 9], [1, 2, 8], [3, 9, 10], [4, 10, 11], [5, 6, 11]]
)

offset_provider = {
    "E2C": gtx.as_connectivity([Edge, E2CDim], codomain=Cell, data=edge_to_cell_table),
    "C2E": gtx.as_connectivity([Cell, C2EDim], codomain=Edge, data=cell_to_edge_table),
}

CellField = gtx.Field[gtx.Dims[Cell], gtx.float64]
EdgeField = gtx.Field[gtx.Dims[Edge], gtx.float64]

cell_values = gtx.as_field([Cell], np.array([1.0, 1.0, 2.0, 3.0, 5.0, 8.0]))
edge_values = gtx.zeros({Edge: 12})
```

```{figure} ../images/connectivity_cell_field.svg
---
width: 40%
align: center
---
The values of `cell_values`
```

### Selecting one neighbor

Calling a field with an offset moves its values to the neighboring locations.
`E2C` goes from the edges to their cells, so `cells(E2C[0])` is a field **on
the edges** holding, for every edge, the value of its first cell:

```{code-cell} ipython3
@gtx.field_operator
def first_cell(cells: CellField) -> EdgeField:
    return cells(E2C[0])


first_cell(cell_values, out=edge_values, offset_provider=offset_provider)
print(edge_values.asnumpy())
```

```{figure} ../images/connectivity_edge_0th_cell.svg
---
width: 40%
align: center
---
The result of `cells(E2C[0])`
```

Note that the offset is named after the direction of the lookup: the result is
defined on the _first_ location of its name (edges) and the values are read
from the _second_ one (cells).

### All the neighbors: local dimensions and reductions

Without an index, `cells(E2C)` gives the values of all the neighbors at once.
The result has an additional **local dimension**, `E2CDim`, with one entry for
every neighbor of the edge. A local dimension is removed with one of the
**reductions**, which combine the values of the neighbors of each location:

| Reduction                       | Result                             |
| ------------------------------- | ---------------------------------- |
| `neighbor_sum(field, axis=...)` | Sum of the values of the neighbors |
| `min_over(field, axis=...)`     | Minimum value of the neighbors     |
| `max_over(field, axis=...)`     | Maximum value of the neighbors     |

```{code-cell} ipython3
from gt4py.next import neighbor_sum


@gtx.field_operator
def sum_of_adjacent_cells(cells: CellField) -> EdgeField:
    return neighbor_sum(cells(E2C), axis=E2CDim)


sum_of_adjacent_cells(cell_values, out=edge_values, offset_provider=offset_provider)
print(edge_values.asnumpy())
```

```{figure} ../images/connectivity_edge_cell_sum.svg
---
width: 40%
align: center
---
The result of `neighbor_sum(cells(E2C), axis=E2CDim)`
```

Operations can be done before reducing. For instance,
`neighbor_sum(weights * cells(E2C), axis=E2CDim)` is a weighted sum, where
`weights` is a field with dimensions `Edge, E2CDim`.

### Composing operators

The result of a neighbor access is a regular field, which can be passed to
other operators or shifted again. This operator goes from the cells to the
edges and back to the cells:

```{code-cell} ipython3
@gtx.field_operator
def sum_over_edges(cells: CellField) -> CellField:
    edge_sums = sum_of_adjacent_cells(cells)
    return neighbor_sum(edge_sums(C2E), axis=C2EDim)


cell_result = gtx.zeros({Cell: 6})

sum_over_edges(cell_values, out=cell_result, offset_provider=offset_provider)
print(cell_result.asnumpy())
```

All the connectivities used by an operator, including the ones needed by the
operators it calls, must be in the offset provider.

### Missing neighbors

The edges at the boundary of the mesh have only one cell, and their second
entry in the table is the skip value `-1`. This has two consequences:

- **Reductions ignore the missing neighbors.** In the example above, the sum
  for a boundary edge is just the value of its only cell.
- **A selected neighbor is only defined where it exists.** `cells(E2C[1])` has
  no value at the boundary edges, so an operator using it can only be computed
  on the interior ones.

In this mesh the interior edges are numbered last, from 6 to 11, so they can be
selected with a domain:

```{code-cell} ipython3
@gtx.field_operator
def jump_across_edge(cells: CellField) -> EdgeField:
    return cells(E2C[1]) - cells(E2C[0])


edge_values = gtx.zeros({Edge: 12})

jump_across_edge(
    cell_values, out=edge_values, domain={Edge: (6, 12)}, offset_provider=offset_provider
)
print(edge_values.asnumpy())
```

```{tip}
Number the locations of a mesh so that the ones needing a special treatment,
like the boundary, are contiguous. Domains can then select them without masks.
```
