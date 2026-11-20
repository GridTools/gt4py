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

# Premap and neighbors

Stencil computations combine the value at a grid point with the values at
neighboring points. In GT4Py there are no indices to do that. Instead, a field
is **called** with an offset, and the result is a new field holding the values
of the neighbors. This operation is called **premap**.

- On a Cartesian grid the offset is a dimension plus an integer, as in
  `a(I + 1)`.
- On an unstructured mesh it is a `FieldOffset` such as `E2C`, a tag which
  stands for a connectivity table, as in `a(E2C[0])`.

```{note}
The name comes from seeing a field as a function from grid points to values.
An offset is a function too, from every point to one of its neighbors. Calling
the field with the offset composes both: the offset is applied first, _before_
the field, and the result is again a field. It is plain function composition,
like the familiar `map` (or `fmap`) which applies a function to the values of a
collection, but in the opposite order. Other languages have other names for the
same operation, for example `contramap` in Haskell.
```

```{code-cell} ipython3
import numpy as np

import gt4py.next as gtx
```

## Cartesian offsets

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

Premap can be applied to any expression and chained, and fields with several
dimensions accept offsets along any of them:

```python
a(I + 1)(J - 1)  # diagonal neighbor of a field over I, J
(a * b)(K - 1)  # value of the product at the previous K level
```

The offset must be a literal: `a(I + n)` with a variable `n` is not supported.

### Premap and domains

Running the same expressions in plain Python shows what premap does to the
domain of a field:

```{code-cell} ipython3
print(a.domain)
print(a(I + 1).domain)
print((a(I + 1) - a(I - 1)).domain)
```

`a(I + 1)` contains the same values as `a`, relabeled one position to the left:
premap with a Cartesian offset moves the domain of a field, it does not make
it smaller. An operation between fields is defined at the points which all its
operands have in common. Combining the results of different offsets therefore
gives a result on a smaller domain than the one of the original field.

```{warning}
Embedded execution checks the domains, and it reports an error when a result is
requested at points where it cannot be computed. The compiled backends do not
check them: they read outside of the fields and give wrong values without any
error. Test new operators in embedded execution first.
```

### Boundaries with `concat_where`

```{note}
`concat_where` is an experimental feature, imported from
`gt4py.next.experimental`. Its interface may still change.
```

Restricting the domain leaves the boundary points untouched. When they need a
value of their own, as in boundary conditions, use `concat_where`. It is similar
to `where`, but its condition is a **region of the grid**, written as a
comparison between a dimension and an index, instead of a mask computed from
values:

```{code-cell} ipython3
from gt4py.next.experimental import concat_where


@gtx.field_operator
def smooth(a: IField, boundary_value: IField) -> IField:
    smoothed = 0.25 * a(I - 1) + 0.5 * a + 0.25 * a(I + 1)
    return concat_where((I > 0) & (I < 5), smoothed, boundary_value)


smooth(a, gtx.full({I: 6}, -1.0), out=result)
print(result.asnumpy())
```

This is the difference with [`where`](field-operators.md#local-variables-and-conditionals):

- `where(mask, a, b)` is an operation between three fields. Its result is only
  defined at the points which the mask and **both** branches have in common,
  even at the points where the mask does not select one of them.
- `concat_where(region, a, b)` takes `a` only inside the region and `b` only
  outside of it, and concatenates the two pieces. A branch does not need to be
  defined outside of its own part of the domain.

Here `smoothed` is only defined at the interior points, the only ones having
both neighbors, and still the operator can be called on the whole grid.

```{important}
In embedded execution `concat_where` has some restrictions:

- Every condition must select a single range of indices of one dimension, like
  `I < 1`, `K == 0` or `(I > 0) & (I < 5)`. Conditions with `|` or `!=`, or
  involving two dimensions, are not supported. Nest several `concat_where` calls
  instead.
- A scalar is only accepted as the branch of a region bounded on both sides.
  Otherwise use fields in both branches.
```

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

The following cell defines the mesh, as explained in
[Grids and fields](../python/grids-and-fields.md#unstructured-meshes): the
dimensions, the offsets `E2C` (the cells of every edge) and `C2E` (the edges of
every cell), and the connectivity tables behind them.

```{code-cell} ipython3
Cell = gtx.Dimension("Cell")
Edge = gtx.Dimension("Edge")
E2CDim = gtx.Dimension("E2C", kind=gtx.DimensionKind.LOCAL)
C2EDim = gtx.Dimension("C2E", kind=gtx.DimensionKind.LOCAL)

E2C = gtx.FieldOffset("E2C", source=Cell, target=(Edge, E2CDim))
C2E = gtx.FieldOffset("C2E", source=Edge, target=(Cell, C2EDim))

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

Read `E2C` as "for every edge, its cells": calling a field with it gives a
result on the edges, with values coming from the cells. `cells(E2C[0])` holds,
for every edge, the value of its first cell:

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

The connectivities are passed in the `offset_provider` argument of the call. All
the connectivities used by an operator, including the ones needed by the
operators it calls, must be there.

```{important}
Select a neighbor by its index only when **every** location has that neighbor,
that is, when the connectivity table has no missing entries in that column.
This is the case of the connectivities with a fixed number of neighbors, like
the two vertices of an edge. In this mesh every edge has a first cell, so
`cells(E2C[0])` is fine, but the boundary edges do not have a second one. When
the number of neighbors is not the same for all the locations, use the
reductions explained next.
```

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

Reductions **ignore the missing neighbors**: in this example, the sum for a
boundary edge is just the value of its only cell. They are the way to work with
connectivities where some locations have fewer neighbors than others.

The values of the neighbors can be combined with other fields before reducing.
For instance, `neighbor_sum(weights * cells(E2C), axis=E2CDim)` is a weighted
sum, where `weights` is a field with dimensions `Edge, E2CDim`.

A field can only have one local dimension, so `cells(E2C)(C2E)` is not
supported: reduce the first local dimension before using another offset.

## Staggered dimensions

```{note}
Staggered dimensions are an experimental feature. The interface may still
change.
```

Many discretizations on Cartesian grids place some quantities at the centers of
the cells and others, like fluxes, at the interfaces between them. Such
interfaces are described by the **staggered** counterpart of a dimension:

```{code-cell} ipython3
IHalf = gtx.flip_staggered(I)

IHalfField = gtx.Field[gtx.Dims[IHalf], gtx.float64]
```

The point `i` of `IHalf` is located half a cell _before_ the point `i` of `I`,
that is, between the cells `i - 1` and `i`. Offsets of half a point move from
one dimension to the other. The dimension written in the offset is the one of
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
quantities are not mixed by mistake. The type of a field in an operator cannot
have a dimension and its staggered counterpart at the same time.
