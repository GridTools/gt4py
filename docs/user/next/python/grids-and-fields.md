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

# Grids and fields

Before running any operator, the grid has to be described and the fields have to
be allocated. This is done in regular Python code with the objects presented in
this page: dimensions, domains, fields and, for unstructured meshes,
connectivities.

```{code-cell} ipython3
import numpy as np

import gt4py.next as gtx
```

## Dimensions

A `Dimension` is a named axis. Two dimensions are the same if they have the same
name and kind.

```{code-cell} ipython3
I = gtx.Dimension("I")
J = gtx.Dimension("J")
K = gtx.Dimension("K", kind=gtx.DimensionKind.VERTICAL)
```

```{tip}
Create the dimensions of a model once, in a module, and import them wherever
they are needed.
```

There are three kinds of dimensions:

| Kind                   | Meaning                                                                            |
| ---------------------- | ---------------------------------------------------------------------------------- |
| `HORIZONTAL` (default) | Horizontal axes of a Cartesian grid, or locations of an unstructured mesh          |
| `VERTICAL`             | The vertical axis. [Scan operators](../dsl/scan-operators.md) run along it         |
| `LOCAL`                | The neighbors of a mesh location (see [Unstructured meshes](#unstructured-meshes)) |

The same definitions work for any grid. In a Cartesian grid every horizontal axis
is a dimension (`I`, `J`). In an unstructured mesh, every type of location is a
dimension (`Cell`, `Edge`, `Vertex`), and all the cells are numbered along the
single `Cell` axis.

## Domains

A domain is a set of grid points: a range of indices for each of its dimensions.
`gtx.domain` builds one from a dictionary, where a range is given either as its
size or as a `(start, stop)` pair:

```{code-cell} ipython3
domain = gtx.domain({I: 4, J: (2, 5), K: 10})

print(domain)
print(domain.shape)
```

Ranges do not need to start at zero. This is useful, for instance, to give
negative indices to the halo points of a grid, or to work on a part of a larger
grid while keeping its numbering.

## Allocating fields

The following functions create a field over a domain:

| Function                              | Result                                |
| ------------------------------------- | ------------------------------------- |
| `gtx.zeros(domain, dtype)`            | Field filled with zeros               |
| `gtx.ones(domain, dtype)`             | Field filled with ones                |
| `gtx.full(domain, fill_value, dtype)` | Field filled with `fill_value`        |
| `gtx.empty(domain, dtype)`            | Field with uninitialized values       |
| `gtx.as_field(domain, data, dtype)`   | Field with a copy of the array `data` |

The domain can be passed as a `Domain` or directly as the dictionary accepted by
`gtx.domain`. The default data type is `float64`, except in `full` and
`as_field`, which take it from the given values.

```{code-cell} ipython3
plane = gtx.domain({I: 4, J: (2, 5)})

temperature = gtx.full(plane, 280.0)
mask = gtx.zeros(plane, dtype=bool)
level_index = gtx.as_field({K: 10}, np.arange(10, dtype=np.int32))

print(temperature.domain, temperature.ndarray.dtype)
print(level_index.domain, level_index.ndarray.dtype)
```

`gtx.as_field` always **copies** the data: the field does not share memory with
the original array. It also accepts a list of dimensions instead of a domain. In
that case the ranges start at zero and have the sizes of the array, unless the
`origin` argument says which array index corresponds to the grid index zero:

```{code-cell} ipython3
data = np.arange(18.0).reshape(6, 3)

print(gtx.as_field([I, J], data).domain)
print(gtx.as_field([I, J], data, origin={I: 1}).domain)
```

### Allocating for a backend

By default, fields are allocated in main memory as NumPy arrays. The `allocator`
argument selects a different way of allocating the buffer. Passing a backend
allocates the field as that backend prefers, with the memory layout, the
alignment and the device (CPU or GPU) it needs:

```{code-cell} ipython3
temperature_cpu = gtx.zeros(plane, allocator=gtx.gtfn_cpu)
```

```{tip}
Allocate the fields with the backend which is going to run the operators. For
GPU backends it is required, since the fields must be on the device.
```

## Reading and writing the data

A field gives access to its domain and to its data:

```{code-cell} ipython3
field = gtx.as_field(plane, np.arange(12.0).reshape(4, 3))

print(field.domain)
print(field.asnumpy())
```

- `field.ndarray` is the underlying buffer, a NumPy array or the equivalent
  array of another library (for example CuPy, for fields on a GPU).
- `field.asnumpy()` always returns a NumPy array, copying the data from the
  device if needed.

Indexing a field gives another field, a view of a part of the original one.
Plain integers and slices are **relative** indices: they count positions from
the start of the field, as in NumPy. **Absolute** indices, written
`Dimension(index)`, refer to the indices of the domain. The domain of `field`
starts at 2 in the `J` direction, so these two expressions select the same
column:

```{code-cell} ipython3
print(field[:, 0].domain)
print(field[J(2)].domain)
```

A scalar or another field can be assigned to a part of a field. To write the
values of an array, use the buffer:

```{code-cell} ipython3
field[I(0)] = -1.0
field.ndarray[-1, :] = np.array([10.0, 20.0, 30.0])

print(field.asnumpy())
```

## Unstructured meshes

In an unstructured mesh the neighbors of a location cannot be computed from its
index, so they are given in tables called **connectivities**. The mesh below has
six cells (numbered in red) and twelve edges (in blue):

```{figure} ../images/connectivity_numbered_grid.svg
---
width: 40%
align: center
---
A mesh with 6 cells and 12 edges
```

Every type of location is a dimension, and fields are allocated as usual:

```{code-cell} ipython3
Cell = gtx.Dimension("Cell")
Edge = gtx.Dimension("Edge")

cell_values = gtx.as_field([Cell], np.array([1.0, 1.0, 2.0, 3.0, 5.0, 8.0]))
edge_values = gtx.zeros({Edge: 12})
```

### Connectivities and offsets

The edge-to-cell table has one row per edge, with the indices of the cells at
both sides of it. The edges on the boundary of the mesh have only one cell, and
the missing neighbor is marked with `-1`:

```{code-cell} ipython3
edge_to_cell_table = np.array(
    [
        [0, -1],  # edge 0 (neighbors: cell 0)
        [2, -1],  # edge 1
        [2, -1],  # edge 2
        [3, -1],  # edge 3
        [4, -1],  # edge 4
        [5, -1],  # edge 5
        [0, 5],  # edge 6 (neighbors: cell 0, cell 5)
        [0, 1],  # edge 7
        [1, 2],  # edge 8
        [1, 3],  # edge 9
        [3, 4],  # edge 10
        [4, 5],  # edge 11
    ]
)
```

Three objects describe this relation in GT4Py:

```{code-cell} ipython3
# 1. A local dimension, which numbers the neighbors of each edge (0 and 1)
E2CDim = gtx.Dimension("E2C", kind=gtx.DimensionKind.LOCAL)

# 2. An offset, the name used inside operators to go from the edges to their cells
E2C = gtx.FieldOffset("E2C", source=Cell, target=(Edge, E2CDim))

# 3. A connectivity, which holds the actual table
e2c_connectivity = gtx.as_connectivity([Edge, E2CDim], codomain=Cell, data=edge_to_cell_table)

print(e2c_connectivity.domain)
```

- In the offset, `source` is the dimension of the field which is read (cells),
  and `target` are the dimensions of the result: the edges, plus the local
  dimension numbering the neighbors. `E2C` therefore turns a field on cells into
  a field on edges.
- In `as_connectivity`, the first argument gives the dimensions of the table and
  `codomain` is the dimension its entries point to.
- The value marking the missing neighbors is the `skip_value` of the
  connectivity. `as_connectivity` uses `-1` when the table contains any `-1`,
  and it can also be set with the `skip_value` argument.

The offset is only a symbol. This keeps the operators independent of the mesh:
the same code runs on a different mesh by providing a different table.

```{important}
Give the same name to the local dimension and to the offset (`"E2C"` above).
The reductions over the neighbors rely on it to find the connectivity.
```

### The offset provider

When an operator using offsets is called, the connectivities are passed in the
`offset_provider` argument, a dictionary from the names of the offsets to the
connectivities:

```{code-cell} ipython3
offset_provider = {"E2C": e2c_connectivity}
```

[Shifts and neighbors](../dsl/shifts.md#unstructured-meshes) shows how the
offsets are used in operators.
