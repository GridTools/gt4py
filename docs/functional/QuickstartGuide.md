---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# GT4Py User Quickstart Guide

+++

## Info for writing the guide
Goals:
- Read by someone completely new
- Focus on unstructured
They will be able to:
- Install GT4Py
- Write simple code
- Understand basic concepts of GT4Py
Examples could be:
- Laplacian
    - Main concepts: fields, field operators, programs, reductions
- Realistic example: diffusion, advection, solve some simple PDE on a 2D grid

TODO:
explain scalar types of fields: https://github.com/GridTools/gt4py/pull/711

+++

## Installation

GT4Py is distributed as a Python package and can be installed directly from GitHub:

```{code-cell} ipython3
#! pip install git+https://github.com/gridtools/gt4py.git@functional
```

## Concepts

### Basics

Before we start, let's import the most important parts of GT4Py which we are going to use throughout this document:

```{code-cell} ipython3
import numpy as np

from functional.ffront.fbuiltins import Field, float32, FieldOffset, neighbor_sum
from functional.iterator.runtime import CartesianAxis
from functional.ffront.decorator import field_operator, program
from functional.iterator.embedded import np_as_located_field, NeighborTableOffsetProvider
```

GT4Py uses so-called *fields* to represent multi-dimensional arrays. In this example, we are going to work with two-dimensional fields: one dimension for an unstructured horizontal grid, and another dimension for vertical layers. The dimensions are declared as a `CartesianAxis` in GT4Py.

The fields themselves are best created using utility functions such as `np_as_located_field` that converts `numpy` arrays into GT4Py `Field`s. The code below creates two fields, both with 5 cells in the horizontal grid and 6 vertical layers, with all the 5\*6=30 values set to 2.0 for one fields and 3.0 for the other field.

```{code-cell} ipython3
CellDim = CartesianAxis("Cell")
KDim = CartesianAxis("K")

num_cells = 5
num_layers = 6
grid_shape = (num_cells, num_layers)

a_value = 2.0
b_value = 3.0
a = np_as_located_field(CellDim, KDim)(np.full(shape=grid_shape, fill_value=a_value, dtype=np.float32))
b = np_as_located_field(CellDim, KDim)(np.full(shape=grid_shape, fill_value=b_value, dtype=np.float32))
```

To define operations involving one or more fields, GT4Py allows us to declare *field operators*. Field operators are pure functions (i.e. functions without side effects) that take immutable `Field`s as arguments and output another `Field` as a result. Field operators must be declared with the `@field_operator` decorator, and are allowed to use a certain subset of the Python syntax.

```{code-cell} ipython3
@field_operator
def add(a : Field[[CellDim, KDim], float32],
        b : Field[[CellDim, KDim], float32]) -> Field[[CellDim, KDim], float32]:
    return a + b
```

*Programs* are similar to fields operators, but they allow mutability of the arguments. Programs must be declared with the `@program` decorator and are allowed to use a different subset of the Python syntax compared to field operators. Programs are used to call and chain field operators:

```{code-cell} ipython3
@program
def run_add(a : Field[[CellDim, KDim], float32],
                b : Field[[CellDim, KDim], float32],
                out : Field[[CellDim, KDim], float32]):
    add(a, b, out=out)
```

To add the two fields elementwise, we can execute the program we just declared. The expectation is that every cell of the resulting field will be 2+3=5.

```{code-cell} ipython3
result = np_as_located_field(CellDim, KDim)(np.zeros(shape=grid_shape))
run_add(a, b, result, offset_provider={})

print("{} + {} = {} ± {}".format(a_value, b_value, np.average(np.asarray(result)), np.std(np.asarray(result))))
```

### Unstructured grids and connectivity

When using unstructured grids, we have to define adjacency between nodes, cells and edges manually. In this section, we will create the mesh illustrated below and we will do some calculations with fields on this mesh.

![grid_topo](connectivity_numbered_grid.svg)

The <span style="color: #C02020">faces</span> and the <span style="color: #0080FF">edges</span> of the mesh have been numbered with zero-based indices.

#### Define grid and connectivity

We are going to use two fields: one on the cells of the grid, and on the edges. Both fields will have only one dimension, declared as `CellDim` for the field on the cells and `EdgeDim` for the field on the edges.

Furthermore, we will define the edge-to-cell connectivity that tells us which cells are neighbours to a particular edge. The connectivity is thus defined with a matrix where each line corresponds to an edge, and has 2 entries for the two cells to the side of that edge. (Missing neighbors are filled with -1.)

The *field offset* `E2C` is used inside the field operator to indicate that we want to access the cells neighboring the edges. The *offset provider* forwards the actual connectivity matrix to the field operator.

```{code-cell} ipython3
CellDim = CartesianAxis("Cell")
EdgeDim = CartesianAxis("Edge")
E2CDim = CartesianAxis("E2C")
E2C = FieldOffset("E2C", source=CellDim, target=(EdgeDim, E2CDim))

cell_neighbours_of_edges = np.array([
    [0, -1],
    [2, -1],
    [2, -1],
    [3, -1],
    [4, -1],
    [5, -1],
    [0, 5],
    [0, 1],
    [1, 2],
    [1, 3],
    [3, 4],
    [4, 5]
])

offset_provider={"E2C": NeighborTableOffsetProvider(cell_neighbours_of_edges, EdgeDim, CellDim, 2)}
```

Let's create a field on the cells and fill it with some values:
![cell_values](connectivity_cell_field.svg)

```{code-cell} ipython3
cell_values = np_as_located_field(CellDim)(np.array([1.0, 1.0, 2.0, 3.0, 5.0, 8.0]))
```

#### Get value of 0th adjacent cell

The field operator `nearest_cell_to_edge` returns a field on the edges. The `E2C` field offset is used to map edge iterators to cell iterators using the connectivity matrix, and the cell iterator is used to extract the value from the cell field that's provided as input argument to the field operator. Note how `E2C` maps one edge iterator to two cell iterators (due to an edge having up to two cell neighbors). In this example, we take the 0th neighboring cell, but in the next one we will sum up the values of all neighboring cells.

```{code-cell} ipython3
@field_operator
def nearest_cell_to_edge(cells : Field[[CellDim], float32]) -> Field[[EdgeDim], float32]:
    return cells(E2C[0])

@program
def run_nearest_cell_to_edge(cells : Field[[CellDim], float32], out : Field[[EdgeDim], float32]):
    nearest_cell_to_edge(cells, out=out)
    
result_edge_values = np_as_located_field(EdgeDim)(np.zeros(shape=(12,)))

run_nearest_cell_to_edge(cell_values, result_edge_values, offset_provider=offset_provider)

print("0th adjacent cell's value: {}".format(np.asarray(result_edge_values)))
```

After running the code, we should see the following values assigned to the edges:

![nearest_cell_values](connectivity_edge_0th_cell.svg)

+++

#### Get the sum of the two cells adjacent to edges

This is very similar to the previous example, but instead of getting value of the 0th cell neighbor of an edge, we are going to sum all the neighboring cells. This is done by replacing the `E2C[0]` accessor by a `neighbor_sum` operation on the `E2C` field offset.

```{code-cell} ipython3
@field_operator
def sum_adjacent_cells(cells : Field[[CellDim], float32]) -> Field[[EdgeDim], float32]:
    return neighbor_sum(cells(E2C), axis=E2CDim)

@program
def run_sum_adjacent_cells(cells : Field[[CellDim], float32], out : Field[[EdgeDim], float32]):
    sum_adjacent_cells(cells, out=out)
    
result_edge_sums = np_as_located_field(EdgeDim)(np.zeros(shape=(12,)))

run_sum_adjacent_cells(cell_values, result_edge_sums, offset_provider=offset_provider)

print("sum of adjacent cells: {}".format(np.asarray(result_edge_sums)))
```

The results should be unchanged for the border edges, but the inner edge should be the following:

![cell_values](connectivity_edge_cell_sum.svg)

+++

Follow-up to averages:
- calculate the average of edges around a cell for each cell

```{code-cell} ipython3

```

```{code-cell} ipython3

```

## Examples

+++

### Space derivatives

```{code-cell} ipython3
@field_operator
def ddx():
    pass

@program
def compute_ddx():
    pass

get_ddx()
```

### Diffusion

```{code-cell} ipython3
@field_operator
def op1():
    pass

@field_operator
def op1():
    pass

@program
def diffuse():
    pass

timestep = 0.01
endtime = 5
for time in range(0:endtime:timestep):
    diffuse()
```
