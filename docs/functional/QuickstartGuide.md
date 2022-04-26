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
```bash
pip install git+https://github.com/gridtools/gt4py.git@functional
```

+++

## Concepts

### Fields, field operators, programs

Before introducing the main concepts, let us import a few things from GT4Py. These are all the constructs that will be used throughout the *concepts* section of this guide.

```{code-cell} ipython3
import numpy as np

from functional.ffront.fbuiltins import Field, float32, FieldOffset, neighbor_sum
from functional.iterator.runtime import CartesianAxis
from functional.ffront.decorator import field_operator, program
from functional.iterator.embedded import np_as_located_field, NeighborTableOffsetProvider
```

#### Fields

In GT4Py, multi-dimensional data is represented by *fields*. In the snippet below, we are declaring a two-dimensional field: one of the two axes is called *Cell*, while the other axis is called *K*. Both axes have to be declared as a `CartesianAxis`in GT4Py.

Conceptually, the *Cell* axis could represent an array of cells that form an unstructured grid to cover a surface, while the *K* axis could represent an array of arrays of cells to add vertical layering over the surface.

The fields themselves can be created using utility functions such as `np_as_located_field` that converts `numpy` arrays into GT4Py `Field`s. The code below creates two fields, both with 5 cells in the horizontal grid and 6 vertical layers, with all the 5\*6=30 values set to 2.0 for one field and 3.0 for the other field.

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

#### Field operators

In GT4Py, *field operators* are pure functions (i.e. functions without side effects) that take immutable `Field`s as arguments and output another `Field` as a result. Field operators are used to implement mathematical operations on fields. They must be declared with the `@field_operator` decorator, and are allowed to use a certain subset of the Python syntax.

The field operator below takes two fields as arguments, and returns the elementwise sum of the fields.

```{code-cell} ipython3
@field_operator
def add(a : Field[[CellDim, KDim], float32],
        b : Field[[CellDim, KDim], float32]) -> Field[[CellDim, KDim], float32]:
    return a + b
```

#### Programs

*Programs* are similar to fields operators, but they allow mutability of the arguments. Programs must be declared with the `@program` decorator and are allowed to use a different subset of the Python syntax compared to field operators. Programs are used to call and chain field operators:

```{code-cell} ipython3
@program
def run_add(a : Field[[CellDim, KDim], float32],
                b : Field[[CellDim, KDim], float32],
                out : Field[[CellDim, KDim], float32]):
    add(a, b, out=out)
```

Executing the program like a regular Python function will sum the two fields. The expectation is that every cell of the resulting field will be 2+3=5.

```{code-cell} ipython3
result = np_as_located_field(CellDim, KDim)(np.zeros(shape=grid_shape))
run_add(a, b, result, offset_provider={})

print("{} + {} = {} ± {}".format(a_value, b_value, np.average(np.asarray(result)), np.std(np.asarray(result))))
```

### Unstructured grids and connectivity

When using unstructured grids, we have to define adjacency between nodes, cells and edges manually. In this section, we will create the mesh illustrated below and we will do some calculations with fields over the edges and cells of this mesh.

![grid_topo](connectivity_numbered_grid.svg)

The <span style="color: #C02020">faces</span> and the <span style="color: #0080FF">edges</span> of the mesh have been numbered with zero-based indices.

#### Define grid and connectivity

We are going to use two fields: one on the cells of the grid, and on the edges. Both fields will have only one dimension, declared as `CellDim` for the field on the cells and `EdgeDim` for the field on the edges.

Furthermore, we will define the edge-to-cell connectivity that tells us which cells are neighbours to a particular edge. The connectivity is thus defined with a matrix where each line corresponds to an edge, and has 2 entries for the two cells to the side of that edge. (Missing neighbors for boundary edges are filled with -1.)

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

e2c_neighbor_table = NeighborTableOffsetProvider(cell_neighbours_of_edges, EdgeDim, CellDim, 2)
offset_provider={"E2C": e2c_neighbor_table}
```

Let's create a field on the cells and fill it with some values:
![cell_values](connectivity_cell_field.svg)

```{code-cell} ipython3
cell_values = np_as_located_field(CellDim)(np.array([1.0, 1.0, 2.0, 3.0, 5.0, 8.0]))
```

#### Using adjacencies in field operators

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

#### Using reductions on adjacencies

This is very similar to the previous example, but instead of getting value of the 0th cell neighbor of an edge, we are going to sum all the neighboring cells. This is done by replacing the `E2C[0]` accessor by a `neighbor_sum` operation on the `E2C` field offset.

```{code-cell} ipython3
@field_operator
def sum_adjacent_cells(cells : Field[[CellDim], float32]) -> Field[[EdgeDim], float32]:
    return neighbor_sum(cells(E2C), axis=E2CDim)

@program
def run_sum_adjacent_cells(cells : Field[[CellDim], float32], out : Field[[EdgeDim], float32]):
    sum_adjacent_cells(cells, out=out)
    
result_edge_sums = np_as_located_field(EdgeDim)(np.zeros(shape=(12,)))

#run_sum_adjacent_cells(cell_values, result_edge_sums, offset_provider=offset_provider)

print("sum of adjacent cells: {}".format(np.asarray(result_edge_sums)))
```

The results should be unchanged for the border edges, but the inner edge should be the following:

![cell_values](connectivity_edge_cell_sum.svg)

+++

Follow-up to averages:
- calculate the average of edges around a cell for each cell

+++

#### Using fields on connectivities, combining connectivities

This example similar to a structured grid laplacian kernel, where the average of the neighboring cells is subtracted from the current cell. In the case of our unstructured trianglular mesh, we will subtract the average of the cells that share an edge with our current cell.

Ultimately, we want to know the cell neighbors of cells. There are two way to go about this:
1. define an all-new cell-to-cell connectivity
2. define a cell-to-edge connectivity in addition to the edge-to-cell connectivity that we already have

Although defining cell-to-cell connectivity directly is simpler, we will choose the second approach to see how one can go from cell to the nearby edges, and from the nearby edges to the nearby cells, in turn reached cell neighbors of cells.

Let us start by defining the cell-to-edge connectivity. The 6-by-3 matrix below lists the 3 adjacent edges for each of the 6 cells:

```{code-cell} ipython3
C2EDim = CartesianAxis("C2E")
C2E = FieldOffset("C2E", source=EdgeDim, target=(CellDim, C2EDim))

edge_neighbors_of_cells = np.array([
    [0, 6, 7],
    [7, 8, 9],
    [1, 2, 8],
    [3, 9, 10],
    [4, 10, 11],
    [5, 6, 11],
])

c2e_neighbor_table = NeighborTableOffsetProvider(edge_neighbors_of_cells, CellDim, EdgeDim, 3)
```

To understand the procedure of combining the cell-to-edge and edge-to-cell connectivities, let's calculate the pseudo-laplacian by hand for cell index 3.

Cell index 3 has two neighbors: cell 1 and cell 4. The three cells contains the values 3, 1, 5, respectively. Therefore, the pseudo-laplacian will be 2*3 - (1 + 5) = 0.

Unfortunately, in absence of the cell-to-cell connectivities, we don't know that cells 1 and 4 are the neighbors of cell 3. However, based on the cell-to-edge connectivities, we do know that edges 3, 9 and 10 are adjacent to cell 3. Based on the edge-to-cell connectivities, we also know that edge 9 has cells 1 and 3 adjacent to it. With this information we can associate the difference \*(cell 1) - \*(cell 3) = 1 - 3 = -2 to edge 9. We could have also done \*(cell 3) - \*(cell 1), which would make the result +2. Similarly, we can calculate this difference for edge 10, which is ±2. Since edge 3 has only one cell adjacent to it, we won't calculate the difference, we will simply assume it's zero.

Now knowing the differences on all three edges adjacent to our chosen cell, we simply have to add the differences up: 0 - 2 + 2 = 0 -- the same value that we calculated by simply looking at the grid.

However, there are still two problems to tackle:
1. To **figure out the sign** in the sum of the three over-edge differences, we have to know if the difference on the edge was calculated with our target cell as first or second operand to the subtraction. We will record this information in a field over the cell-to-edge connectivity. This field will therefore have the cells as first dimension and the cell-to-edge axis as second, represented by a 6x3 matrix of -1s and +1s for the sign.

```{code-cell} ipython3
edge_difference_polarity = np.array([
    [1, 1, 1],   # cell 0
    [-1, 1, 1],  # cell 1
    [1, 1, -1],  # cell 2
    [1, -1, 1],  # cell 3
    [1, -1, 1],  # cell 4
    [1, -1, -1], # cell 5
])

edge_difference_polarity_field = np_as_located_field(CellDim, C2EDim)(edge_difference_polarity)
```

2. To make sure that **border edges get a difference of zero**, we will slightly modify the edge-to-cell connectivity matrix so that border edges list the single cells they are attached to twice. This will results in us calculating the difference on edge 3 as \*(cell 3) - \*(cell 3) = 0. The modified edge-to-cell connectivity matrix is as follows:

```{code-cell} ipython3
cell_neighbours_of_edges_mod = np.array([
    [0, 0], # edge 0
    [2, 2], # edge 1
    [2, 2], # edge 2
    [3, 3], # edge 3
    [4, 4], # edge 4
    [5, 5], # edge 5
    [0, 5], # edge 6
    [0, 1], # edge 7
    [1, 2], # edge 8
    [1, 3], # edge 9
    [3, 4], # edge 10
    [4, 5]  # edge 11
])

e2c_neighbor_table_mod = NeighborTableOffsetProvider(cell_neighbours_of_edges_mod, EdgeDim, CellDim, 2)
```

Note that the offset provider now needs to have both connectivities:

```{code-cell} ipython3
offset_provider={"E2C": e2c_neighbor_table_mod, "C2E": c2e_neighbor_table}
```

With all the connectivities and auxiliary data defined, we can write the corresponding field operator:

```{code-cell} ipython3
@field_operator
def edge_differences(cells : Field[[CellDim], float32]) -> Field[[EdgeDim], float32]:        
    return cells(E2C[0]) - cells(E2C[1])

@field_operator
def sum_differences(differences : Field[[EdgeDim], float32],
                    polarities : Field[[CellDim, C2EDim], float32]) -> Field[[CellDim], float32]:        
    return differences(C2E[0]) + differences(C2E[1]) + differences(C2E[2])
    # return differences(C2E[0]) + differences(C2E[1]) + differences(C2E[2])

@program
def run_pseudo_laplacian(cells : Field[[CellDim], float32],
                         polarities : Field[[CellDim, C2EDim], float32],
                         diffs : Field[[EdgeDim], float32],
                         out : Field[[CellDim], float32]):
    edge_differences(cells, out=diffs)
    sum_differences(diffs, polarities, out=out)

result_edge_diffs = np_as_located_field(EdgeDim)(np.zeros(shape=(12,)))
result_pseudo_lap = np_as_located_field(CellDim)(np.zeros(shape=(6,)))

run_pseudo_laplacian(cell_values,
                     edge_difference_polarity_field,
                     result_edge_diffs,
                     result_pseudo_lap,
                     offset_provider=offset_provider)

print("pseudo-laplacian: {}".format(np.asarray(result_pseudo_lap)))
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
