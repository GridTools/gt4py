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

# GT4Py Declarative Frontend - Quickstart Guide

This guide introduces basic concepts of the new declarative version of GT4Py.

+++

## Installation

The library is distributed as a Python package and can be installed directly from GitHub via pip:

```{raw-cell}
pip install git+https://github.com/gridtools/gt4py.git@functional
```

For now, use the below branch which contains sub bugfixes needed to run the code in the guide:

```{raw-cell}
pip install git+https://github.com/tehrengruber/gt4py.git@fix_reduction
```

## Programming GT4Py


### Basic structure of an application using the library

In this section, we will write a simple application that adds two arrays. The goal is to understand how represent data in the form of fields and perform basic operations between them.

+++

#### Importing features

The following snippet imports the most commonly used functionality from the library. These are all that's needed to run all the code snippets in this document. Numpy is also used for the examples.

```{code-cell} ipython3

```

```{code-cell} ipython3
import numpy as np

from functional.ffront.fbuiltins import Dimension, Field, float32, FieldOffset, neighbor_sum
from functional.ffront.decorator import field_operator, program
from functional.iterator.embedded import np_as_located_field, NeighborTableOffsetProvider
```

#### Fields

To represent array-like data, *fields* defined over one or more *dimensions* are used. As seen in the following code snippet, the dimensions are defined as a `Dimension`, whereas the fields are created with helper functions such as `np_as_located_field`. The 2D fields used in this section are defined over the *Cell* and *K* dimensions, have a size of 5 cells by 6 Ks, and are uniformly filled with the values 2 and 3:

```{code-cell} ipython3
CellDim = Dimension("Cell")
KDim = Dimension("K")

num_cells = 5
num_layers = 6
grid_shape = (num_cells, num_layers)

a_value = 2.0
b_value = 3.0
a = np_as_located_field(CellDim, KDim)(np.full(shape=grid_shape, fill_value=a_value, dtype=np.float32))
b = np_as_located_field(CellDim, KDim)(np.full(shape=grid_shape, fill_value=b_value, dtype=np.float32))
```

#### Field operators

Most often you will be working with fields of data rather than single scalar elements. The operations over fields can be defined using *field operators*, which are essentially pure functions that can take some fields as immutable arguments and return another field as result. Because of optimizations, field operators are only allowed to use a subset of the Python syntax. Syntactical correctness is checked by the `@field_operator` decorator.

This field operator returns the sum of the two arguments:

```{code-cell} ipython3
@field_operator
def add(a : Field[[CellDim, KDim], float32],
        b : Field[[CellDim, KDim], float32]) -> Field[[CellDim, KDim], float32]:
    return a + b
```

Running the field operator should produce a `result` of which all the elements are equal to 2+3=5:

```{code-cell} ipython3
result = np_as_located_field(CellDim, KDim)(np.zeros(shape=grid_shape))
add(a, b, out=result, offset_provider={})

print("{} + {} = {} ± {}".format(a_value, b_value, np.average(np.asarray(result)), np.std(np.asarray(result))))
```

#### Programs

+++

Multiple field operator calls can be grouped together to create a *program* that unlike field operators allow mutability of the arguments. The only operations allowed within programs right now are field operator calls. Later on this might be extended for other stateful operations. As a guideline one is advised to write as much code as possible inside field operators, as that enhances the optimization potential.

```{code-cell} ipython3
@program
def run_add(a : Field[[CellDim, KDim], float32],
                b : Field[[CellDim, KDim], float32],
                out : Field[[CellDim, KDim], float32]):
    add(a, b, out=out)
    add(b, out, out=out)
```

Executing this program should result in a field filled with the value 8:

```{code-cell} ipython3
result = np_as_located_field(CellDim, KDim)(np.zeros(shape=grid_shape))
run_add(a, b, result, offset_provider={})

print("{} + {} = {} ± {}".format(b_value, (a_value + b_value), np.average(np.asarray(result)), np.std(np.asarray(result))))
```

### Operations on unstructured meshes

In this section, we will write a slightly more elaborate application that performs a laplacian-like operation on an unstructured mesh to introduce additional APIs. We will define the *pseudo-laplacian* as the sum of the differences between the values on neighboring cells with the current cell, where two cells are said to be neighboring if they share a common edge.

For example, if cell \#1 has three neighbors, cell \#0, \#2 and \#3, its pseudo-laplacian is $$\begin{aligned}\text{pseudolap}(cell_1) =\,& (\text{value_of}(\text{cell}_1) - \text{value_of}(\text{cell}_0)) \\
&+ (\text{value_of}(\text{cell}_1)-\text{value_of}(\text{cell}_2)) \\
& + (\text{value_of}(\text{cell}_1)-\text{value_of}(\text{cell}_3))\end{aligned}$$.

This section is broken down into smaller parts to introduce concepts required for the pseudo-laplacian bit by bit:
- Defining the mesh and the connectivities (adjacencies) between cells and edges
- Learning to apply connectivities within field operators
- Learning to use reductions on adjacent mesh elements
- Implementing the actual pseudo-laplacian

+++

#### Defining the mesh and its connectivities

Consider the following mesh, of which the <span style="color: #C02020">cells</span> and the <span style="color: #0080FF">edges</span> have been numbered:

| ![grid_topo](connectivity_numbered_grid.svg) |
|:--:| 
| *Cell and edge indices* |

+++

For our examples, we are concerned only with fields over cells and edges, for which we declare two independent dimensions:

```{code-cell} ipython3
CellDim = Dimension("Cell")
EdgeDim = Dimension("Edge")
```

In addition to the cells and edges, we will also define the connectivities: one table for the edges reachable from a cell and another table for the cells reachable from an edge. The $i$th entry of the connectivity table contains the indices of the <span style="color: #C02020">cells</span> (<span style="color: #0080FF">edges</span>) adjacent to the $i$th <span style="color: #0080FF">edge</span> (<span style="color: #C02020">cell</span>). The connectivity tables for the mesh above are:

```{code-cell} ipython3
edge_to_cell_table = np.array([
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

cell_to_edge_table = np.array([
    [0, 6, 7],
    [7, 8, 9],
    [1, 2, 8],
    [3, 9, 10],
    [4, 10, 11],
    [5, 6, 11],
])
```

#### Using connectivities in field operators

Let's start by defining two fields: one over the cells and another one over the edges. The field over cells will be used as input for subsequent calculations and is therefore filled up with values, whereas the field over the edges will be used to store the results and is therefore left blank.

```{code-cell} ipython3
cell_values = np_as_located_field(CellDim)(np.array([1.0, 1.0, 2.0, 3.0, 5.0, 8.0]))
edge_values = np_as_located_field(EdgeDim)(np.zeros((12,)))
```

| ![cell_values](connectivity_cell_field.svg) | 
|:--:| 
| *Cell values* |

+++

*Field offsets* are used to transform fields of one domain to a field of another domain using connectivities. The field offset declared in the snippet below is used to transform a one-dimensional field over cells to a two-dimensional field over edges and an auxiliary dimension (`E2CDim`) using the edge-to-cell connectivities.

One way to understand this transform is to look at the edge-to-cell connectivity table `edge_to_cell_table` defined above. This table has one dimension over the edges and another auxiliary dimension, just like the output of the transformation, and stores indices into a field over cells. Now replace the indices with the actual values taken from a field over cells, and you get the result of the transform.

Another way is to say that transform uses the edge-to-cell connectivity to look up all the cell neighbors of edges, and associates the values of those neighbor cells with each edge.

```{code-cell} ipython3
E2CDim = Dimension("E2C")
E2C = FieldOffset("E2C", source=CellDim, target=(EdgeDim, E2CDim))
```

While the field offset specifies the source and target dimensions of the transform, the actual connectivity table is provided separately through an *offset provider*:

```{code-cell} ipython3
E2C_offset_provider = NeighborTableOffsetProvider(edge_to_cell_table, EdgeDim, CellDim, 2)
```

The field operator below uses the field offset and the transform explained above to create a field on the edges from the field on cells we just defined. Note that we would have two elements for every edge (for a non-border edge has exactly two neighbor cells), but we only keep the first neighbor cell's value in the field operator.

While the field offset `E2C` is simply accessed within the field operator, the offset provider `E2C_offset_provider` is passed via the call to the program.

```{code-cell} ipython3
@field_operator
def nearest_cell_to_edge(cell_values : Field[[CellDim], float32]) -> Field[[EdgeDim], float32]:
    return cell_values(E2C[0])

@program
def run_nearest_cell_to_edge(cell_values : Field[[CellDim], float32], out : Field[[EdgeDim], float32]):
    nearest_cell_to_edge(cell_values, out=out)

run_nearest_cell_to_edge(cell_values, edge_values, offset_provider={"E2C": E2C_offset_provider})

print("0th adjacent cell's value: {}".format(np.asarray(edge_values)))
```

| ![nearest_cell_values](connectivity_edge_0th_cell.svg) |
|:--:| 
| *Resulting edge values* |

+++

#### Reductions on connectivities

Similarly to the previous example, we will yet again output a field on edges. This time, however, instead of taking the first column (first neighbour) out of the field created using the connectivity table, we will sum the elements alongside the `E2CDim`. For this, we can use the `neighbor_sum` builtin function of GT4Py:

```{code-cell} ipython3
@field_operator
def sum_adjacent_cells(cells : Field[[CellDim], float32]) -> Field[[EdgeDim], float32]:
    return neighbor_sum(cells(E2C), axis=E2CDim)

@program
def run_sum_adjacent_cells(cells : Field[[CellDim], float32], out : Field[[EdgeDim], float32]):
    sum_adjacent_cells(cells, out=out)
    
run_sum_adjacent_cells(cell_values, edge_values, offset_provider={"E2C": E2C_offset_provider})

print("sum of adjacent cells: {}".format(np.asarray(edge_values)))
```

The results should be unchanged for the border edges, but the inner edges should now contains the sum of the two adjacent cells:

| ![cell_values](connectivity_edge_cell_sum.svg) |
|:--:| 
| *Resulting edge values* |

+++

#### Implementing the pseudo-laplacian by combining the above

As explained in the section outline, we will need the cell-to-edge connectivities as well. We have already constructed the connectivity table, so now we only have to define the local dimension, the field offset and the offset provider that describe how to use the connectivity matrix. The procedure is identical to the edge-to-cell connectivity we covered before:

```{code-cell} ipython3
C2EDim = Dimension("C2E", True)
C2E = FieldOffset("C2E", source=EdgeDim, target=(CellDim, C2EDim))

C2E_offset_provider = NeighborTableOffsetProvider(cell_to_edge_table, CellDim, EdgeDim, 3)
```

+++ {"tags": []}

**Sign of edge differences:**

Revisiting the example from the beginning of the section, except now with the actual mesh, we can calculate the pseudo-laplacian for cell \#1 by the following equation:
$$\text{plap}(cell_1) = -\text{edge_diff}_{0,1} + \text{edge_diff}_{1,2} + \text{edge_diff}_{1,3}$$

Notice how $\text{edge_diff}_{0,1}$ is actually subtracted from the sum rather than added because the edge to cell connectivity table lists cell \#1 as the second argument rather than the first. To fix this, we will need a table that has 3 elements for every cell to tell the sign of the differences. If you look for cell \#1 in the table below, you will see that the sign is negative for the first edge difference and positive for the second and third, just like in the equation above.

```{code-cell} ipython3
edge_difference_signs = np.array([
    [1, 1, 1],   # cell 0
    [-1, 1, 1],  # cell 1
    [1, 1, -1],  # cell 2
    [1, -1, 1],  # cell 3
    [1, -1, 1],  # cell 4
    [1, -1, -1], # cell 5
])

edge_difference_sign_field = np_as_located_field(CellDim, C2EDim)(edge_difference_signs)
```

**Difference on border edges:**

We cannot actually calculate an edge difference on border edges, because they only have one cell neighbor. For the calculation of the pseudo-laplacian, we want to consider border edges to have a difference of zero. We can achieve this by modifying the edge to cell connectivity so that for the border edges that single neighbor cell is listed twice. This way, as we subtract the value of that single cell from itself, we will get zero. The modified table is the following:

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

E2C_offset_provider_mod = NeighborTableOffsetProvider(cell_neighbours_of_edges_mod, EdgeDim, CellDim, 2)
```

**TODO**: The code:

```{code-cell} ipython3
@field_operator
def pseudo_lap(cells : Field[[CellDim], float32],
               polarities : Field[[CellDim, C2EDim], float32]) -> Field[[CellDim], float32]:
    edge_differences = cells(E2C[0]) - cells(E2C[1])
    return neighbor_sum(edge_differences(C2E) * polarities, axis=C2EDim)

@field_operator
def pseudo_laplap(cells : Field[[EdgeDim], float32],
                  polarities : Field[[CellDim, C2EDim], float32]) -> Field[[CellDim], float32]:
    return pseudo_lap(pseudo_lap(cells, polarities), polarities)

@program
def run_pseudo_laplacian(cells : Field[[CellDim], float32],
                         polarities : Field[[CellDim, C2EDim], float32],
                         out : Field[[CellDim], float32]):
    pseudo_lap(cells, polarities, out=out)

result_pseudo_lap = np_as_located_field(CellDim)(np.zeros(shape=(6,)))

run_pseudo_laplacian(cell_values,
                     edge_difference_sign_field,
                     result_pseudo_lap,
                     offset_provider={"E2C": E2C_offset_provider_mod, "C2E": C2E_offset_provider})

print("pseudo-laplacian: {}".format(np.asarray(result_pseudo_lap)))
```
