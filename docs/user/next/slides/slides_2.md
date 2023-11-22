---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

<img src="logos/cscs_logo.jpeg" alt="cscs" style="width:270px;"/> <img src="logos/c2sm_logo.gif" alt="c2sm" style="width:220px;"/>
<img src="logos/exclaim_logo.png" alt="exclaim" style="width:270px;"/> <img src="logos/mch_logo.svg" alt="mch" style="width:270px;"/>

```{code-cell} ipython3
import warnings
warnings.filterwarnings('ignore')
```

```{code-cell} ipython3
import numpy as np
import gt4py.next as gtx
from gt4py.next import float64, neighbor_sum, where
from gt4py.next.common import DimensionKind
from gt4py.next.program_processors.runners import roundtrip
```

```{code-cell} ipython3
CellDim = gtx.Dimension("Cell")
KDim = gtx.Dimension("K", kind=DimensionKind.VERTICAL)
grid_shape = (5, 6)
```

## Offsets
Fields can be offset by a predefined number of indices.

Take an array with values ranging from 0 to 5:

```{code-cell} ipython3
a_off = gtx.as_field([CellDim], np.array([1.0, 1.0, 2.0, 3.0, 5.0, 8.0]))

print("a_off array: \n {}".format(a_off.asnumpy()))
```

Visually, offsetting this field by 1 would result in the following:

| ![Coff](../simple_offset.png) |
| :------------------------: |
|  _CellDim Offset (Coff)_   |

+++

Fields can be offset by a predefined number of indices.

Take an array with values ranging from 0 to 5:

```{code-cell} ipython3
Coff = gtx.FieldOffset("Coff", source=CellDim, target=(CellDim,))

@gtx.field_operator
def a_offset(a_off: gtx.Field[[CellDim], float64]) -> gtx.Field[[CellDim], float64]:
    return a_off(Coff[1])
    
a_offset(a_off, out=a_off[:-1], offset_provider={"Coff": CellDim})
print("result array: \n {}".format(a_off.asnumpy()))
```

## Defining the mesh and its connectivities
Take an unstructured mesh with numbered cells (in red) and edges (in blue).

| ![grid_topo](../connectivity_numbered_grid.svg) |
| :------------------------------------------: |
|         _The mesh with the indices_          |

```{code-cell} ipython3
CellDim = gtx.Dimension("Cell")
EdgeDim = gtx.Dimension("Edge")
```

Connectivity among mesh elements is expressed through connectivity tables.

For example, `e2c_table` lists for each edge its adjacent rows. 

Similarly, `c2e_table` lists the edges that are neighbors to a particular cell.

Note that if an edge is lying at the border, one entry will be filled with -1.

```{code-cell} ipython3
e2c_table = np.array([
    [0, -1], # edge 0 (neighbours: cell 0)
    [2, -1], # edge 1
    [2, -1], # edge 2
    [3, -1], # edge 3
    [4, -1], # edge 4
    [5, -1], # edge 5
    [0, 5],  # edge 6 (neighbours: cell 0, cell 5)
    [0, 1],  # edge 7
    [1, 2],  # edge 8
    [1, 3],  # edge 9
    [3, 4],  # edge 10
    [4, 5]   # edge 11
])

c2e_table = np.array([
    [0, 6, 7],   # cell 0 (neighbors: edge 0, edge 6, edge 7)
    [7, 8, 9],   # cell 1
    [1, 2, 8],   # cell 2
    [3, 9, 10],  # cell 3
    [4, 10, 11], # cell 4
    [5, 6, 11],  # cell 5
])
```

#### Using connectivities in field operators

Let's start by defining two fields: one over the cells and another one over the edges. The field over cells serves input for subsequent calculations and is therefore filled up with values, whereas the field over the edges stores the output of the calculations and is therefore left blank.

```{code-cell} ipython3
cell_field = gtx.as_field([CellDim], np.array([1.0, 1.0, 2.0, 3.0, 5.0, 8.0]))
edge_field = gtx.as_field([EdgeDim], np.zeros((12,)))
```

| ![cell_values](../connectivity_cell_field.svg) |
| :-----------------------------------------: |
|                _Cell values_                |

+++

`field_offset` is used to remap fields over one domain to another domain, e.g. cells -> edges.

Field remappings are just composition of mappings
- Field defined on cells: $f_C: C \to \mathbb{R}$
- Connectivity from _edges to cells_: $c_{E \to C_0}$
- We define a new field on edges composing both mappings
$$ f_E: E \to \mathbb{R}, e \mapsto (f_C \circ c_{E \to C_0})(e) := f_c(c_{E \to C_0}(e)) $$
- In point-free notation: $f_E = f_C(c_{E \to C_0}) \Rightarrow$ `f_c(E2C[0])`


We extend the connectivities to refer to more than just one neighbor
- `E2C` is the local dimension of all cell neighbors of an edge

$$ c_{E \to C}: E \times E2C \to C $$
$$ f_E: E \to \mathbb{R}, e \mapsto \big(f_C \circ c_{E \to C}\big)(e, 0) $$
$$ f_E(e, c) := f_C(c_{E \to C}(e, c)), e \in E, c \in \{0,1\} $$

```{code-cell} ipython3
E2CDim = gtx.Dimension("E2C", kind=gtx.DimensionKind.LOCAL)
E2C = gtx.FieldOffset("E2C", source=CellDim, target=(EdgeDim, E2CDim))
```

```{code-cell} ipython3
E2C_offset_provider = gtx.NeighborTableOffsetProvider(e2c_table, EdgeDim, CellDim, 2)
```

```{code-cell} ipython3
@gtx.field_operator
def nearest_cell_to_edge(cell_field: gtx.Field[[CellDim], float64]) -> gtx.Field[[EdgeDim], float64]:
    return cell_field(E2C[0]) # 0th index to isolate edge dimension

@gtx.program(backend=roundtrip.executor) # TODO uses skip_values, therefore cannot use embedded
def run_nearest_cell_to_edge(cell_field: gtx.Field[[CellDim], float64], edge_field: gtx.Field[[EdgeDim], float64]):
    nearest_cell_to_edge(cell_field, out=edge_field)

run_nearest_cell_to_edge(cell_field, edge_field, offset_provider={"E2C": E2C_offset_provider})

print("0th adjacent cell's value: {}".format(edge_field.asnumpy()))
```

Running the above snippet results in the following edge field:

| ![nearest_cell_values](../connectivity_numbered_grid.svg) | $\mapsto$ | ![grid_topo](../connectivity_edge_0th_cell.svg) |
| :----------------------------------------------------: | :-------: | :------------------------------------------: |
|                    _Domain (edges)_                    |           |                _Edge values_                 |

+++

### Using reductions on connected mesh elements

To sum up all the cells adjacent to an edge the `neighbor_sum` builtin function can be called to operate along the `E2CDim` dimension.

```{code-cell} ipython3
@gtx.field_operator
def sum_adjacent_cells(cell_field : gtx.Field[[CellDim], float64]) -> gtx.Field[[EdgeDim], float64]:
    return neighbor_sum(cell_field(E2C), axis=E2CDim)

@gtx.program(backend=roundtrip.executor) # TODO uses skip_values, therefore cannot use embedded
def run_sum_adjacent_cells(cell_field : gtx.Field[[CellDim], float64], edge_field: gtx.Field[[EdgeDim], float64]):
    sum_adjacent_cells(cell_field, out=edge_field)

run_sum_adjacent_cells(cell_field, edge_field, offset_provider={"E2C": E2C_offset_provider})

print("sum of adjacent cells: {}".format(edge_field.asnumpy()))
```

For the border edges, the results are unchanged compared to the previous example, but the inner edges now contain the sum of the two adjacent cells:

| ![nearest_cell_values](../connectivity_numbered_grid.svg) | $\mapsto$ | ![cell_values](../connectivity_edge_cell_sum.svg) |
| :----------------------------------------------------: | :-------: | :--------------------------------------------: |
|                    _Domain (edges)_                    |           |                 _Edge values_                  |

```{code-cell} ipython3

```
