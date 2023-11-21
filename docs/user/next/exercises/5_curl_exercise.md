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

```{code-cell} ipython3
import numpy as np

import gt4py.next as gtx
from gt4py.next.iterator.embedded import MutableLocatedField 
from gt4py.next import neighbor_sum, where
from gt4py.next import Dimension, DimensionKind, FieldOffset
```

```{code-cell} ipython3
def random_field(
    sizes, *dims, low: float = -1.0, high: float = 1.0
) -> MutableLocatedField:
    return gtx.as_field([*dims],
        np.random.default_rng().uniform(
            low=low, high=high, size=sizes
        )
    )

def zero_field(
    sizes, *dims: Dimension, dtype=float
) -> MutableLocatedField:
    return gtx.as_field([*dims], 
        np.zeros(shape=sizes, dtype=dtype)
    )
```

For simplicity we use a triangulated donut in the horizontal.

```
0v---0e-- 1v---3e-- 2v---6e-- 0v
|  \ 0c   |  \ 1c   |  \2c
|   \1e   |   \4e   |   \7e
|2e   \   |5e   \   |8e   \
|  3c   \ |   4c  \ |    5c\
3v---9e-- 4v--12e-- 5v--15e-- 3v
|  \ 6c   |  \ 7c   |  \ 8c
|   \10e  |   \13e  |   \16e
|11e  \   |14e  \   |17e  \
|  9c  \  |  10c \  |  11c \
6v--18e-- 7v--21e-- 8v--24e-- 6v
|  \12c   |  \ 13c  |  \ 14c
|   \19e  |   \22e  |   \25e
|20e  \   |23e  \   |26e  \
|  15c  \ | 16c   \ | 17c  \
0v       1v         2v        0v
```

+++

## Connectivities

```{code-cell} ipython3
n_edges = 27
n_vertices = 9
n_cells = 18
n_levels = 10
e2c2v_table = np.asarray(
    [
        [0, 1, 4, 6],  # 0
        [0, 4, 1, 3],  # 1
        [0, 3, 4, 2],  # 2
        [1, 2, 5, 7],  # 3
        [1, 5, 2, 4],  # 4
        [1, 4, 5, 0],  # 5
        [2, 0, 3, 8],  # 6
        [2, 3, 5, 0],  # 7
        [2, 5, 1, 3],  # 8
        [3, 4, 0, 7],  # 9
        [3, 7, 4, 6],  # 10
        [3, 6, 7, 5],  # 11
        [4, 5, 8, 1],  # 12
        [4, 8, 7, 5],  # 13
        [4, 7, 3, 8],  # 14
        [5, 3, 6, 2],  # 15
        [6, 5, 3, 8],  # 16
        [8, 5, 6, 4],  # 17
        [6, 7, 3, 1],  # 18
        [6, 1, 7, 0],  # 19
        [6, 0, 1, 8],  # 20
        [7, 8, 2, 4],  # 21
        [7, 2, 8, 1],  # 22
        [7, 1, 2, 6],  # 23
        [8, 6, 0, 5],  # 24
        [8, 0, 6, 2],  # 25
        [8, 2, 0, 6],  # 26
    ]
)

e2c_table = np.asarray(
    [
        [0, 15],
        [0, 3],
        [3, 2],
        [1, 16],
        [1, 4],
        [0, 4],
        [2, 17],
        [2, 5],
        [1, 5],
        [3, 6],
        [6, 9],
        [9, 8],
        [4, 7],
        [7, 10],
        [6, 10],
        [5, 8],
        [8, 11],
        [7, 11],
        [9, 12],
        [12, 15],
        [15, 14],
        [10, 13],
        [13, 16],
        [12, 16],
        [11, 14],
        [14, 17],
        [13, 17],
    ]
)

e2v_table = np.asarray(
    [
        [0, 1],
        [0, 4],
        [0, 3],
        [1, 2],
        [1, 5],
        [1, 4],
        [2, 0],
        [2, 3],
        [2, 5],
        [3, 4],
        [3, 7],
        [3, 6],
        [4, 5],
        [4, 8],
        [4, 7],
        [5, 3],
        [5, 6],
        [5, 8],
        [6, 7],
        [6, 1],
        [6, 0],
        [7, 8],
        [7, 2],
        [7, 1],
        [8, 6],
        [8, 0],
        [8, 2],
    ]
)

e2c2e_table = np.asarray(
    [
        [1, 5, 19, 20],
        [0, 5, 2, 9],
        [1, 9, 6, 7],
        [4, 8, 22, 23],
        [3, 8, 5, 12],
        [0, 1, 4, 12],
        [7, 2, 25, 26],
        [6, 2, 8, 15],
        [3, 4, 7, 15],
        [1, 2, 10, 14],
        [9, 14, 11, 18],
        [10, 18, 15, 16],
        [4, 5, 13, 17],
        [12, 17, 14, 21],
        [9, 10, 13, 21],
        [7, 8, 16, 11],
        [15, 11, 17, 24],
        [12, 13, 16, 24],
        [10, 11, 19, 23],
        [18, 23, 20, 0],
        [19, 0, 24, 25],
        [13, 14, 22, 26],
        [21, 26, 23, 3],
        [18, 19, 22, 3],
        [16, 17, 25, 20],
        [24, 20, 26, 6],
        [25, 6, 21, 22],
    ]
)

e2c2eO_table = np.asarray(
    [
        [0, 1, 5, 19, 20],
        [0, 1, 5, 2, 9],
        [1, 2, 9, 6, 7],
        [3, 4, 8, 22, 23],
        [3, 4, 8, 5, 12],
        [0, 1, 5, 4, 12],
        [6, 7, 2, 25, 26],
        [6, 7, 2, 8, 15],
        [3, 4, 8, 7, 15],
        [1, 2, 9, 10, 14],
        [9, 10, 14, 11, 18],
        [10, 11, 18, 15, 16],
        [4, 5, 12, 13, 17],
        [12, 13, 17, 14, 21],
        [9, 10, 14, 13, 21],
        [7, 8, 15, 16, 11],
        [15, 16, 11, 17, 24],
        [12, 13, 17, 16, 24],
        [10, 11, 18, 19, 23],
        [18, 19, 23, 20, 0],
        [19, 20, 0, 24, 25],
        [13, 14, 21, 22, 26],
        [21, 22, 26, 23, 3],
        [18, 19, 23, 22, 3],
        [16, 17, 24, 25, 20],
        [24, 25, 20, 26, 6],
        [25, 26, 6, 21, 22],
    ]
)

c2e_table = np.asarray(
    [
        [0, 1, 5],  # cell 0
        [3, 4, 8],  # cell 1
        [6, 7, 2],  # cell 2
        [1, 2, 9],  # cell 3
        [4, 5, 12],  # cell 4
        [7, 8, 15],  # cell 5
        [9, 10, 14],  # cell 6
        [12, 13, 17],  # cell 7
        [15, 16, 11],  # cell 8
        [10, 11, 18],  # cell 9
        [13, 14, 21],  # cell 10
        [16, 17, 24],  # cell 11
        [18, 19, 23],  # cell 12
        [21, 22, 26],  # cell 13
        [24, 25, 20],  # cell 14
        [19, 20, 0],  # cell 15
        [22, 23, 3],  # cell 16
        [25, 26, 6],  # cell 17
    ]
)

v2c_table = np.asarray(
    [
        [17, 14, 3, 0, 2, 15],
        [0, 4, 1, 12, 16, 15],
        [1, 5, 2, 16, 13, 17],
        [3, 6, 9, 5, 8, 2],
        [6, 10, 7, 4, 0, 3],
        [7, 11, 8, 5, 1, 4],
        [9, 12, 15, 8, 11, 14],
        [12, 16, 13, 10, 6, 9],
        [13, 17, 14, 11, 7, 10],
    ]
)

v2e_table = np.asarray(
    [
        [0, 1, 2, 6, 25, 20],
        [3, 4, 5, 0, 23, 19],
        [6, 7, 8, 3, 22, 26],
        [9, 10, 11, 15, 7, 2],
        [12, 13, 14, 9, 1, 5],
        [15, 16, 17, 12, 4, 8],
        [18, 19, 20, 24, 16, 11],
        [21, 22, 23, 18, 10, 14],
        [24, 25, 26, 21, 13, 17],
    ]
)

diamond_table = np.asarray(
    [
        [0, 1, 4, 6],  # 0
        [0, 4, 1, 3],
        [0, 3, 4, 2],
        [1, 2, 5, 7],  # 3
        [1, 5, 2, 4],
        [1, 4, 5, 0],
        [2, 0, 3, 8],  # 6
        [2, 3, 0, 5],
        [2, 5, 1, 3],
        [3, 4, 0, 7],  # 9
        [3, 7, 4, 6],
        [3, 6, 5, 7],
        [4, 5, 1, 8],  # 12
        [4, 8, 5, 7],
        [4, 7, 3, 8],
        [5, 3, 2, 6],  # 15
        [5, 6, 3, 8],
        [5, 8, 4, 6],
        [6, 7, 3, 1],  # 18
        [6, 1, 7, 0],
        [6, 0, 1, 8],
        [7, 8, 4, 2],  # 21
        [7, 2, 8, 1],
        [7, 1, 6, 2],
        [8, 6, 5, 0],  # 24
        [8, 0, 6, 2],
        [8, 2, 7, 0],
    ]
)

c2e2cO_table = np.asarray(
    [
        [15, 4, 3, 0],
        [16, 5, 4, 1],
        [17, 3, 5, 2],
        [0, 6, 2, 3],
        [1, 7, 0, 4],
        [2, 8, 1, 5],
        [3, 10, 9, 6],
        [4, 11, 10, 7],
        [5, 9, 11, 8],
        [6, 12, 8, 9],
        [7, 13, 6, 10],
        [8, 14, 7, 11],
        [9, 16, 15, 12],
        [10, 17, 16, 13],
        [11, 15, 17, 14],
        [12, 0, 14, 15],
        [13, 1, 12, 16],
        [14, 2, 13, 17],
    ]
)

c2e2c_table = np.asarray(
    [
        [15, 4, 3],
        [16, 5, 4],
        [17, 3, 5],
        [0, 6, 2],
        [1, 7, 0],
        [2, 8, 1],
        [3, 10, 9],
        [4, 11, 10],
        [5, 9, 11],
        [6, 12, 8],
        [7, 13, 6],
        [8, 14, 7],
        [9, 16, 15],
        [10, 17, 16],
        [11, 15, 17],
        [12, 0, 14],
        [13, 1, 12],
        [14, 2, 13],
    ]
)
```

```{code-cell} ipython3
C = Dimension("C")
V = Dimension("V")
E = Dimension("E")
K = Dimension("K")
```

## 2. reduction: gradient

+++

Next we will translate a divergence stencil. The normal velocity of each edge is multipled with the edge length, the contributions from all three edges of a cell are summed up and then divided by the area of the cell. In the next pictures we can see a graphical representation of all of the quantities involved:
![](../divergence.png "Divergence")
The orientation of the edge plays a role for this operation in ICON, as we need to be aware if the normal vector of an edge points inwards or outwards of the cell we are currently looking at.
![](../edge_orientation.png "Edge Orientation")
One such divergence stencil is stencil 02 in diffusion:

```fortran
      DO jb = i_startblk,i_endblk

        CALL get_indices_c(p_patch, jb, i_startblk, i_endblk, &
                           i_startidx, i_endidx, rl_start, rl_end)
        DO jk = 1, nlev
          DO jc = i_startidx, i_endidx

            div(jc,jk) = p_nh_prog%vn(ieidx(jc,jb,1),jk,ieblk(jc,jb,1))*p_int%geofac_div(jc,1,jb) + &
                         p_nh_prog%vn(ieidx(jc,jb,2),jk,ieblk(jc,jb,2))*p_int%geofac_div(jc,2,jb) + &
                         p_nh_prog%vn(ieidx(jc,jb,3),jk,ieblk(jc,jb,3))*p_int%geofac_div(jc,3,jb)
          ENDDO
        ENDDO
      ENDDO
```
where `p_int%geofac_div` is set up as a constant field at ICON startup time and contains the geometrical factors for the divergence operator:
```fortran
    DO jb = i_startblk, i_endblk

      CALL get_indices_c(ptr_patch, jb, i_startblk, i_endblk, &
        & i_startidx, i_endidx, rl_start, rl_end)

      DO je = 1, ptr_patch%geometry_info%cell_type
        DO jc = i_startidx, i_endidx

          ile = ptr_patch%cells%edge_idx(jc,jb,je)
          ibe = ptr_patch%cells%edge_blk(jc,jb,je)

          ptr_int%geofac_div(jc,je,jb) = &
            & ptr_patch%edges%primal_edge_length(ile,ibe) * &
            & ptr_patch%cells%edge_orientation(jc,jb,je)  / &
            & ptr_patch%cells%area(jc,jb)

        ENDDO !cell loop
      ENDDO

    END DO !block loop

```

```{code-cell} ipython3
C2EDim = Dimension("C2E", kind=DimensionKind.LOCAL)
C2E = FieldOffset("C2E", source=E, target=(C, C2EDim))
V2EDim = Dimension("V2E", kind=DimensionKind.LOCAL)
V2E = FieldOffset("V2E", source=E, target=(V, V2EDim))
```

```{code-cell} ipython3
def curl_numpy(
    v2e: np.array,
    u: np.array,
    v: np.array,
    nx: np.array,
    ny: np.array,
    dualL: np.array,
    dualA: np.array,
    edge_orientation: np.array,
) -> np.array:
    uv_curl = np.sum((u[v2e]*nx[v2e] + v[v2e]*ny[v2e]) * dualL[v2e] * edge_orientation, axis=1) / dualA

    return uv_curl
```

```{code-cell} ipython3
@gtx.field_operator
def curl(
    u: gtx.Field[[E], float],
    v: gtx.Field[[E], float],
    nx: gtx.Field[[E], float],
    ny: gtx.Field[[E], float],
    dualL: gtx.Field[[E], float],
    dualA: gtx.Field[[V], float],
    edge_orientation: gtx.Field[[V, V2EDim], float],
) -> gtx.Field[[V], float]:
    uv_curl = neighbor_sum((u(V2E)*nx(V2E) + v(V2E)*ny(V2E)) * dualL(V2E) * edge_orientation, axis=V2EDim) / dualA
    
    return uv_curl
```

```{code-cell} ipython3
def test_curl():
    u = random_field((n_edges), E)
    v = random_field((n_edges), E)
    nx = random_field((n_edges), E)
    ny = random_field((n_edges), E)
    dualL = random_field((n_edges), E)
    dualA = random_field((n_vertices), V)
    edge_orientation = random_field((n_vertices, 6), V, V2EDim)

    divergence_ref = curl_numpy(
        v2e_table,
        np.asarray(u),
        np.asarray(v),
        np.asarray(nx),
        np.asarray(ny),
        np.asarray(dualL),
        np.asarray(dualA),
        np.asarray(edge_orientation),
    )

    v2e_connectivity = gtx.NeighborTableOffsetProvider(v2e_table, V, E, 6)

    curl_gt4py = zero_field((n_vertices), V)

    curl(
        u, v, nx, ny, dualL, dualA, edge_orientation, out = curl_gt4py, offset_provider = {V2E.value: v2e_connectivity}
    )
    
    assert np.allclose(curl_gt4py, divergence_ref)
```

```{code-cell} ipython3
test_curl()
print("Test successful")
```

```{code-cell} ipython3

```
