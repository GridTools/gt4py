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

```{code-cell} ipython3
from helpers import *

import gt4py.next as gtx
```

```{code-cell} ipython3
def diffusion_step_numpy(
    e2c2v: np.array,
    v2e: np.array,
    TE: np.array,
    TE_t: np.array,
    inv_primal_edge_length: np.array,
    inv_vert_vert_length: np.array,
    nnbhV: np.array,
    boundary_edge: np.array,
    kappa: float,
    dt: float,
) -> np.array:

    # initialize
    TEinit = TE

    # predict
    TE = TEinit + 0.5*dt*TE_t

    # interpolate temperature from edges to vertices
    TV = neighbor_sum(TE(v2e), axis=1) / nnbhV

    # compute nabla2 using the finite differences
    TEnabla2 = neighbor_sum(
        (TV(e2c2v[0]) + TV(e2c2v[1])) * inv_primal_edge_length ** 2
        + (TV(e2c2v[3]) + TV(e2c2v[4])) * inv_vert_vert_length ** 2,
        axis=1,
    )

    TEnabla2 = TEnabla2 - (
        (2.0 * TE * inv_primal_edge_length ** 2)
        + (2.0 * TE * inv_vert_vert_length ** 2)
    )

    # build ODEs
    TE_t = where(
        boundary_edge,
        0.,
        kappa*TEnabla2,
    )

    # correct
    TE = TEinit + dt*TE_t
```

```{code-cell} ipython3
@gtx.field_operator
def diffusion_step(
    TE: gtx.Field[Dims[E], float],
    TE_t: gtx.Field[Dims[E], float],
    inv_primal_edge_length: gtx.Field[Dims[E], float],
    inv_vert_vert_length: gtx.Field[Dims[E], float],
    nnbhV: gtx.Field[Dims[V], float],
    boundary_edge: gtx.Field[Dims[E], bool],
    kappa: float,
    dt: float,
) -> gtx.tuple[
    gtx.Field[Dims[E], float],
    gtx.Field[Dims[E], float],
]:

    # initialize
    TEinit = TE

    # predict
    TE = TEinit + 0.5*dt*TE_t

    # interpolate temperature from edges to vertices
    TV = neighbor_sum(TE(V2E), axis=V2EDim) / nnbhV

    # compute nabla2 using the finite differences
    TEnabla2 = neighbor_sum(
        (TV(E2C2V[0]) + TV(E2C2V[1])) * inv_primal_edge_length ** 2
        + (TV(E2C2V[3]) + TV(E2C2V[4])) * inv_vert_vert_length ** 2,
        axis=E2C2VDim,
    )

    TEnabla2 = TEnabla2 - (
        (2.0 * TE * inv_primal_edge_length ** 2)
        + (2.0 * TE * inv_vert_vert_length ** 2)
    )

    # build ODEs
    TE_t = where(
        boundary_edge,
        0.,
        kappa*TEnabla2,
    )

    # correct
    TE = TEinit + dt*TE_t
    
    return TE_t, TE
```

```{code-cell} ipython3
def test_diffusion_step():
    u = random_field((n_edges), E)
    v = random_field((n_edges), E)
    nx = random_field((n_edges), E)
    ny = random_field((n_edges), E)
    L = random_field((n_edges), E)
    dualL = random_field((n_edges), E)
    A = random_field((n_cells), C)
    dualA = random_field((n_vertices), V)
    edge_orientation_vertex = random_field((n_cells, 6), V, V2EDim)
    edge_orientation_cell = random_field((n_cells, 3), C, C2EDim)

    laplacian_ref = laplacian_numpy(
        c2e_table,
        u.asnumpy(),
        v.asnumpy(),
        nx.asnumpy(),
        ny.asnumpy(),
        L.asnumpy(),
        A.asnumpy(),
        edge_orientation.asnumpy(),
    )

    c2e_connectivity = gtx.NeighborTableOffsetProvider(c2e_table, C, E, 3)

    laplacian_gt4py = zero_field((n_edges), E)

    laplacian_fvm(
        u, v, nx, ny, L, A, edge_orientation, out = divergence_gt4py, offset_provider = {C2E.value: c2e_connectivity}
    )
    
    assert np.allclose(divergence_gt4py.asnumpy(), divergence_ref)
```

```{code-cell} ipython3
test_diffusion_step()
print("Test successful")
```

```{code-cell} ipython3

```
