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
def divergence_numpy(
    c2e: np.array,
    u: np.array,
    v: np.array,
    nx: np.array,
    ny: np.array,
    L: np.array,
    A: np.array,
    edge_orientation: np.array,
) -> np.array:
    uv_div = np.sum((u[c2e]*nx[c2e] + v[c2e]*ny[c2e]) * L[c2e] * edge_orientation, axis=1) / A
    return uv_div
```

```{code-cell} ipython3
@gtx.field_operator(backend=roundtrip.executor)
def divergence(
    u: gtx.Field[[E], float],
    v: gtx.Field[[E], float],
    nx: gtx.Field[[E], float],
    ny: gtx.Field[[E], float],
    L: gtx.Field[[E], float],
    A: gtx.Field[[C], float],
    edge_orientation: gtx.Field[[C, C2EDim], float],
) -> gtx.Field[[C], float]:
    uv_div = neighbor_sum((u(C2E)*nx(C2E) + v(C2E)*ny(C2E)) * L(C2E) * edge_orientation, axis=C2EDim) / A
    return uv_div
```

```{code-cell} ipython3
def test_divergence():
    u = random_field((n_edges), E)
    v = random_field((n_edges), E)
    nx = random_field((n_edges), E)
    ny = random_field((n_edges), E)
    L = random_field((n_edges), E)
    A = random_field((n_cells), C)
    edge_orientation = random_field((n_cells, 3), C, C2EDim)

    divergence_ref = divergence_numpy(
        c2e_table,
        np.asarray(u),
        np.asarray(v),
        np.asarray(nx),
        np.asarray(ny),
        np.asarray(L),
        np.asarray(A),
        np.asarray(edge_orientation),
    )

    c2e_connectivity = gtx.NeighborTableOffsetProvider(c2e_table, C, E, 3)

    divergence_gt4py = zero_field((n_cells), C)

    divergence(
        u, v, nx, ny, L, A, edge_orientation, out = divergence_gt4py, offset_provider = {C2E.value: c2e_connectivity}
    )
    
    assert np.allclose(divergence_gt4py, divergence_ref)
```

```{code-cell} ipython3
test_divergence()
print("Test successful")
```
