import numpy as np
import xarray as xr

# Load the dataset
ds = xr.open_dataset("/home/raphael/Documents/Studium/Msc_thesis/grid-generator/parallelogram_grid.nc")
# raw_edges = ds['edge_data'].values # The big 1D array

# Define your grid sizes (from your description)
nx = np.int32(ds.attrs['domain_length'] / ds.attrs['mean_edge_length'])
ny = np.int32((ds.sizes['cell'])/(2*nx))

# raw_edges = np.linspace(0, nx*(ny+1) + (nx+1)*ny + nx*ny - 1, nx*(ny+1) + (nx+1)*ny + nx*ny, dtype=np.float64)
raw_edges = np.ones((nx*(ny+1) + (nx+1)*ny + nx*ny,), dtype=np.float64)
raw_vertices = np.linspace(0, (nx+1)*(ny+1) - 1, (nx+1)*(ny+1), dtype=np.float64)
# Calculate start/end indices for each block
# 1. East Edges: nx * (ny + 1)
count_east = nx * (ny + 1)
edges_east = raw_edges[0 : count_east]

# 2. NE Edges: (nx + 1) * ny
count_ne = (nx + 1) * ny
edges_ne = raw_edges[count_east : count_east + count_ne]

# 3. SE Edges: nx * ny
count_se = nx * ny
edges_se = raw_edges[count_east + count_ne : ]

import gt4py.next as gtx
from gt4py.next import where, broadcast
from gt4py.next.experimental import concat_where
# Define Dimensions
# 'Edge' is the index along the specific edge type (e.g., 0 to count_east-1)
IDim = gtx.Dimension("IDim")
JDim = gtx.Dimension("JDim")
Color = gtx.Dimension("Kolor")

# mask_0 = broadcast( (1, 0, 0), (IDim, JDim, Color) )
    
# mask_1 = broadcast( (0, 1, 0), (IDim, JDim, Color) )

# @gtx.field_operator
# def get_color_indices() -> gtx.Field[[Color], int]:
#     # This creates a Local Field implicitly.
#     # Gt4Py understands tuples as values along the last dimension.
#     return (0, 1, 2)

@gtx.field_operator
def compute_zavgS_cartesian_0(
    pp: gtx.Field[[IDim, JDim], float], S_M: gtx.Field[[IDim, JDim, Color], float]
) -> gtx.Field[[IDim, JDim, Color], float]:
    zavg = 0.5 * (pp + pp(JDim + 1))
    return S_M * zavg


@gtx.field_operator
def compute_zavgS_cartesian_1(
    pp: gtx.Field[[IDim, JDim], float], S_M: gtx.Field[[IDim, JDim, Color], float]
) -> gtx.Field[[IDim, JDim, Color], float]:
    zavg = 0.5 * (pp + pp(IDim + 1))
    return S_M * zavg


@gtx.field_operator
def compute_zavgS_cartesian_2(
    pp: gtx.Field[[IDim, JDim], float], S_M: gtx.Field[[IDim, JDim, Color], float]
) -> gtx.Field[[IDim, JDim, Color], float]:
    zavg = 0.5 * (pp(IDim + 1) + pp(JDim + 1))
    return S_M * zavg

@gtx.field_operator
def on_edges(
    f0: gtx.Field[[IDim, JDim, Color], float],
    f1: gtx.Field[[IDim, JDim, Color], float],
    f2: gtx.Field[[IDim, JDim, Color], float],
    idx: gtx.Field[[Color], np.int32]
) -> gtx.Field[[IDim, JDim, Color], float]:
    return where(
        idx == 0,
        f0,
        where(idx == 1, f1, f2),
    )

@gtx.field_operator
def compute_zavgS_cartesian(
    pp: gtx.Field[[IDim, JDim], float],
    S_M: gtx.Field[[IDim, JDim, Color], float],
    idx: gtx.Field[[Color], np.int32],
) -> gtx.Field[[IDim, JDim, Color], float]:

    return on_edges(
        compute_zavgS_cartesian_0(pp, S_M),
        compute_zavgS_cartesian_1(pp, S_M),
        compute_zavgS_cartesian_2(pp, S_M),
        idx
    )

@gtx.program
def zavg(
    pp: gtx.Field[[IDim, JDim], float],
    S_M: gtx.Field[[IDim, JDim, Color], float],
    idx: gtx.Field[[Color], np.int32],
    out: gtx.Field[[IDim, JDim, Color], float],
):
    compute_zavgS_cartesian(pp, S_M, idx, out=out) # , domain={IDim: (0, 1), JDim: (0, 1), Color: (0, 3)}


edges_east_2d = edges_east.reshape((nx, ny + 1))  # Shape: [nx, ny+1]
edges_ne_2d   = edges_ne.reshape((nx + 1, ny))    # Shape: [nx+1, ny]
edges_se_2d   = edges_se.reshape((nx, ny))        # Shape: [nx, ny]

print("Edges East:", edges_east, "\n2D version:\n", edges_east_2d)

# Also reshape vertices (pp) because you shift on them!
pp_2d = raw_vertices.reshape((nx + 1, ny + 1))    # Shape: [nx+1, ny+1]

# 2. Pad them to a common size for Gt4Py
# To stack them in one field, they must all be the same shape (Max I, Max J).
# We pad with 0.0 or NaN.
max_i = nx + 1
max_j = ny + 1

S_M_field = np.zeros((max_i, max_j, 3)) # [IDim, JDim, Color]

# Fill Color 0 (East) - fits in [0:nx, 0:ny+1]
S_M_field[0:nx, 0:ny+1, 0] = edges_east_2d

# Fill Color 1 (NE) - fits in [0:nx+1, 0:ny]
S_M_field[0:nx+1, 0:ny, 1] = edges_ne_2d

# Fill Color 2 (SE) - fits in [0:nx, 0:ny]
S_M_field[0:nx, 0:ny, 2]   = edges_se_2d

# Prepare Vertices (pp)
pp_field = np.zeros((max_i, max_j))
pp_field[:, :] = pp_2d

pp = gtx.as_field([IDim, JDim], pp_2d)
S_M = gtx.as_field([IDim, JDim, Color], S_M_field)
out = gtx.as_field([IDim, JDim, Color], np.zeros_like(S_M_field))
print("pp: ", pp.asnumpy())
print("S_M: ", S_M[:,:,2].asnumpy())
print("Output before computation: ", out)

from gt4py.next.program_processors.runners.dace import run_dace_cpu
zavg_dace = zavg.with_backend(run_dace_cpu)
# Pseudo-code for execution
output = zavg_dace(
    pp=pp, 
    S_M=S_M,
    idx=gtx.as_field([Color], np.array([0, 1, 2], dtype=np.int32)),
    out=out,
    offset_provider={},
)

print("Output data East:\n", out[:,:,0].asnumpy())
print("Output data NE:\n", out[:,:,1].asnumpy())
print("Output data SE:\n", out[:,:,2].asnumpy())
diff_i = pp.asnumpy()[1,0] - pp.asnumpy()[0,0]
diff_j = pp.asnumpy()[0,1] - pp.asnumpy()[0,0]
print("diff_i: ", diff_i)
print("diff_j: ", diff_j)