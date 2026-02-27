import numpy as np
import xarray as xr
import gt4py.next as gtx
from gt4py.next.experimental import concat_where
# from gt4py.next import where as concat_where
from gt4py.next import neighbor_sum
from program_setup_utils import setup_program

# Define Dimensions
IDim = gtx.Dimension("IDim")
JDim = gtx.Dimension("JDim")
Kolor = gtx.Dimension("Kolor")


@gtx.field_operator
def compute_zavgS_cartesian_0(
    pp: gtx.Field[[IDim, JDim], float],
    S_M: gtx.Field[[IDim, JDim, Kolor], float],
    domain_max_j: gtx.int32,
) -> gtx.Field[[IDim, JDim, Kolor], float]:
    zavg = 0.5 * concat_where(
        JDim == domain_max_j - 1,
        pp,
        pp + pp(JDim + 1))
    # zavg = 0.5 * (pp + pp(JDim + 1))
    return S_M * zavg


@gtx.field_operator
def compute_zavgS_cartesian_1(
    pp: gtx.Field[[IDim, JDim], float],
    S_M: gtx.Field[[IDim, JDim, Kolor], float],
    domain_max_i: gtx.int32,
) -> gtx.Field[[IDim, JDim, Kolor], float]:
    zavg = 0.5 * concat_where(
        IDim == domain_max_i - 1,
        pp,
        pp + pp(IDim + 1))
    # zavg = 0.5 * (pp + pp(IDim + 1))
    return S_M * zavg


@gtx.field_operator
def compute_zavgS_cartesian_2(
    pp: gtx.Field[[IDim, JDim], float],
    S_M: gtx.Field[[IDim, JDim, Kolor], float],
    domain_max_i: gtx.int32,
    domain_max_j: gtx.int32,
) -> gtx.Field[[IDim, JDim, Kolor], float]:
    zavg = 0.5 * concat_where(
        IDim == domain_max_i - 1, concat_where(
            JDim == domain_max_j - 1,
            pp-pp,
            pp + pp(JDim + 1)),
        concat_where(JDim == domain_max_j - 1,
            pp(IDim + 1), pp(IDim + 1) + pp(JDim + 1))
    )
    # zavg = 0.5 * (pp(IDim + 1) + pp(JDim + 1))
    return S_M * zavg

@gtx.field_operator
def on_edges(
    f0: gtx.Field[[IDim, JDim, Kolor], float],
    f1: gtx.Field[[IDim, JDim, Kolor], float],
    f2: gtx.Field[[IDim, JDim, Kolor], float],
) -> gtx.Field[[IDim, JDim, Kolor], float]:
    return concat_where(
        Kolor == 0, 
        f0,
        concat_where(Kolor == 1, f1, f2),
    )

@gtx.field_operator
def compute_zavgS_cartesian(
    pp: gtx.Field[[IDim, JDim], float],
    S_M: gtx.Field[[IDim, JDim, Kolor], float],
    domain_max_i: gtx.int32,
    domain_max_j: gtx.int32,
) -> gtx.Field[[IDim, JDim, Kolor], float]:

    return on_edges(
        compute_zavgS_cartesian_0(pp, S_M, domain_max_j),
        compute_zavgS_cartesian_1(pp, S_M, domain_max_i),
        compute_zavgS_cartesian_2(pp, S_M, domain_max_i, domain_max_j),
    )

@gtx.program
def zavg(
    pp: gtx.Field[[IDim, JDim], float],
    S_M: gtx.Field[[IDim, JDim, Kolor], float],
    out: gtx.Field[[IDim, JDim, Kolor], float],
    domain_max_i: gtx.int32,
    domain_max_j: gtx.int32,
    domain_max_kolor: gtx.int32,

):
    compute_zavgS_cartesian(
        pp,
        S_M,
        domain_max_i,
        domain_max_j,
        out=out,
        domain={IDim: (0, domain_max_i), JDim: (0, domain_max_j), Kolor: (0, domain_max_kolor)},
    )

@gtx.field_operator
def compute_pnabla_cartesian(
    pp: gtx.Field[[IDim, JDim], float],
    S_M: gtx.Field[[IDim, JDim, Kolor], float],
    sign: gtx.Field[[IDim, JDim, Kolor], float],
    vol: gtx.Field[[IDim, JDim], float],
    domain_max_i: gtx.int32,
    domain_max_j: gtx.int32,
) -> gtx.Field[[IDim, JDim], float]:
    zavgS = compute_zavgS_cartesian(pp, S_M, domain_max_i, domain_max_j)

    pnabla_M = concat_where(
    Kolor == 0,
    concat_where(JDim == 0, zavgS, zavgS + zavgS(JDim - 1)),
    concat_where(
        Kolor == 1,
        concat_where(IDim == 0, zavgS, zavgS + zavgS(IDim - 1)),
        concat_where(
            IDim == 0,
            concat_where(JDim == 0, zavgS - zavgS, zavgS(JDim - 1)),
            concat_where(JDim == 0, zavgS(IDim - 1), zavgS(IDim - 1) + zavgS(JDim - 1)),
        ),
    ),
)
    pnabla_M = pnabla_M * sign
    return neighbor_sum(pnabla_M, axis=Kolor) / vol

@gtx.program
def pnabla_cartesian(
    pp: gtx.Field[[IDim, JDim], float],
    S_M: gtx.Field[[IDim, JDim, Kolor], float],
    sign: gtx.Field[[IDim, JDim, Kolor], float],
    vol: gtx.Field[[IDim, JDim], float],
    out: gtx.Field[[IDim, JDim], float],
    domain_max_i: gtx.int32,
    domain_max_j: gtx.int32,
    domain_max_kolor: gtx.int32,
):
    compute_pnabla_cartesian(
        pp,
        S_M,
        sign,
        vol,
        domain_max_i,
        domain_max_j,
        out=out,
        domain={IDim: (0, domain_max_i), JDim: (0, domain_max_j), Kolor: (0, domain_max_kolor)},
    )

# Load the dataset
ds = xr.open_dataset("/home/raphael/Documents/Studium/Msc_thesis/grid-generator/parallelogram_grid.nc")
# raw_edges = ds['edge_data'].values # The big 1D array

# Define your grid sizes (from your description)
nx = np.int32(ds.attrs['domain_length'] / ds.attrs['mean_edge_length'])
ny = np.int32((ds.sizes['cell'])/(2*nx))
max_j = np.int32(nx + 1)
max_i = np.int32(ny + 1)
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

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up 4 levels to reach '.../gt4py/tests' 
# (current: ffront_tests -> multi_feature_tests -> integration_tests -> next_tests -> tests)
tests_root = os.path.abspath(os.path.join(current_dir, "../../../../"))
if tests_root not in sys.path:
    sys.path.append(tests_root)

edges_east_2d = edges_east.reshape((ny+1, nx))  # Shape: [nx, ny+1]
edges_ne_2d   = edges_ne.reshape((ny, nx+1))    # Shape: [nx+1, ny]
edges_se_2d   = edges_se.reshape((ny, nx))        # Shape: [nx, ny]

print("Edges East:", edges_east, "\n2D version:\n", edges_east_2d)

pp_2d = raw_vertices.reshape((ny + 1, nx + 1))    # Shape: [nx+1, ny+1]

# To stack them in one field

S_M_field = np.zeros((max_i, max_j, 3)) # [IDim, JDim, Kolor]

# Fill Kolor 0 (East) - fits in [0:nx, 0:ny+1]
S_M_field[0:ny+1, 0:nx, 0] = edges_east_2d

# Fill Kolor 1 (NE) - fits in [0:nx+1, 0:ny]
S_M_field[0:ny, 0:nx+1, 1] = edges_ne_2d

# Fill Kolor 2 (SE) - fits in [0:nx, 0:ny]
S_M_field[0:ny, 0:nx, 2]   = edges_se_2d

# Prepare Vertices (pp)
pp_field = np.zeros((max_i, max_j))
pp_field[:, :] = pp_2d

pp = gtx.as_field([IDim, JDim], pp_2d)
S_M = gtx.as_field([IDim, JDim, Kolor], S_M_field)
out = gtx.as_field([IDim, JDim, Kolor], np.zeros_like(S_M_field))
print("pp: ", pp.asnumpy())
print("S_M: ", S_M[:,:,2].asnumpy())
print("Output before computation: ", out)

from gt4py.next.program_processors.runners.dace import run_dace_cpu
from gt4py.next.program_processors.runners.gtfn import (
    run_gtfn_cached as gtfn_cpu
    )
# zavg_dace = setup_program(
#     zavg,
#     backend=gtfn_cpu,
#     horizontal_sizes={
#         "domain_max_i": gtx.int32(max_i),
#         "domain_max_j": gtx.int32(max_j),
#         "domain_max_kolor": gtx.int32(3),
#     },
# )
# Pseudo-code for execution
# zavg_dace(
#     pp=pp, 
#     S_M=S_M,
#     out=out,
#     offset_provider={},
#     # domain_max_i=gtx.int32(max_i), 
#     # domain_max_j=gtx.int32(max_j),
# )

print("Output data East:\n", out[:,:,0].asnumpy())
print("Output data NE:\n", out[:,:,1].asnumpy())
print("Output data SE:\n", out[:,:,2].asnumpy())
diff_i = pp.asnumpy()[1,0] - pp.asnumpy()[0,0]
diff_j = pp.asnumpy()[0,1] - pp.asnumpy()[0,0]
print("diff_i: ", diff_i)
print("diff_j: ", diff_j)


vol = np.ones((max_i, max_j))
sign = np.ones((max_i, max_j, 3))
vol_field = gtx.as_field([IDim, JDim], vol)
sign_field = gtx.as_field([IDim, JDim, Kolor], sign)

pnabla_out = gtx.as_field([IDim, JDim], np.zeros((max_i, max_j)))

# pnabla_cartesian_dace = compute_pnabla_cartesian.with_backend(run_dace_cpu)
pnabla_cartesian_dace = setup_program(
    pnabla_cartesian,
    backend=gtfn_cpu,
    horizontal_sizes={
        "domain_max_i": gtx.int32(max_i),
        "domain_max_j": gtx.int32(max_j),
        "domain_max_kolor": gtx.int32(3),
    },
)

pnabla_cartesian_dace(
    pp=pp,
    S_M=S_M,
    sign=sign_field,
    vol=vol_field,
    out=pnabla_out,
    # domain_max_i=gtx.int32(max_i), 
    # domain_max_j=gtx.int32(max_j),
    offset_provider={},
)

print("Output of pnabla_cartesian:\n", pnabla_out.asnumpy()[:, :])

# from gt4py.next.iterator.transforms import pass_manager

# ir = neighbor_sum_program.gtir
# print(ir)
# # print(pass_manager.apply_fieldview_transforms(ir, offset_provider={}))
# print(pass_manager.apply_common_transforms(ir, offset_provider={}))

# TODO:
# - write Hannes, gtfn /ohne concat_where
# - look at unit test unroll_reduce what skip value does and 1d or 2d 
# - see where max/min get introduced and look at test
# - test direct Color dimension addressing in icon4py (see e.g. nabla4)

