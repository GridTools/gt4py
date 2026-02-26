import numpy as np
import xarray as xr
import gt4py.next as gtx
from gt4py.next.experimental import concat_where
from gt4py.next import neighbor_sum

# Define Dimensions
IDim = gtx.Dimension("IDim")
JDim = gtx.Dimension("JDim")
Kolor = gtx.Dimension("Kolor")

MAX_I = gtx.int32(13)
MAX_J = gtx.int32(11)

@gtx.field_operator
def compute_zavgS_cartesian_0(
    pp: gtx.Field[[IDim, JDim], float], S_M: gtx.Field[[IDim, JDim, Kolor], float],
#    domain_max_j: gtx.int32
) -> gtx.Field[[IDim, JDim, Kolor], float]:
    zavg = 0.5 * concat_where(
        JDim == 11,
        pp,
        pp + pp(JDim + 1))
    # zavg = 0.5 * (pp + pp(JDim + 1))
    return S_M * zavg


@gtx.field_operator
def compute_zavgS_cartesian_1(
    pp: gtx.Field[[IDim, JDim], float], S_M: gtx.Field[[IDim, JDim, Kolor], float]#, domain_max_i: gtx.int32
) -> gtx.Field[[IDim, JDim, Kolor], float]:
    zavg = 0.5 * concat_where(
        IDim == 13,
        pp,
        pp + pp(IDim + 1))
    # zavg = 0.5 * (pp + pp(IDim + 1))
    return S_M * zavg


@gtx.field_operator
def compute_zavgS_cartesian_2(
    pp: gtx.Field[[IDim, JDim], float], S_M: gtx.Field[[IDim, JDim, Kolor], float]#, domain_max_i: gtx.int32, domain_max_j: gtx.int32
) -> gtx.Field[[IDim, JDim, Kolor], float]:
    zavg = 0.5 * concat_where(
        IDim == 13, concat_where(
            JDim == 11,
            pp-pp,
            pp + pp(JDim + 1)),
        concat_where(JDim == 11,
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
    # domain_max_i: gtx.int32,
    # domain_max_j: gtx.int32,
) -> gtx.Field[[IDim, JDim, Kolor], float]:

    return on_edges(
        compute_zavgS_cartesian_0(pp, S_M),
        compute_zavgS_cartesian_1(pp, S_M),
        compute_zavgS_cartesian_2(pp, S_M),
    )

@gtx.program
def zavg(
    pp: gtx.Field[[IDim, JDim], float],
    S_M: gtx.Field[[IDim, JDim, Kolor], float],
    out: gtx.Field[[IDim, JDim, Kolor], float],
    # domain_max_i: gtx.int32,
    # domain_max_j: gtx.int32,

):
    compute_zavgS_cartesian(pp, S_M, out=out, domain={IDim: (0, 13), JDim: (0, 11), Kolor: (0, 3)})

@gtx.field_operator
def compute_pnabla_cartesian(
    pp: gtx.Field[[IDim, JDim], float],
    S_M: gtx.Field[[IDim, JDim, Kolor], float],
    sign: gtx.Field[[IDim, JDim, Kolor], float],
    vol: gtx.Field[[IDim, JDim], float],
    # max_i: gtx.int32,
    # max_j: gtx.int32,
) -> gtx.Field[[IDim, JDim], float]:
    zavgS = compute_zavgS_cartesian(pp, S_M)

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
    # domain_max_i: gtx.int32,
    # domain_max_j: gtx.int32,
):
    compute_pnabla_cartesian(pp, S_M, sign, vol,
                  out=out, domain={IDim: (0, 13), JDim: (0, 11), Kolor: (0, 3)},
)

import functools
import logging
from collections.abc import Callable
from typing import Any

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
from gt4py.next import backend as gtx_backend
from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations
log = logging.getLogger(__name__)

# from icon4py.model.common import model_backends


def dict_values_to_list(d: dict[str, Any]) -> dict[str, list]:
    return {k: [v] for k, v in d.items()}

def customize_backend(
    program: gtx_typing.Program | gtx.typing.FieldOperator | None,
    backend: gtx_typing.Backend
    # | model_backends.DeviceType
    # | model_backends.BackendDescriptor
    | None,
) -> gtx_typing.Backend | None:
    program_name = program.__name__ if program is not None else ""
    if backend is None or isinstance(backend, gtx_backend.Backend):
        backend_name = backend.name if backend is not None else "embedded"
        log.info(f"Using non-custom backend '{backend_name}' for '{program_name}'.")
        return backend  # type: ignore[return-value]

    backend_descriptor = (
        {"device": backend} if isinstance(backend, model_backends.DeviceType) else backend
    )
    backend_descriptor = get_options(program_name, **backend_descriptor)
    backend_descriptor["device"] = backend_descriptor.get(
        "device", model_backends.DeviceType.CPU
    )  # set default device
    backend_factory = backend_descriptor.pop(
        "backend_factory", model_backends.make_custom_dace_backend
    )
    custom_backend = backend_factory(**backend_descriptor)
    log.info(
        f"Using custom backend '{custom_backend.name}' for '{program_name}' with options: {backend_descriptor}."
    )
    return custom_backend



def setup_program(
    program: gtx_typing.Program,
    backend: gtx_typing.Backend
    # | model_backends.DeviceType
    # | model_backends.BackendDescriptor
    | None,
    constant_args: dict[str, gtx.Field | gtx_typing.Scalar] | None = None,
    variants: dict[str, list[gtx_typing.Scalar]] | None = None,
    horizontal_sizes: dict[str, gtx.int32] | None = None,
    vertical_sizes: dict[str, gtx.int32] | None = None,
    offset_provider: gtx_typing.OffsetProvider | None = None,
) -> Callable[..., None]:
    """
    This function processes arguments to the GT4Py program. It
    - binds arguments that don't change during model run ('constant_args', 'horizontal_sizes', "vertical_sizes');
    - inlines scalar arguments into the GT4Py program at compile-time (via GT4Py's 'compile').
    Args:
        - backend: GT4Py backend,
        - program: GT4Py program,
        - constant_args: constant fields and scalars,
        - variants: list of all scalars potential values from which one is selected at run time,
        - horizontal_sizes: horizontal domain bounds,
        - vertical_sizes: vertical domain bounds,
        - offset_provider: GT4Py offset_provider,
    """
    constant_args = {} if constant_args is None else constant_args
    variants = {} if variants is None else variants
    horizontal_sizes = {} if horizontal_sizes is None else horizontal_sizes
    vertical_sizes = {} if vertical_sizes is None else vertical_sizes
    offset_provider = {} if offset_provider is None else offset_provider

    backend = customize_backend(program, backend)

    bound_static_args = {k: v for k, v in constant_args.items() if gtx.is_scalar_type(v)}
    static_args_program = program.with_backend(backend)
    if backend is not None:
        static_args_program = static_args_program.with_compilation_options(enable_jit=False)
        static_args_program.compile(
            **dict_values_to_list(horizontal_sizes),
            **dict_values_to_list(vertical_sizes),
            **variants,
            **dict_values_to_list(bound_static_args),
            offset_provider=offset_provider,
        )

    return functools.partial(
        static_args_program,
        **constant_args,
        **horizontal_sizes,
        **vertical_sizes,
        offset_provider=offset_provider,
    )


# Load the dataset
ds = xr.open_dataset("/home/raphael/Documents/Studium/Msc_thesis/grid-generator/parallelogram_grid.nc")
# raw_edges = ds['edge_data'].values # The big 1D array

# Define your grid sizes (from your description)
nx = np.int32(ds.attrs['domain_length'] / ds.attrs['mean_edge_length'])
ny = np.int32((ds.sizes['cell'])/(2*nx))
max_j = 11 # nx + 1
max_i = 13 # ny + 1
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
zavg_dace = zavg.with_backend(run_dace_cpu)
zavg_dace = setup_program(zavg_dace, backend=run_dace_cpu)
# Pseudo-code for execution
zavg_dace(
    pp=pp, 
    S_M=S_M,
    out=out,
    offset_provider={},
    # domain_max_i=gtx.int32(max_i), 
    # domain_max_j=gtx.int32(max_j),
)

print("Output data East:\n", out[:,:,0].asnumpy())
print("Output data NE:\n", out[:,:,1].asnumpy())
print("Output data SE:\n", out[:,:,2].asnumpy())
diff_i = pp.asnumpy()[1,0] - pp.asnumpy()[0,0]
diff_j = pp.asnumpy()[0,1] - pp.asnumpy()[0,0]
print("diff_i: ", diff_i)
print("diff_j: ", diff_j)


# @gtx.field_operator
# def neighbor_sum_cart(
#     edges_0: gtx.Field[[IDim, JDim], float],
#     edges_1: gtx.Field[[IDim, JDim], float],
#     edges_2: gtx.Field[[IDim, JDim], float],
# ) -> gtx.Field[[IDim, JDim], float]:
#     return (edges_0 
#         + edges_0(JDim - 1)
#         + edges_1
#         + edges_1(IDim - 1)
#         + edges_2(IDim - 1)
#         + edges_2(JDim - 1)
#     )

# @gtx.program
# def neighbor_sum_program(
#     edges_0: gtx.Field[[IDim, JDim], float],
#     edges_1: gtx.Field[[IDim, JDim], float],
#     edges_2: gtx.Field[[IDim, JDim], float],
#     out: gtx.Field[[IDim, JDim], float],
#     domain_max_i: int,
#     domain_max_j: int,
# ):
#     neighbor_sum_cart(edges_0, edges_1, edges_2,
#                   out=out, domain={IDim: (0, domain_max_i), JDim: (0, domain_max_j)},
# )
    

# @gtx.field_operator
# def reduce_Kolors(
#     edges_0: gtx.Field[[IDim, JDim, Kolor], float],
# ) -> gtx.Field[[IDim, JDim], float]:
#     return edges_0(Kolor == 0) + edges_0(Kolor == 1) + edges_0(Kolor == 2)



# Test the neighbor_sum operator on the output of compute_zavgS_cartesian
# neighbor_sum_out = gtx.as_field([IDim, JDim], np.zeros((max_i, max_j)))

# Slice the output field 'out' into its 3 Kolor components in Python
# We already computed 'out' in the previous step
# edges_field_np = out.asnumpy()

# padded_shape = (max_i + 1, max_j + 1)
# edges_0_np = np.zeros(padded_shape)
# edges_1_np = np.zeros(padded_shape)
# edges_2_np = np.zeros(padded_shape)
# edges_0_np[1:, 1:] = edges_field_np[:,:,0]
# edges_1_np[1:, 1:] = edges_field_np[:,:,1]
# edges_2_np[1:, 1:] = edges_field_np[:,:,2]

# # Add padding and shift the origin
# edges_0_padded = gtx.as_field([IDim, JDim], edges_0_np, origin={IDim: 1, JDim: 1})
# edges_1_padded = gtx.as_field([IDim, JDim], edges_1_np, origin={IDim: 1, JDim: 1})
# edges_2_padded = gtx.as_field([IDim, JDim], edges_2_np, origin={IDim: 1, JDim: 1})


# neighbor_sum_program.with_backend(run_dace_cpu)(
#     edges_0=edges_0_padded,
#     edges_1=edges_1_padded,
#     edges_2=edges_2_padded,
#     out=neighbor_sum_out,
#     domain_max_i=gtx.int64(max_i), 
#     domain_max_j=gtx.int64(max_j),
#     offset_provider={}
# )

# print("Neighbor sum output:\n", neighbor_sum_out.asnumpy())

vol = np.ones((max_i, max_j))
sign = np.ones((max_i, max_j, 3))
vol_field = gtx.as_field([IDim, JDim], vol)
sign_field = gtx.as_field([IDim, JDim, Kolor], sign)

pnabla_out = gtx.as_field([IDim, JDim], np.zeros((max_i, max_j)))

# pnabla_cartesian_dace = compute_pnabla_cartesian.with_backend(run_dace_cpu)
pnabla_cartesian_dace = setup_program(pnabla_cartesian, backend=run_dace_cpu, )

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


# neighbor_sum_onefield_program.with_backend(run_dace_cpu)(
#     edges=out,
#     out=neighbor_sum_out,
#     domain_max_i=gtx.int64(max_i), 
#     domain_max_j=gtx.int64(max_j),
#     offset_provider={},
# )

# from gt4py.next.iterator.transforms import pass_manager

# ir = neighbor_sum_program.gtir
# print(ir)
# # print(pass_manager.apply_fieldview_transforms(ir, offset_provider={}))
# print(pass_manager.apply_common_transforms(ir, offset_provider={}))

# TODOs:
# write Hannes, gtfn /ohne concat_where
# - look at unit test unroll_reduce what skip value does and 1d or 2d 
# - see where max/min get introduced and look at test
# -  test direct Color dimension addressing in icon4py (e.g. nabla4)