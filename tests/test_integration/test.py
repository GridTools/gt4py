from gt4py import gtscript
import numpy as np
import gt4py.definitions as gt_definitions

Field3D = gtscript.Field[np.float_]

# @gtscript.stencil(rebuild=True, backend="numpy")
# def runtime_if(field_a: Field3D, field_b: Field3D):
#     with computation(BACKWARD), interval(...):
#         if field_a > 0.0:
#             field_b = -1
#             field_a = -field_a
#         else:
#             field_b = 1
#             field_a = field_a


@gtscript.stencil(rebuild=True, backend="numpy")
def tridiagonal_solver(inf: Field3D, diag: Field3D, sup: Field3D, rhs: Field3D, out: Field3D):
    with computation(FORWARD):
        with interval(0, 1), region(gt_definitions.selection[:3, :]):
            sup = sup / diag
            rhs = rhs / diag
        with interval(1, None):
            sup = sup / (diag - sup[0, 0, -1] * inf)
            rhs = (rhs - inf * rhs[0, 0, -1]) / (diag - sup[0, 0, -1] * inf)
    with computation(BACKWARD):
        with interval(0, -1):
            out = rhs - sup * out[0, 0, 1]
        with interval(-1, None):
            out = rhs
