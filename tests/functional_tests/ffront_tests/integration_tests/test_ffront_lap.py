import numpy as np

from functional.common import Field
from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import FieldOffset
from functional.iterator.builtins import *
from functional.iterator.embedded import np_as_located_field
from functional.iterator.runtime import CartesianAxis, fundef


IDim = CartesianAxis("IDim")
JDim = CartesianAxis("JDim")

Ioff = FieldOffset("Ioff", source=IDim, target=(IDim,))
Joff = FieldOffset("Joff", source=JDim, target=(JDim,))


# @field_operator
# def lap(
#     in_field: Field[[IDim, JDim], "float"], a4: Field[[IDim, JDim], "float"]
# ) -> Field[[IDim, JDim], "float"]:
#     return (
#         a4 * in_field
#         + in_field(Ioff[1])
#         + in_field(Joff[1])
#         + in_field(Ioff[-1])
#         + in_field(Joff[-1])
#     )


@fundef
def lap(in_field, a4):
    return (
        deref(a4) * deref(in_field)
        + deref(shift(Ioff, 1)(in_field))
        + deref(shift(Joff, 1)(in_field))
        + deref(shift(Ioff, -1)(in_field))
        + deref(shift(Joff, -1)(in_field))
    )


@field_operator
def laplap(
    in_field: Field[[IDim, JDim], "float"], a4: Field[[IDim, JDim], "float"]
) -> Field[[IDim, JDim], "float"]:
    return lap(lap(in_field, a4), a4)


@program
def lap_program(
    in_field: Field[[IDim, JDim], "float"],
    a4: Field[[IDim, JDim], "float"],
    out_field: Field[[IDim, JDim], "float"],
):
    lap(in_field, a4, out=out_field[1:-1, 1:-1])


@program
def laplap_program(
    in_field: Field[[IDim, JDim], "float"],
    a4: Field[[IDim, JDim], "float"],
    out_field: Field[[IDim, JDim], "float"],
):
    laplap(in_field, a4, out=out_field[2:-2, 2:-2])


def lap_ref(inp):
    """Compute the laplacian using numpy"""
    return -4.0 * inp[1:-1, 1:-1] + inp[:-2, 1:-1] + inp[2:, 1:-1] + inp[1:-1, :-2] + inp[1:-1, 2:]


def test_ffront_lap():
    shape = (20, 20)
    as_ij = np_as_located_field(IDim, JDim)
    input = as_ij(np.fromfunction(lambda x, y: x**2 + y**2, shape))
    a4 = as_ij(np.ones(shape) * -4.0)  # TODO support scalar field

    result_lap = as_ij(np.zeros_like(input))
    lap_program(input, a4, result_lap, offset_provider={"Ioff": IDim, "Joff": JDim})
    assert np.allclose(np.asarray(result_lap)[1:-1, 1:-1], lap_ref(np.asarray(input)))

    result_laplap = as_ij(np.zeros_like(input))
    laplap_program(input, a4, result_laplap, offset_provider={"Ioff": IDim, "Joff": JDim})
    assert np.allclose(np.asarray(result_laplap)[2:-2, 2:-2], lap_ref(lap_ref(np.asarray(input))))


test_ffront_lap()
