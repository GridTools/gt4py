import numpy as np
import pytest

from functional.common import Field
from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import FieldOffset
from functional.iterator.builtins import *
from functional.iterator.embedded import np_as_located_field
from functional.iterator.runtime import CartesianAxis, closure, fendef, fundef


IDim = CartesianAxis("IDim")
JDim = CartesianAxis("JDim")

Ioff = FieldOffset("Ioff", source=IDim, target=(IDim,))
Joff = FieldOffset("Joff", source=JDim, target=(JDim,))


@field_operator
def lapf(in_field: Field[[IDim, JDim], "float"]) -> Field[[IDim, JDim], "float"]:
    return (
        -4.0 * in_field
        + in_field(Ioff[1])
        + in_field(Joff[1])
        + in_field(Ioff[-1])
        + in_field(Joff[-1])
    )


@fundef
def lapi(in_field):
    return (
        -4.0 * deref(in_field)
        + deref(shift(Ioff, 1)(in_field))
        + deref(shift(Joff, 1)(in_field))
        + deref(shift(Ioff, -1)(in_field))
        + deref(shift(Joff, -1)(in_field))
    )


@fundef
def lapilapi(in_field):
    return lapi(lift(lapi)(in_field))


@fundef
def lapilapf(in_field):
    return lapi(lift(lapf)(in_field))


@field_operator
def lapflapi(in_field: Field[[IDim, JDim], "float"]) -> Field[[IDim, JDim], "float"]:
    return lapf(lapi(in_field))


@field_operator
def lapflapf(in_field: Field[[IDim, JDim], "float"]) -> Field[[IDim, JDim], "float"]:
    return lapf(lapf(in_field))


@program
def lapf_program(
    in_field: Field[[IDim, JDim], "float"],
    out_field: Field[[IDim, JDim], "float"],
):
    lapf(in_field, out=out_field[1:-1, 1:-1])


@program
def lapi_program(
    in_field: Field[[IDim, JDim], "float"],
    out_field: Field[[IDim, JDim], "float"],
):
    lapi(in_field, out=out_field[1:-1, 1:-1])


@fendef
def lapi_fencil(in_field, out_field):
    closure(
        domain(named_range(IDim, 1, 19), named_range(JDim, 1, 19)),
        lapi,
        out_field,
        [in_field],
    )


@fendef(backend="roundtrip")  # embedded not possible, as fieldview doesn't have it
def lapf_fencil(in_field, out_field):
    closure(
        domain(named_range(IDim, 1, 19), named_range(JDim, 1, 19)),
        lapf,
        out_field,
        [in_field],
    )


@program
def lapilapi_program(
    in_field: Field[[IDim, JDim], "float"],
    out_field: Field[[IDim, JDim], "float"],
):
    lapilapi(in_field, out=out_field[2:-2, 2:-2])


@program
def lapilapf_program(
    in_field: Field[[IDim, JDim], "float"],
    out_field: Field[[IDim, JDim], "float"],
):
    lapilapf(in_field, out=out_field[2:-2, 2:-2])


@program
def lapflapi_program(
    in_field: Field[[IDim, JDim], "float"],
    out_field: Field[[IDim, JDim], "float"],
):
    lapflapi(in_field, out=out_field[2:-2, 2:-2])


@program
def lapflapf_program(
    in_field: Field[[IDim, JDim], "float"],
    out_field: Field[[IDim, JDim], "float"],
):
    lapflapf(in_field, out=out_field[2:-2, 2:-2])


@fendef
def lapilapi_fencil(in_field, out_field):
    closure(
        domain(named_range(IDim, 2, 18), named_range(JDim, 2, 18)),
        lapilapi,
        out_field,
        [in_field],
    )


@fendef(backend="roundtrip")  # embedded not possible, as fieldview doesn't have it
def lapilapf_fencil(in_field, out_field):
    closure(
        domain(named_range(IDim, 2, 18), named_range(JDim, 2, 18)),
        lapilapf,
        out_field,
        [in_field],
    )


@fendef(backend="roundtrip")  # embedded not possible, as fieldview doesn't have it
def lapflapi_fencil(in_field, out_field):
    closure(
        domain(named_range(IDim, 2, 18), named_range(JDim, 2, 18)),
        lapflapi,
        out_field,
        [in_field],
    )


@fendef(backend="roundtrip")  # embedded not possible, as fieldview doesn't have it
def lapflapf_fencil(in_field, out_field):
    closure(
        domain(named_range(IDim, 2, 18), named_range(JDim, 2, 18)),
        lapflapf,
        out_field,
        [in_field],
    )


def lap_ref(inp):
    """Compute the laplacian using numpy"""
    return -4.0 * inp[1:-1, 1:-1] + inp[:-2, 1:-1] + inp[2:, 1:-1] + inp[1:-1, :-2] + inp[1:-1, 2:]


# TODO cleanup
shape = (20, 20)
as_ij = np_as_located_field(IDim, JDim)
input = as_ij(np.fromfunction(lambda x, y: x**2 + y**2, shape))


@pytest.mark.parametrize("prog", [lapi_program, lapf_program, lapi_fencil, lapf_fencil])
def test_ffront_lap(prog):
    result_lap = as_ij(np.zeros_like(input))
    prog(input, result_lap, offset_provider={"Ioff": IDim, "Joff": JDim})
    assert np.allclose(np.asarray(result_lap)[1:-1, 1:-1], lap_ref(np.asarray(input)))


@pytest.mark.parametrize(
    "prog",
    [
        lapilapi_program,
        lapflapf_program,
        lapilapf_program,
        lapflapi_program,
        lapilapi_fencil,
        lapilapf_fencil,
        lapflapi_fencil,
        lapflapf_fencil,
    ],
)
def test_ffront_laplap(prog):
    result_laplap = as_ij(np.zeros_like(input))
    prog(input, result_laplap, offset_provider={"Ioff": IDim, "Joff": JDim})
    assert np.allclose(np.asarray(result_laplap)[2:-2, 2:-2], lap_ref(lap_ref(np.asarray(input))))
