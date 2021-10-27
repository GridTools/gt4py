import numpy as np

from functional.iterator.builtins import *
from functional.iterator.embedded import np_as_located_field
from functional.iterator.runtime import *

from .hdiff_reference import hdiff_reference


I = offset("I")
J = offset("J")

IDim = CartesianAxis("IDim")
JDim = CartesianAxis("JDim")


@fundef
def laplacian(inp):
    return -4.0 * inp() + (inp(I + 1) + inp(I - 1) + inp(J + 1) + inp(J - 1))


@fundef
def flux(d):
    @fundef
    def flux_impl(inp):
        lap = laplacian[...](inp)
        flux = lap() - lap(d + 1)
        return if_(flux * (inp(d + 1) - inp()) > 0.0, 0.0, flux)

    return flux_impl


@fundef
def hdiff_sten(inp, coeff):
    flx = flux(I)[...](inp)
    fly = flux(J)[...](inp)
    return inp() - (coeff() * (flx() - flx(I - 1) + fly() - fly(J - 1)))


@fendef(offset_provider={"I": IDim, "J": JDim})
def hdiff(inp, coeff, out, x, y):
    closure(
        domain(named_range(IDim, 0, x), named_range(JDim, 0, y)),
        hdiff_sten,
        [out],
        [inp, coeff],
    )


def test_hdiff(hdiff_reference):
    backend = None  # TODO tracing doesn't support the beautified syntax
    use_tmps = False
    validate = True

    inp, coeff, out = hdiff_reference
    shape = (out.shape[0], out.shape[1])

    inp_s = np_as_located_field(IDim, JDim, origin={IDim: 2, JDim: 2})(inp[:, :, 0])
    coeff_s = np_as_located_field(IDim, JDim)(coeff[:, :, 0])
    out_s = np_as_located_field(IDim, JDim)(np.zeros_like(coeff[:, :, 0]))

    hdiff(inp_s, coeff_s, out_s, shape[0], shape[1], backend=backend, use_tmps=use_tmps)

    if validate:
        assert np.allclose(out[:, :, 0], out_s)
