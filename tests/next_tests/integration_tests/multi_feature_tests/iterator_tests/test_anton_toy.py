# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next.iterator.builtins import cartesian_domain, deref, lift, named_range, shift
from gt4py.next.iterator.runtime import closure, fendef, fundef, offset
from gt4py.next.program_processors.runners import gtfn

from next_tests.unit_tests.conftest import program_processor, run_processor


# cross-reference why new type inference does not support this
@fundef
def ldif(d):
    return lambda inp: deref(shift(d, -1)(inp)) - deref(inp)


@fundef
def rdif(d):
    return lambda inp: ldif(d)(shift(d, 1)(inp))


@fundef
def dif2(d):
    return lambda inp: ldif(d)(lift(rdif(d))(inp))


i = offset("i")
j = offset("j")


@fundef
def lap(inp):
    return dif2(i)(inp) + dif2(j)(inp)


@fundef
def lap_flat(inp):
    return -4.0 * deref(inp) + (
        deref(shift(i, 1)(inp))
        + deref(shift(i, -1)(inp))
        + deref(shift(j, 1)(inp))
        + deref(shift(j, -1)(inp))
    )


IDim = gtx.Dimension("IDim")
JDim = gtx.Dimension("JDim")
KDim = gtx.Dimension("KDim")


def naive_lap(inp):
    shape = [inp.shape[0] - 2, inp.shape[1] - 2, inp.shape[2]]
    out = np.zeros(shape)
    inp_data = inp.asnumpy()
    for i in range(1, shape[0] + 1):
        for j in range(1, shape[1] + 1):
            for k in range(0, shape[2]):
                out[i - 1, j - 1, k] = -4 * inp_data[i, j, k] + (
                    inp_data[i + 1, j, k]
                    + inp_data[i - 1, j, k]
                    + inp_data[i, j + 1, k]
                    + inp_data[i, j - 1, k]
                )
    return out


@pytest.mark.uses_origin
@pytest.mark.parametrize("stencil", [lap, lap_flat])
def test_anton_toy(stencil, program_processor):
    program_processor, validate = program_processor

    if stencil is lap:
        pytest.xfail(
            "Type inference does not support calling lambdas with offset arguments of changing type."
        )

    @fendef(offset_provider={"i": IDim, "j": JDim})
    def fencil(x, y, z, out, inp):
        closure(
            cartesian_domain(
                named_range(IDim, 0, x), named_range(JDim, 0, y), named_range(KDim, 0, z)
            ),
            stencil,
            out,
            [inp],
        )

    shape = [5, 7, 9]
    rng = np.random.default_rng()
    inp = gtx.as_field(
        [IDim, JDim, KDim],
        rng.normal(size=(shape[0] + 2, shape[1] + 2, shape[2])),
        origin={IDim: 1, JDim: 1, KDim: 0},
    )
    out = gtx.as_field([IDim, JDim, KDim], np.zeros(shape))
    ref = naive_lap(inp)

    run_processor(fencil, program_processor, shape[0], shape[1], shape[2], out, inp)

    if validate:
        assert np.allclose(out.asnumpy(), ref)
