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
from gt4py.next.iterator.builtins import *
from gt4py.next.iterator.runtime import set_at, fendef, fundef, offset
from gt4py.next.program_processors.runners import gtfn

from next_tests.integration_tests.cases import IDim, JDim
from next_tests.integration_tests.multi_feature_tests.iterator_tests.hdiff_reference import (
    hdiff_reference,
)
from next_tests.unit_tests.conftest import program_processor, run_processor


I = offset("I")
J = offset("J")


@fundef
def laplacian(inp):
    return -4.0 * deref(inp) + (
        deref(shift(I, 1)(inp))
        + deref(shift(I, -1)(inp))
        + deref(shift(J, 1)(inp))
        + deref(shift(J, -1)(inp))
    )


@fundef
def flux(d):
    def flux_impl(inp):
        lap = lift(laplacian)(inp)
        flux = deref(lap) - deref(shift(d, 1)(lap))
        return if_(flux * (deref(shift(d, 1)(inp)) - deref(inp)) > 0.0, 0.0, flux)

    return flux_impl


@fundef
def hdiff_sten(inp, coeff):
    flx = lift(flux(I))(inp)
    fly = lift(flux(J))(inp)
    return deref(inp) - (
        deref(coeff)
        * (deref(flx) - deref(shift(I, -1)(flx)) + deref(fly) - deref(shift(J, -1)(fly)))
    )


@fendef(offset_provider={"I": IDim, "J": JDim})
def hdiff(inp, coeff, out, x, y):
    domain = cartesian_domain(named_range(IDim, 0, x), named_range(JDim, 0, y))
    set_at(as_fieldop(hdiff_sten, domain)(inp, coeff), domain, out)


@pytest.mark.uses_origin
def test_hdiff(hdiff_reference, program_processor):
    program_processor, validate = program_processor

    inp, coeff, out = hdiff_reference
    shape = (out.shape[0], out.shape[1])

    inp_s = gtx.as_field([IDim, JDim], inp[:, :, 0], origin={IDim: 2, JDim: 2})
    coeff_s = gtx.as_field([IDim, JDim], coeff[:, :, 0])
    out_s = gtx.as_field([IDim, JDim], np.zeros_like(coeff[:, :, 0]))

    run_processor(hdiff, program_processor, inp_s, coeff_s, out_s, shape[0], shape[1])

    if validate:
        assert np.allclose(out[:, :, 0], out_s.asnumpy())
