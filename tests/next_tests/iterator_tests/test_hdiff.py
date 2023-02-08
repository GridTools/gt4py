# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pytest

from gt4py.next.common import Dimension
from gt4py.next.iterator.builtins import *
from gt4py.next.iterator.embedded import np_as_located_field
from gt4py.next.iterator.runtime import closure, fendef, fundef, offset
from gt4py.next.program_processors.runners.gtfn_cpu import run_gtfn, run_gtfn_imperative

from .conftest import run_processor
from .hdiff_reference import hdiff_reference


I = offset("I")
J = offset("J")

IDim = Dimension("IDim")
JDim = Dimension("JDim")


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
    closure(
        cartesian_domain(named_range(IDim, 0, x), named_range(JDim, 0, y)),
        hdiff_sten,
        out,
        [inp, coeff],
    )


def test_hdiff(hdiff_reference, program_processor, lift_mode):
    program_processor, validate = program_processor
    if program_processor == run_gtfn or program_processor == run_gtfn_imperative:
        pytest.xfail("origin not yet supported in gtfn")

    inp, coeff, out = hdiff_reference
    shape = (out.shape[0], out.shape[1])

    inp_s = np_as_located_field(IDim, JDim, origin={IDim: 2, JDim: 2})(inp[:, :, 0])
    coeff_s = np_as_located_field(IDim, JDim)(coeff[:, :, 0])
    out_s = np_as_located_field(IDim, JDim)(np.zeros_like(coeff[:, :, 0]))

    run_processor(
        hdiff, program_processor, inp_s, coeff_s, out_s, shape[0], shape[1], lift_mode=lift_mode
    )

    if validate:
        assert np.allclose(out[:, :, 0], out_s)
