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

from gt4py.next.iterator.builtins import *
from gt4py.next.iterator.embedded import np_as_located_field
from gt4py.next.iterator.runtime import CartesianAxis, closure, fendef, fundef

from .conftest import run_processor


IDim = CartesianAxis("IDim")
JDim = CartesianAxis("JDim")
KDim = CartesianAxis("KDim")


@fundef
def test_conditional(inp):
    return if_(deref(inp), make_tuple(1, 2), make_tuple(3, 4))


def test_conditional_w_tuple(program_processor_no_gtfn_exec):
    program_processor, validate = program_processor_no_gtfn_exec

    shape = [5, 7, 9]

    inp = np.random.randint(0, 2, shape)
    inp = np_as_located_field(IDim, JDim, KDim)(inp)

    out = (
        np_as_located_field(IDim, JDim, KDim)(np.zeros(shape)),
        np_as_located_field(IDim, JDim, KDim)(np.zeros(shape)),
    )

    dom = {
        IDim: range(0, shape[0]),
        JDim: range(0, shape[1]),
        KDim: range(0, shape[2]),
    }
    run_processor(
        test_conditional[dom],
        program_processor,
        inp,
        out=out,
        offset_provider={},
    )
    if validate:
        assert np.all(out[0][inp == 1] == 1)
        assert np.all(out[1][inp == 1] == 2)
