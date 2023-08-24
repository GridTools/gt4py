# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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

import gt4py.next as gtx
from gt4py.next.iterator.builtins import *
from gt4py.next.iterator.runtime import fundef
from gt4py.next.program_processors.runners.dace_iterator import run_dace_iterator

from next_tests.unit_tests.conftest import program_processor, run_processor


I = gtx.Dimension("I")

_isize = 10


@pytest.fixture
def dom():
    return {I: range(_isize)}


def a_field():
    return gtx.np_as_located_field(I)(np.arange(0, _isize, dtype=np.float64))


def out_field():
    return gtx.np_as_located_field(I)(np.zeros(shape=(_isize,)))


@fundef
def copy_stencil(inp):
    return deref(inp)


def test_single_argument(program_processor, dom):
    program_processor, validate = program_processor

    inp = a_field()
    out = out_field()

    run_processor(copy_stencil[dom], program_processor, inp, out=out, offset_provider={})
    if validate:
        assert np.allclose(inp, out)


def test_2_arguments(program_processor, dom):
    program_processor, validate = program_processor
    if program_processor == run_dace_iterator:
        pytest.xfail(
            "Not supported in DaCe backend: argument types are not propagated for ITIR tests"
        )

    @fundef
    def fun(inp0, inp1):
        return deref(inp0) + deref(inp1)

    inp0 = a_field()
    inp1 = a_field()
    out = out_field()

    run_processor(fun[dom], program_processor, inp0, inp1, out=out, offset_provider={})

    if validate:
        assert np.allclose(inp0.array() + inp1.array(), out)


def test_lambda_domain(program_processor):
    program_processor, validate = program_processor
    inp = a_field()
    out = out_field()

    dom = lambda: cartesian_domain(named_range(I, 0, 10))
    run_processor(copy_stencil[dom], program_processor, inp, out=out, offset_provider={})

    if validate:
        assert np.allclose(inp, out)
