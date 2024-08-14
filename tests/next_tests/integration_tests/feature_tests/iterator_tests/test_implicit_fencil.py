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
from gt4py.next.iterator.runtime import fundef

from next_tests.unit_tests.conftest import program_processor, run_processor


I = gtx.Dimension("I")

_isize = 10


@pytest.fixture
def dom():
    return {I: range(_isize)}


def a_field():
    return gtx.as_field([I], np.arange(0, _isize, dtype=np.float64))


def out_field():
    return gtx.as_field([I], np.zeros(shape=(_isize,)))


@fundef
def copy_stencil(inp):
    return deref(inp)


def test_single_argument(program_processor, dom):
    program_processor, validate = program_processor

    inp = a_field()
    out = out_field()

    run_processor(copy_stencil[dom], program_processor, inp, out=out, offset_provider={})
    if validate:
        assert np.allclose(inp.asnumpy(), out.asnumpy())


def test_2_arguments(program_processor, dom):
    program_processor, validate = program_processor

    @fundef
    def fun(inp0, inp1):
        return deref(inp0) + deref(inp1)

    inp0 = a_field()
    inp1 = a_field()
    out = out_field()

    run_processor(fun[dom], program_processor, inp0, inp1, out=out, offset_provider={})

    if validate:
        assert np.allclose(inp0.asnumpy() + inp1.asnumpy(), out.asnumpy())


def test_lambda_domain(program_processor):
    program_processor, validate = program_processor
    inp = a_field()
    out = out_field()

    dom = lambda: cartesian_domain(named_range(I, 0, 10))
    run_processor(copy_stencil[dom], program_processor, inp, out=out, offset_provider={})

    if validate:
        assert np.allclose(inp.asnumpy(), out.asnumpy())
