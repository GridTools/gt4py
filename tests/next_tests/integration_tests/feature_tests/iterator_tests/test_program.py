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
from gt4py.next.iterator.builtins import as_fieldop, cartesian_domain, deref, named_range
from gt4py.next.iterator.runtime import fendef, fundef, set_at

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


@fendef
def copy_program(inp, out, size):
    set_at(
        as_fieldop(copy_stencil, cartesian_domain(named_range(I, 0, size)))(inp),
        cartesian_domain(named_range(I, 0, size)),
        out,
    )


# @fundef
# def plus_stencil(inp0,inp1):
#     return plus(deref(inp0),deref(inp1))

# set_at(
#         # as_fieldop(copy_stencil, domain=cartesian_domain(named_range(I, 0, size)))(inp),
#         as_fieldop(plus_stencil)(inp0, as_fieldop(plus_stencil)(inp1,inp2)),
#         cartesian_domain(named_range(I, 0, size)),
#         out,
#     )


def test_prog():
    validate = True

    inp = a_field()
    out = out_field()

    copy_program(inp, out, _isize, offset_provider={})
    if validate:
        assert np.allclose(inp.asnumpy(), out.asnumpy())


# example for
# @field_operator
# def sum(a, b, c):
#     a + b + c

# def plus(a,b):
#     return deref(a)+deref(b)

# def sum_prog(a, b, c, out):
#     set_at(as_fieldop(plus)(a, as_fieldop(plus)(b, c)), out.domain, out)
