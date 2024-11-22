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
from gt4py.next import backend
from gt4py.next.iterator.builtins import *
from gt4py.next.iterator.runtime import set_at, fendef, fundef
from gt4py.next.program_processors.formatters import gtfn as gtfn_formatters
from gt4py.next.program_processors.runners import gtfn

from next_tests.integration_tests.cases import IDim, JDim, KDim
from next_tests.unit_tests.conftest import program_processor, run_processor


@fundef
def tridiag_forward(state, a, b, c, d):
    return make_tuple(
        deref(c) / (deref(b) - deref(a) * tuple_get(0, state)),
        (deref(d) - deref(a) * tuple_get(1, state)) / (deref(b) - deref(a) * tuple_get(0, state)),
    )


@fundef
def tridiag_backward(x_kp1, cpdp):
    cpdpv = deref(cpdp)
    cp = tuple_get(0, cpdpv)
    dp = tuple_get(1, cpdpv)
    return dp - cp * x_kp1


@fundef
def tridiag_backward2(x_kp1, cp, dp):
    return deref(dp) - deref(cp) * x_kp1


@fundef
def solve_tridiag(domain, a, b, c, d):
    cpdp = as_fieldop(scan(tridiag_forward, True, make_tuple(0.0, 0.0)), domain)(a, b, c, d)
    return as_fieldop(scan(tridiag_backward, False, 0.0), domain)(cpdp)


@fundef
def solve_tridiag2(domain, a, b, c, d):
    cpdp = as_fieldop(scan(tridiag_forward, True, make_tuple(0.0, 0.0)), domain)(a, b, c, d)
    return as_fieldop(scan(tridiag_backward2, False, 0.0), domain)(
        tuple_get(0, cpdp), tuple_get(1, cpdp)
    )


@pytest.fixture
def tridiag_reference():
    shape = (3, 7, 5)
    rng = np.random.default_rng()
    a = rng.normal(size=shape)
    b = rng.normal(size=shape) * 2
    c = rng.normal(size=shape)
    d = rng.normal(size=shape)

    matrices = np.zeros(shape + shape[-1:])
    i = np.arange(shape[2])
    matrices[:, :, i[1:], i[:-1]] = a[:, :, 1:]
    matrices[:, :, i, i] = b
    matrices[:, :, i[:-1], i[1:]] = c[:, :, :-1]
    x = np.linalg.solve(matrices, d)
    return a, b, c, d, x


@fendef
def fen_solve_tridiag(i_size, j_size, k_size, a, b, c, d, x):
    domain = cartesian_domain(
        named_range(IDim, 0, i_size), named_range(JDim, 0, j_size), named_range(KDim, 0, k_size)
    )
    set_at(solve_tridiag(domain, a, b, c, d), domain, x)


@fendef
def fen_solve_tridiag2(i_size, j_size, k_size, a, b, c, d, x):
    domain = cartesian_domain(
        named_range(IDim, 0, i_size), named_range(JDim, 0, j_size), named_range(KDim, 0, k_size)
    )
    set_at(solve_tridiag2(domain, a, b, c, d), domain, x)


@pytest.mark.parametrize("fencil", [fen_solve_tridiag, fen_solve_tridiag2])
def test_tridiag(fencil, tridiag_reference, program_processor):
    program_processor, validate = program_processor

    if isinstance(program_processor, backend.Backend) and "dace" in program_processor.name:
        pytest.xfail("Dace ITIR backend doesn't support the IR format used in this test.")

    a, b, c, d, x = tridiag_reference
    shape = a.shape
    as_3d_field = gtx.as_field.partial([IDim, JDim, KDim])
    a_s = as_3d_field(a)
    b_s = as_3d_field(b)
    c_s = as_3d_field(c)
    d_s = as_3d_field(d)
    x_s = as_3d_field(np.zeros_like(x))

    run_processor(
        fencil,
        program_processor,
        shape[0],
        shape[1],
        shape[2],
        a_s,
        b_s,
        c_s,
        d_s,
        x_s,
        offset_provider={},
        column_axis=KDim,
    )

    if validate:
        assert np.allclose(x, x_s.asnumpy())
