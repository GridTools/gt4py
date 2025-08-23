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

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import IDim, JDim, cartesian_case
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
)

pytestmark = pytest.mark.uses_cartesian_shift

# TODO move to a proper location


@gtx.field_operator
def foo(
    a: gtx.Field[[IDim], gtx.float32], b: gtx.Field[[JDim], gtx.float32]
) -> tuple[gtx.Field[[JDim], gtx.float32], gtx.Field[[IDim], gtx.float32]]:
    return b, a


@gtx.program
def foo_program(
    a: gtx.Field[[IDim], gtx.float32],
    b: gtx.Field[[JDim], gtx.float32],
    out_a: gtx.Field[[IDim], gtx.float32],
    out_b: gtx.Field[[JDim], gtx.float32],
    i_size: gtx.int32,
    j_size: gtx.int32,
):
    foo(a, b, out=(out_b, out_a), domain=({JDim: (0, j_size)}, {IDim: (0, i_size)}))


def test_foo(cartesian_case):
    a = cases.allocate(cartesian_case, foo_program, "a")()
    b = cases.allocate(cartesian_case, foo_program, "b")()
    out_a = cases.allocate(cartesian_case, foo_program, "out_a")()
    out_b = cases.allocate(cartesian_case, foo_program, "out_b")()

    cases.verify(
        cartesian_case,
        foo_program,
        a,
        b,
        out_a,
        out_b,
        cartesian_case.default_sizes[IDim],
        cartesian_case.default_sizes[JDim],
        inout=(out_b, out_a),
        ref=(b, a),
    )
