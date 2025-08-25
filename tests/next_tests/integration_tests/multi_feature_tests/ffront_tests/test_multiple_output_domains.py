# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import gt4py.next as gtx

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import IDim, JDim, cartesian_case
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import exec_alloc_descriptor

pytestmark = pytest.mark.uses_cartesian_shift


@gtx.field_operator
def testee_orig(
    a: gtx.Field[[IDim], gtx.float32], b: gtx.Field[[IDim], gtx.float32]
) -> tuple[gtx.Field[[IDim], gtx.float32], gtx.Field[[IDim], gtx.float32]]:
    return b, a


@gtx.program
def prog_orig(
    a: gtx.Field[[IDim], gtx.float32],
    b: gtx.Field[[IDim], gtx.float32],
    out_a: gtx.Field[[IDim], gtx.float32],
    out_b: gtx.Field[[IDim], gtx.float32],
):
    testee_orig(a, b, out=(out_b, out_a), domain={IDim: (0, 10)})


def test_program_orig(cartesian_case):
    a = cases.allocate(cartesian_case, prog_orig, "a")()
    b = cases.allocate(cartesian_case, prog_orig, "b")()
    out_a = cases.allocate(cartesian_case, prog_orig, "out_a")()
    out_b = cases.allocate(cartesian_case, prog_orig, "out_b")()

    cases.verify(
        cartesian_case,
        prog_orig,
        a,
        b,
        out_a,
        out_b,
        inout=(out_b, out_a),
        ref=(b, a),
    )


@gtx.field_operator
def testee(
    a: gtx.Field[[IDim], gtx.float32], b: gtx.Field[[JDim], gtx.float32]
) -> tuple[gtx.Field[[JDim], gtx.float32], gtx.Field[[IDim], gtx.float32]]:
    return b, a


@gtx.program
def prog(
    a: gtx.Field[[IDim], gtx.float32],
    b: gtx.Field[[JDim], gtx.float32],
    out_a: gtx.Field[[IDim], gtx.float32],
    out_b: gtx.Field[[JDim], gtx.float32],
    i_size: gtx.int32,
    j_size: gtx.int32,
):
    testee(a, b, out=(out_b, out_a), domain=({JDim: (0, j_size)}, {IDim: (0, i_size)}))


def test_program(cartesian_case):
    a = cases.allocate(cartesian_case, prog, "a")()
    b = cases.allocate(cartesian_case, prog, "b")()
    out_a = cases.allocate(cartesian_case, prog, "out_a")()
    out_b = cases.allocate(cartesian_case, prog, "out_b")()

    cases.verify(
        cartesian_case,
        prog,
        a,
        b,
        out_a,
        out_b,
        cartesian_case.default_sizes[IDim],
        cartesian_case.default_sizes[JDim],
        inout=(out_b, out_a),
        ref=(b, a),
    )


def test_direct_fo_orig(cartesian_case):
    a = cases.allocate(cartesian_case, testee_orig, "a")()
    b = cases.allocate(cartesian_case, testee_orig, "b")()
    out = cases.allocate(cartesian_case, testee_orig, cases.RETURN)()

    cases.verify(
        cartesian_case,
        testee_orig,
        a,
        b,
        out=out,
        ref=(b, a),
        domain={IDim: (0, cartesian_case.default_sizes[IDim])}
    )

# TODO:
#  - test without domain
#  - test with nested tuples
#  - test with different vertical domains KDim and KHalfDim
#  - test from  https://hackmd.io/m__8sBBATiqFWOPNMEPsfg
#  - unstructured test with Local dimensions e.g. Vertex, E2V and Edge

#
# def test_direct_fo(cartesian_case):
#     a = cases.allocate(cartesian_case, testee, "a")()
#     b = cases.allocate(cartesian_case, testee, "b")()
#     out = cases.allocate(cartesian_case, testee, cases.RETURN)()
#
#     cases.verify(
#         cartesian_case,
#         testee,
#         a,
#         b,
#         out=out,
#         ref=(b, a),
#         domain=(
#             {JDim: (0, cartesian_case.default_sizes[JDim])},
#             {IDim: (0, cartesian_case.default_sizes[IDim])},
#         ),
#     )


