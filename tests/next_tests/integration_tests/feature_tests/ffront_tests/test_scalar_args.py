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
from gt4py.next import broadcast, int32

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import (
    IDim,
    Ioff,
    Vertex,
    cartesian_case,
    unstructured_case,
)
from next_tests.integration_tests.cases_utils import (
    exec_alloc_descriptor,
    mesh_descriptor,
)


def test_scalar_arg(unstructured_case):
    """Test scalar argument being turned into 0-dim field."""

    @gtx.field_operator
    def testee(a: int32) -> cases.VField:
        return broadcast(a + 1, (Vertex,))

    cases.verify_with_default_data(
        unstructured_case,
        testee,
        ref=lambda a: np.full([unstructured_case.default_sizes[Vertex]], a + 1, dtype=int32),
        comparison=lambda a, b: np.all(a == b),
    )


def test_np_bool_scalar_arg(unstructured_case):
    """Test scalar argument being turned into 0-dim field."""

    @gtx.field_operator
    def testee(a: gtx.bool) -> cases.VBoolField:
        return broadcast(not a, (Vertex,))

    a = np.bool_(True)  # explicitly using a np.bool

    ref = np.full([unstructured_case.default_sizes[Vertex]], not a, dtype=np.bool_)
    out = cases.allocate(unstructured_case, testee, cases.RETURN)()

    cases.verify(unstructured_case, testee, a, out=out, ref=ref)


def test_nested_scalar_arg(unstructured_case):
    @gtx.field_operator
    def testee_inner(a: int32) -> cases.VField:
        return broadcast(a + 1, (Vertex,))

    @gtx.field_operator
    def testee(a: int32) -> cases.VField:
        return testee_inner(a + 1)

    cases.verify_with_default_data(
        unstructured_case,
        testee,
        ref=lambda a: np.full([unstructured_case.default_sizes[Vertex]], a + 2, dtype=int32),
    )


@pytest.mark.uses_cartesian_shift
def test_scalar_arg_with_field(cartesian_case):
    @gtx.field_operator
    def testee(a: cases.IJKField, b: int32) -> cases.IJKField:
        tmp = b * a
        return tmp(Ioff[1])

    a = cases.allocate(cartesian_case, testee, "a").extend({IDim: (0, 1)})()
    b = cases.allocate(cartesian_case, testee, "b")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()
    ref = a[1:] * b

    cases.verify(cartesian_case, testee, a, b, out=out, ref=ref)


def test_double_use_scalar(cartesian_case):
    # TODO(tehrengruber): This should be a regression test on ITIR level, but tracing doesn't
    #  work for this case.
    @gtx.field_operator
    def testee(a: int32, b: int32, c: cases.IField) -> cases.IField:
        tmp = a * b
        tmp2 = tmp * tmp
        # important part here is that we use the intermediate twice so that it is
        # not inlined
        return tmp2 * tmp2 * c

    cases.verify_with_default_data(
        cartesian_case, testee, ref=lambda a, b, c: a * b * a * b * a * b * a * b * c
    )
