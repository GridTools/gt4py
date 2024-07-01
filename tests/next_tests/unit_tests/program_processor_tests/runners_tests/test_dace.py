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
"""Test specific features of DaCe backends."""

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next import (
    int32,
)

from next_tests.integration_tests import cases
from gt4py.next.ffront.fbuiltins import where
from next_tests.integration_tests.cases import (
    cartesian_case,
)
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
)


def test_dace_fastcall(cartesian_case):
    """Test reuse of arguments between program calls by means of SDFG fastcall API."""

    if not cartesian_case.executor or "dace" not in cartesian_case.executor.__name__:
        pytest.skip("DaCe-specific testcase.")

    @gtx.field_operator
    def testee(
        a: cases.IField,
        a_idx: cases.IField,
        unused_field: cases.IField,
        a0: int32,
        a1: int32,
        a2: int32,
        unused_scalar: int32,
    ) -> cases.IField:
        t0 = where(a_idx == 0, a + a0, a)
        t1 = where(a_idx == 1, t0 + a1, t0)
        t2 = where(a_idx == 2, t1 + a2, t1)
        return t2

    a = cases.allocate(cartesian_case, testee, "a")()
    a_index = cases.allocate(cartesian_case, testee, "a_idx", strategy=cases.IndexInitializer())()
    unused_field = cases.allocate(cartesian_case, testee, "unused_field")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    a_offset = np.random.randint(1, 100, size=4, dtype=np.int32)
    a_ref = lambda a, a0, a1, a2: [a[0] + a0, a[1] + a1, a[2] + a2, *a[3:]]

    # SDFG fastcall API cannot be used on first run: the SDFG arguments will have to be constructed
    cases.verify(
        cartesian_case,
        testee,
        a,
        a_index,
        unused_field,
        *a_offset,
        out=out,
        ref=a_ref(a.asnumpy(), *a_offset[0:3]),
    )

    # modify the scalar arguments, used and unused ones: the SDFG fastcall API should be used
    for i in range(4):
        a_offset[i] += 1
        cases.verify(
            cartesian_case,
            testee,
            a,
            a_index,
            unused_field,
            *a_offset,
            out=out,
            ref=a_ref(a.asnumpy(), *a_offset[0:3]),
        )

    # modify content of current buffer: the SDFG fastcall API should be used
    a[0] += 1
    cases.verify(
        cartesian_case,
        testee,
        a,
        a_index,
        unused_field,
        *a_offset,
        out=out,
        ref=a_ref(a.asnumpy(), *a_offset[0:3]),
    )

    # pass a new buffer, which should trigger reconstruct of SDFG arguments: fastcall API will not be used
    a = cases.allocate(cartesian_case, testee, "a")()
    cases.verify(
        cartesian_case,
        testee,
        a,
        a_index,
        unused_field,
        *a_offset,
        out=out,
        ref=a_ref(a.asnumpy(), *a_offset[0:3]),
    )
