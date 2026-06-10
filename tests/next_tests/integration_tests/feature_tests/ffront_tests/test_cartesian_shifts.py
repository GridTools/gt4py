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
from gt4py.next.ffront.experimental import as_offset

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import (
    IDim,
    Ioff,
    KDim,
    Koff,
    cartesian_case,
)
from next_tests.integration_tests.cases_utils import (
    exec_alloc_descriptor,
)


@pytest.mark.uses_cartesian_shift
def test_cartesian_shift(cartesian_case):
    @gtx.field_operator
    def testee(a: cases.IJKField) -> cases.IJKField:
        return a(Ioff[1])

    a = cases.allocate(cartesian_case, testee, "a").extend({IDim: (0, 1)})()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    cases.verify(cartesian_case, testee, a, out=out, ref=a[1:])


@pytest.mark.uses_cartesian_shift
def test_fold_shifts(cartesian_case):
    """Shifting the result of an addition should work."""

    @gtx.field_operator
    def testee(a: cases.IJKField, b: cases.IJKField) -> cases.IJKField:
        tmp = a + b(Ioff[1])
        return tmp(Ioff[1])

    a = cases.allocate(cartesian_case, testee, "a").extend({cases.IDim: (0, 1)})()
    b = cases.allocate(cartesian_case, testee, "b").extend({cases.IDim: (0, 2)})()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    cases.verify(cartesian_case, testee, a, b, out=out, ref=a.ndarray[1:] + b.ndarray[2:])


@pytest.mark.uses_dynamic_offsets
def test_offset_field(cartesian_case):
    ref = np.full(
        (cartesian_case.default_sizes[IDim], cartesian_case.default_sizes[KDim]), True, dtype=bool
    )

    @gtx.field_operator
    def testee(a: cases.IKField, offset_field: cases.IKField) -> gtx.Field[[IDim, KDim], bool]:
        a_i = a(as_offset(Ioff, offset_field))
        # note: this leads to an access to offset_field in
        # IDim: (0, out.size[I]), KDim: (0, out.size[K]+1)
        a_i_k = a_i(as_offset(Koff, offset_field))
        b_i = a(Ioff[1])
        b_i_k = b_i(Koff[1])
        return a_i_k == b_i_k

    out = cases.allocate(cartesian_case, testee, cases.RETURN)()
    a = cases.allocate(cartesian_case, testee, "a").extend({IDim: (0, 1), KDim: (0, 1)})()
    offset_field = (
        cases.allocate(cartesian_case, testee, "offset_field")
        .strategy(cases.ConstInitializer(1))
        .extend({KDim: (0, 1)})()
    )  # see comment at a_i_k for domain bounds

    cases.verify(
        cartesian_case,
        testee,
        a,
        offset_field,
        out=out,
        offset_provider={"Ioff": IDim, "Koff": KDim},
        ref=ref,
        comparison=lambda out, ref: np.all(out == ref),
    )
