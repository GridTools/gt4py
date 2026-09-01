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
    KDim,
    cartesian_case,
)
from next_tests.integration_tests.cases_utils import (
    Ioff,
    Koff,
    exec_alloc_descriptor,
)


@pytest.mark.uses_cartesian_shift
def test_cartesian_shift(cartesian_case):
    @gtx.field_operator
    def cartesian_shift_op(a: cases.IJKField) -> cases.IJKField:
        return a(IDim + 1)

    a = cases.allocate(cartesian_case, cartesian_shift_op, "a").extend({IDim: (0, 1)})()
    out = cases.allocate(cartesian_case, cartesian_shift_op, cases.RETURN)()

    cases.verify(cartesian_case, cartesian_shift_op, a, out=out, ref=a[1:])


@pytest.mark.uses_cartesian_shift
def test_fold_shifts(cartesian_case):
    """Shifting the result of an addition should work."""

    @gtx.field_operator
    def fold_shifts_op(a: cases.IJKField, b: cases.IJKField) -> cases.IJKField:
        tmp = a + b(IDim + 1)
        return tmp(IDim + 1)

    a = cases.allocate(cartesian_case, fold_shifts_op, "a").extend({cases.IDim: (0, 1)})()
    b = cases.allocate(cartesian_case, fold_shifts_op, "b").extend({cases.IDim: (0, 2)})()
    out = cases.allocate(cartesian_case, fold_shifts_op, cases.RETURN)()

    cases.verify(cartesian_case, fold_shifts_op, a, b, out=out, ref=a.ndarray[1:] + b.ndarray[2:])


@pytest.mark.uses_cartesian_shift
@pytest.mark.uses_dynamic_offsets
def test_offset_field(cartesian_case):
    ref = np.full(
        (cartesian_case.default_sizes[IDim], cartesian_case.default_sizes[KDim]), True, dtype=bool
    )

    @gtx.field_operator
    def offset_field_op(a: cases.IKField, offset_field: cases.IKField) -> gtx.Field[[IDim, KDim], bool]:
        a_i = a(as_offset(Ioff, offset_field))
        # note: this leads to an access to offset_field in
        # IDim: (0, out.size[I]), KDim: (0, out.size[K]+1)
        a_i_k = a_i(as_offset(Koff, offset_field))
        b_i = a(IDim + 1)
        b_i_k = b_i(KDim + 1)
        return a_i_k == b_i_k

    out = cases.allocate(cartesian_case, offset_field_op, cases.RETURN)()
    a = cases.allocate(cartesian_case, offset_field_op, "a").extend({IDim: (0, 1), KDim: (0, 1)})()
    offset_field = (
        cases.allocate(cartesian_case, offset_field_op, "offset_field")
        .strategy(cases.ConstInitializer(1))
        .extend({KDim: (0, 1)})()
    )  # see comment at a_i_k for domain bounds

    cases.verify(
        cartesian_case,
        offset_field_op,
        a,
        offset_field,
        out=out,
        ref=ref,
        comparison=lambda out, ref: np.all(out == ref),
    )


@pytest.mark.uses_dynamic_offsets
def test_offset_field_of_chained_ops(cartesian_case):
    """A dynamic offset on top of a chain of operations must be fused past all of them."""

    @gtx.field_operator
    def offset_field_of_chained_ops_op(a: cases.IKField, offset_field: cases.IKField) -> cases.IKField:
        b = a + 1
        c = b * 2
        return c(as_offset(Koff, offset_field))

    out = cases.allocate(cartesian_case, offset_field_of_chained_ops_op, cases.RETURN)()
    a = cases.allocate(cartesian_case, offset_field_of_chained_ops_op, "a").extend({KDim: (0, 1)})()
    offset_field = cases.allocate(
        cartesian_case, offset_field_of_chained_ops_op, "offset_field", strategy=cases.ConstInitializer(1)
    )()

    cases.verify(
        cartesian_case,
        offset_field_of_chained_ops_op,
        a,
        offset_field,
        out=out,
        ref=(a.asnumpy()[:, 1:] + 1) * 2,
    )
