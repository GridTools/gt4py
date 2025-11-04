# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from typing import Tuple
import pytest
from next_tests.integration_tests.cases import IDim, JDim, KDim, Koff, cartesian_case
from gt4py import next as gtx
from gt4py.next import int32
from gt4py.next.ffront.fbuiltins import where, broadcast
from next_tests.integration_tests import cases
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
)


@pytest.mark.uses_cartesian_shift
def test_where_k_offset(cartesian_case):
    @gtx.field_operator
    def fieldop_where_k_offset(
        inp: cases.IKField, k_index: gtx.Field[[KDim], gtx.IndexType]
    ) -> cases.IKField:
        return where(k_index > 0, inp(Koff[-1]), 2)

    @gtx.program
    def prog(
        inp: cases.IKField,
        k_index: gtx.Field[[KDim], gtx.IndexType],
        isize: int32,
        ksize: int32,
        out: cases.IKField,
    ):
        fieldop_where_k_offset(inp, k_index, out=out, domain={IDim: (0, isize), KDim: (1, ksize)})

    inp = cases.allocate(cartesian_case, fieldop_where_k_offset, "inp")()
    k_index = cases.allocate(
        cartesian_case, fieldop_where_k_offset, "k_index", strategy=cases.IndexInitializer()
    )()
    out = cases.allocate(cartesian_case, fieldop_where_k_offset, cases.RETURN)()

    ref = np.where(k_index.asnumpy() > 0, np.roll(inp.asnumpy(), 1, axis=1), out.asnumpy())
    isize = cartesian_case.default_sizes[IDim]
    ksize = cartesian_case.default_sizes[KDim]
    cases.verify(cartesian_case, prog, inp, k_index, isize, ksize, out=out, ref=ref)


def test_same_size_fields(cartesian_case):
    # Note boundaries can only be implemented with `where` if both fields have the same size, see `concat_where`
    @gtx.field_operator
    def testee(
        k: cases.KField, interior: cases.IJKField, boundary: cases.IJKField
    ) -> cases.IJKField:
        return where(k == 0, boundary, interior)

    k = cases.allocate(cartesian_case, testee, "k", strategy=cases.IndexInitializer())()
    interior = cases.allocate(cartesian_case, testee, "interior")()
    boundary = cases.allocate(cartesian_case, testee, "boundary")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    ref = np.where(
        k.asnumpy()[np.newaxis, np.newaxis, :] == 0, boundary.asnumpy(), interior.asnumpy()
    )

    cases.verify(cartesian_case, testee, k, interior, boundary, out=out, ref=ref)


@pytest.mark.uses_tuple_returns
def test_with_tuples(cartesian_case):
    @gtx.field_operator
    def testee(
        k: cases.KField,
        interior0: cases.IJKField,
        boundary0: cases.IJField,
        interior1: cases.IJKField,
        boundary1: cases.IJField,
    ) -> Tuple[cases.IJKField, cases.IJKField]:
        return where(
            broadcast(k, (IDim, JDim, KDim)) == 0, (boundary0, boundary1), (interior0, interior1)
        )

    k = cases.allocate(cartesian_case, testee, "k", strategy=cases.IndexInitializer())()
    interior0 = cases.allocate(cartesian_case, testee, "interior0")()
    boundary0 = cases.allocate(cartesian_case, testee, "boundary0")()
    interior1 = cases.allocate(cartesian_case, testee, "interior1")()
    boundary1 = cases.allocate(cartesian_case, testee, "boundary1")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    ref0 = np.where(
        k.asnumpy()[np.newaxis, np.newaxis, :] == 0,
        boundary0.asnumpy()[:, :, np.newaxis],
        interior0.asnumpy(),
    )
    ref1 = np.where(
        k.asnumpy()[np.newaxis, np.newaxis, :] == 0,
        boundary1.asnumpy()[:, :, np.newaxis],
        interior1.asnumpy(),
    )

    cases.verify(
        cartesian_case,
        testee,
        k,
        interior0,
        boundary0,
        interior1,
        boundary1,
        out=out,
        ref=(ref0, ref1),
    )
