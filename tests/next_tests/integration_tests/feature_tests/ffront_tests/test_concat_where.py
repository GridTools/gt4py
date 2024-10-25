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
from next_tests.integration_tests.cases import KDim, cartesian_case
from gt4py import next as gtx
from gt4py.next.ffront.experimental import concat_where
from next_tests.integration_tests import cases
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
)


def test_boundary_same_size_fields(cartesian_case):
    @gtx.field_operator
    def testee(
        k: cases.KField, interior: cases.IJKField, boundary: cases.IJKField
    ) -> cases.IJKField:
        return concat_where(k == 0, boundary, interior)

    k = cases.allocate(cartesian_case, testee, "k", strategy=cases.IndexInitializer())()
    interior = cases.allocate(cartesian_case, testee, "interior")()
    boundary = cases.allocate(cartesian_case, testee, "boundary")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    ref = np.where(
        k.asnumpy()[np.newaxis, np.newaxis, :] == 0, boundary.asnumpy(), interior.asnumpy()
    )

    cases.verify(cartesian_case, testee, k, interior, boundary, out=out, ref=ref)


def test_dimension(cartesian_case):
    @gtx.field_operator
    def testee(
        k: cases.KField, interior: cases.IJKField, boundary: cases.IJKField
    ) -> cases.IJKField:
        return concat_where(KDim <= 2, boundary, interior)

    k = cases.allocate(cartesian_case, testee, "k", strategy=cases.IndexInitializer())()
    interior = cases.allocate(cartesian_case, testee, "interior")()
    boundary = cases.allocate(cartesian_case, testee, "boundary")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    ref = np.where(
        k.asnumpy()[np.newaxis, np.newaxis, :] <= 0, boundary.asnumpy(), interior.asnumpy()
    )

    cases.verify(cartesian_case, testee, k, interior, boundary, out=out, ref=ref)  # TODO


def test_boundary_horizontal_slice(cartesian_case):
    @gtx.field_operator
    def testee(
        k: cases.KField, interior: cases.IJKField, boundary: cases.IJField
    ) -> cases.IJKField:
        return concat_where(k == 0, boundary, interior)

    k = cases.allocate(cartesian_case, testee, "k", strategy=cases.IndexInitializer())()
    interior = cases.allocate(cartesian_case, testee, "interior")()
    boundary = cases.allocate(cartesian_case, testee, "boundary")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    ref = np.where(
        k.asnumpy()[np.newaxis, np.newaxis, :] == 0,
        boundary.asnumpy()[:, :, np.newaxis],
        interior.asnumpy(),
    )

    cases.verify(cartesian_case, testee, k, interior, boundary, out=out, ref=ref)


def test_boundary_single_layer(cartesian_case):
    @gtx.field_operator
    def testee(
        k: cases.KField, interior: cases.IJKField, boundary: cases.IJKField
    ) -> cases.IJKField:
        return concat_where(k == 0, boundary, interior)

    k = cases.allocate(cartesian_case, testee, "k", strategy=cases.IndexInitializer())()
    interior = cases.allocate(cartesian_case, testee, "interior")()
    boundary = cases.allocate(cartesian_case, testee, "boundary", sizes={KDim: 1})()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    ref = np.where(
        k.asnumpy()[np.newaxis, np.newaxis, :] == 0,
        np.broadcast_to(boundary.asnumpy(), interior.shape),
        interior.asnumpy(),
    )

    cases.verify(cartesian_case, testee, k, interior, boundary, out=out, ref=ref)


def test_alternating_mask(cartesian_case):
    @gtx.field_operator
    def testee(k: cases.KField, f0: cases.IJKField, f1: cases.IJKField) -> cases.IJKField:
        return concat_where(k % 2 == 0, f1, f0)

    k = cases.allocate(cartesian_case, testee, "k", strategy=cases.IndexInitializer())()
    f0 = cases.allocate(cartesian_case, testee, "f0")()
    f1 = cases.allocate(cartesian_case, testee, "f1")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    ref = np.where(k.asnumpy()[np.newaxis, np.newaxis, :] % 2 == 0, f1.asnumpy(), f0.asnumpy())

    cases.verify(cartesian_case, testee, k, f0, f1, out=out, ref=ref)


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
        return concat_where(k == 0, (boundary0, boundary1), (interior0, interior1))

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
