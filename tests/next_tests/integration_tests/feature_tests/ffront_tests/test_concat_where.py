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
from next_tests.integration_tests.cases import IDim, JDim, KDim, cartesian_case
from gt4py import next as gtx
from gt4py.next import errors
from gt4py.next.ffront.experimental import concat_where
from next_tests.integration_tests import cases
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
)


@pytest.mark.uses_frontend_concat_where
def test_concat_where(cartesian_case):
    @gtx.field_operator
    def testee(ground: cases.IJKField, air: cases.IJKField) -> cases.IJKField:
        return concat_where(KDim == 0, ground, air)

    out = cases.allocate(cartesian_case, testee, cases.RETURN)()
    ground = cases.allocate(cartesian_case, testee, "ground")()
    air = cases.allocate(cartesian_case, testee, "air")()

    k = np.arange(0, cartesian_case.default_sizes[KDim])
    ref = np.where(
        k[np.newaxis, np.newaxis, :] == 0, ground.asnumpy(), air.asnumpy()
    )
    cases.verify(cartesian_case, testee, ground, air, out=out, ref=ref)


@pytest.mark.uses_frontend_concat_where
def test_concat_where_non_overlapping(cartesian_case):
    @gtx.field_operator
    def testee(ground: cases.IJKField, air: cases.IJKField) -> cases.IJKField:
        return concat_where(KDim == 0, ground, air)

    out = cases.allocate(cartesian_case, testee, cases.RETURN)()
    ground = cases.allocate(cartesian_case, testee, "ground", domain=out.domain.slice_at[:, :, 0:1])()
    air = cases.allocate(cartesian_case, testee, "air", domain=out.domain.slice_at[:, :, 1:])()

    ref = np.concatenate((ground.asnumpy(), air.asnumpy()), axis=2)
    cases.verify(cartesian_case, testee, ground, air, out=out, ref=ref)


@pytest.mark.uses_frontend_concat_where
def test_concat_where_non_overlapping_different_dims(cartesian_case):
    @gtx.field_operator
    def testee(
        ground: cases.KField, # note: boundary field is only defined in K
        air: cases.IJKField
    ) -> cases.IJKField:
        return concat_where(KDim == 0, ground, air)

    out = cases.allocate(cartesian_case, testee, cases.RETURN)()
    ground = cases.allocate(cartesian_case, testee, "ground", domain=gtx.domain({KDim: (0, 1)}))()
    air = cases.allocate(cartesian_case, testee, "air", domain=out.domain.slice_at[:, :, 1:])()

    ref = np.concatenate((np.tile(ground.asnumpy(),(*air.domain.shape[0:2], len(ground.domain[KDim].unit_range))), air.asnumpy()), axis=2)

    cases.verify(cartesian_case, testee, ground, air, out=out, ref=ref)


@pytest.mark.uses_frontend_concat_where
def test_dimension_two_nested_conditions(cartesian_case):
    @gtx.field_operator
    def testee(
        interior: cases.IJKField, boundary: cases.IJKField
    ) -> cases.IJKField:
        return concat_where((KDim < 2), boundary, concat_where((KDim >= 5), boundary, interior))

    interior = cases.allocate(cartesian_case, testee, "interior")()
    boundary = cases.allocate(cartesian_case, testee, "boundary")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    k = np.arange(0, cartesian_case.default_sizes[KDim])
    ref = np.where(
        (k[np.newaxis, np.newaxis, :] < 2)
        | (k[np.newaxis, np.newaxis, :] >= 5),
        boundary.asnumpy(),
        interior.asnumpy(),
    )
    cases.verify(cartesian_case, testee, interior, boundary, out=out, ref=ref)


@pytest.mark.uses_frontend_concat_where
def test_dimension_two_conditions_and(cartesian_case):
    @gtx.field_operator
    def testee(interior: cases.KField, boundary: cases.KField) -> cases.KField:
        return concat_where(((KDim > 2) & (KDim <= 5)), interior, boundary)

    interior = cases.allocate(cartesian_case, testee, "interior")()
    boundary = cases.allocate(cartesian_case, testee, "boundary")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    k = np.arange(0, cartesian_case.default_sizes[KDim])
    ref = np.where((k > 2) & (k <= 5), interior.asnumpy(), boundary.asnumpy())
    cases.verify(cartesian_case, testee, interior, boundary, out=out, ref=ref)


@pytest.mark.uses_frontend_concat_where
def test_dimension_two_conditions_eq(cartesian_case):
    @gtx.field_operator
    def testee(interior: cases.KField, boundary: cases.KField) -> cases.KField:
        return concat_where((KDim == 2), interior, boundary)

    interior = cases.allocate(cartesian_case, testee, "interior")()
    boundary = cases.allocate(cartesian_case, testee, "boundary")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    k = np.arange(0, cartesian_case.default_sizes[KDim])
    ref = np.where(k == 2, interior.asnumpy(), boundary.asnumpy())
    cases.verify(cartesian_case, testee, interior, boundary, out=out, ref=ref)


@pytest.mark.uses_frontend_concat_where
def test_dimension_two_conditions_or(cartesian_case):
    @gtx.field_operator
    def testee(interior: cases.KField, boundary: cases.KField) -> cases.KField:
        return concat_where(((KDim < 2) | (KDim >= 5)), boundary, interior)

    interior = cases.allocate(cartesian_case, testee, "interior")()
    boundary = cases.allocate(cartesian_case, testee, "boundary")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    k = np.arange(0, cartesian_case.default_sizes[KDim])
    ref = np.where((k < 2) | (k >= 5), boundary.asnumpy(), interior.asnumpy())
    cases.verify(cartesian_case, testee, interior, boundary, out=out, ref=ref)


@pytest.mark.uses_frontend_concat_where
def test_boundary_horizontal_slice(cartesian_case):
    @gtx.field_operator
    def testee(
        interior: cases.IJKField, boundary: cases.IJField
    ) -> cases.IJKField:
        return concat_where(KDim == 0, boundary, interior)

    interior = cases.allocate(cartesian_case, testee, "interior")()
    boundary = cases.allocate(cartesian_case, testee, "boundary")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    k = np.arange(0, cartesian_case.default_sizes[KDim])
    ref = np.where(
        k[np.newaxis, np.newaxis, :] == 0,
        boundary.asnumpy()[:, :, np.newaxis],
        interior.asnumpy(),
    )

    cases.verify(cartesian_case, testee, interior, boundary, out=out, ref=ref)


@pytest.mark.uses_frontend_concat_where
def test_boundary_single_layer(cartesian_case):
    @gtx.field_operator
    def testee(
        interior: cases.IJKField, boundary: cases.IJKField
    ) -> cases.IJKField:
        return concat_where(KDim == 0, boundary, interior)

    interior = cases.allocate(cartesian_case, testee, "interior")()
    boundary = cases.allocate(cartesian_case, testee, "boundary", sizes={KDim: 1})()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    k = np.arange(0, cartesian_case.default_sizes[KDim])
    ref = np.where(
        k[np.newaxis, np.newaxis, :] == 0,
        np.broadcast_to(boundary.asnumpy(), interior.shape),
        interior.asnumpy(),
    )

    cases.verify(cartesian_case, testee, interior, boundary, out=out, ref=ref)


@pytest.mark.uses_frontend_concat_where
@pytest.mark.uses_tuple_returns
def test_with_tuples(cartesian_case):
    pytest.skip("Not implemented in the frontend.")
    @gtx.field_operator
    def testee(
        k: cases.KField,
        interior0: cases.IJKField,
        boundary0: cases.IJField,
        interior1: cases.IJKField,
        boundary1: cases.IJField,
    ) -> Tuple[cases.IJKField, cases.IJKField]:
        return concat_where(KDim == 0, (boundary0, boundary1), (interior0, interior1))

    interior0 = cases.allocate(cartesian_case, testee, "interior0")()
    boundary0 = cases.allocate(cartesian_case, testee, "boundary0")()
    interior1 = cases.allocate(cartesian_case, testee, "interior1")()
    boundary1 = cases.allocate(cartesian_case, testee, "boundary1")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    k = np.arange(0, cartesian_case.default_sizes[KDim])
    ref0 = np.where(
        k[np.newaxis, np.newaxis, :] == 0,
        boundary0.asnumpy()[:, :, np.newaxis],
        interior0.asnumpy(),
    )
    ref1 = np.where(
        k[np.newaxis, np.newaxis, :] == 0,
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
