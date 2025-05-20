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
from gt4py.next import broadcast
from gt4py.next.ffront.experimental import concat_where
from next_tests.integration_tests import cases
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
)

# TODO test non-Python int for embedded comparison


@pytest.mark.uses_frontend_concat_where
def test_concat_where_simple(cartesian_case):
    @gtx.field_operator
    def testee(ground: cases.IJKField, air: cases.IJKField) -> cases.IJKField:
        return concat_where(KDim > 0, air, ground)

    out = cases.allocate(cartesian_case, testee, cases.RETURN)()
    ground = cases.allocate(cartesian_case, testee, "ground")()
    air = cases.allocate(cartesian_case, testee, "air")()

    k = np.arange(0, cartesian_case.default_sizes[KDim])
    ref = np.where(k[np.newaxis, np.newaxis, :] == 0, ground.asnumpy(), air.asnumpy())
    cases.verify(cartesian_case, testee, ground, air, out=out, ref=ref)


@pytest.mark.uses_frontend_concat_where
def test_concat_where(cartesian_case):
    @gtx.field_operator
    def testee(ground: cases.IJKField, air: cases.IJKField) -> cases.IJKField:
        return concat_where(KDim == 0, ground, air)

    out = cases.allocate(cartesian_case, testee, cases.RETURN)()
    ground = cases.allocate(cartesian_case, testee, "ground")()
    air = cases.allocate(cartesian_case, testee, "air")()

    k = np.arange(0, cartesian_case.default_sizes[KDim])
    ref = np.where(k[np.newaxis, np.newaxis, :] == 0, ground.asnumpy(), air.asnumpy())
    cases.verify(cartesian_case, testee, ground, air, out=out, ref=ref)


@pytest.mark.uses_frontend_concat_where
def test_concat_where_non_overlapping(cartesian_case):
    @gtx.field_operator
    def testee(ground: cases.IJKField, air: cases.IJKField) -> cases.IJKField:
        return concat_where(KDim == 0, ground, air)

    out = cases.allocate(cartesian_case, testee, cases.RETURN)()
    ground = cases.allocate(
        cartesian_case, testee, "ground", domain=out.domain.slice_at[:, :, 0:1]
    )()
    air = cases.allocate(cartesian_case, testee, "air", domain=out.domain.slice_at[:, :, 1:])()

    ref = np.concatenate((ground.asnumpy(), air.asnumpy()), axis=2)
    cases.verify(cartesian_case, testee, ground, air, out=out, ref=ref)


@pytest.mark.uses_frontend_concat_where
def test_concat_where_single_level_broadcast(cartesian_case):
    @gtx.field_operator
    def testee(a: cases.KField, b: cases.IJKField) -> cases.IJKField:
        return concat_where(KDim == 0, a, b)

    out = cases.allocate(cartesian_case, testee, cases.RETURN)()
    a = cases.allocate(
        cartesian_case, testee, "a", domain=gtx.domain({KDim: out.domain.shape[2]})
    )()
    b = cases.allocate(cartesian_case, testee, "b", domain=out.domain.slice_at[:, :, 1:])()

    ref = np.concatenate(
        (
            np.tile(a.asnumpy()[0], (*b.domain.shape[0:2], 1)),
            b.asnumpy(),
        ),
        axis=2,
    )
    cases.verify(cartesian_case, testee, a, b, out=out, ref=ref)


@pytest.mark.uses_frontend_concat_where
def test_concat_where_single_level_restricted_domain_broadcast(cartesian_case):
    @gtx.field_operator
    def testee(a: cases.KField, b: cases.IJKField) -> cases.IJKField:
        return concat_where(KDim == 0, a, b)

    out = cases.allocate(cartesian_case, testee, cases.RETURN)()
    # note: this field is only defined on K: 0, 1, i.e., contains only a single value
    a = cases.allocate(cartesian_case, testee, "a", domain=gtx.domain({KDim: (0, 1)}))()
    b = cases.allocate(cartesian_case, testee, "b", domain=out.domain.slice_at[:, :, 1:])()

    ref = np.concatenate(
        (
            np.tile(a.asnumpy()[0], (*b.domain.shape[0:2], 1)),
            b.asnumpy(),
        ),
        axis=2,
    )
    cases.verify(cartesian_case, testee, a, b, out=out, ref=ref)


@pytest.mark.uses_frontend_concat_where
def test_boundary_single_layer_3d_bc(cartesian_case):
    @gtx.field_operator
    def testee(interior: cases.IJKField, boundary: cases.IJKField) -> cases.IJKField:
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
def test_boundary_single_layer_2d_bc(cartesian_case):
    @gtx.field_operator
    def testee(interior: cases.IJKField, boundary: cases.IJField) -> cases.IJKField:
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
def test_dimension_two_nested_conditions(cartesian_case):
    @gtx.field_operator
    def testee(interior: cases.IJKField, boundary: cases.IJKField) -> cases.IJKField:
        return concat_where((KDim < 2), boundary, concat_where((KDim >= 5), boundary, interior))

    interior = cases.allocate(cartesian_case, testee, "interior")()
    boundary = cases.allocate(cartesian_case, testee, "boundary")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    k = np.arange(0, cartesian_case.default_sizes[KDim])
    ref = np.where(
        (k[np.newaxis, np.newaxis, :] < 2) | (k[np.newaxis, np.newaxis, :] >= 5),
        boundary.asnumpy(),
        interior.asnumpy(),
    )
    cases.verify(cartesian_case, testee, interior, boundary, out=out, ref=ref)


# @pytest.mark.uses_frontend_concat_where
# def test_dimension_two_illegal_threeway_comparison(cartesian_case):
#     @gtx.field_operator
#     def testee(interior: cases.KField, boundary: cases.KField, nlev: np.int32) -> cases.KField:
#         return concat_where(0 < KDim < (nlev - 1), interior, boundary)

#     interior = cases.allocate(cartesian_case, testee, "interior")()
#     boundary = cases.allocate(cartesian_case, testee, "boundary")()
#     out = cases.allocate(cartesian_case, testee, cases.RETURN)()

#     nlev = cartesian_case.default_sizes[KDim]
#     k = np.arange(0, nlev)
#     ref = np.where((0 < k) & (k < (nlev - 1)), interior.asnumpy(), boundary.asnumpy())
#     with pytest.raises:  # TODO
#         cases.verify(cartesian_case, testee, interior, boundary, nlev, out=out, ref=ref)


@pytest.mark.uses_frontend_concat_where
def test_dimension_two_conditions_and(cartesian_case):
    @gtx.field_operator
    def testee(interior: cases.KField, boundary: cases.KField, nlev: np.int32) -> cases.KField:
        return concat_where((0 < KDim) & (KDim < (nlev - 1)), interior, boundary)

    interior = cases.allocate(cartesian_case, testee, "interior")()
    boundary = cases.allocate(cartesian_case, testee, "boundary")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    nlev = cartesian_case.default_sizes[KDim]
    k = np.arange(0, nlev)
    ref = np.where((0 < k) & (k < (nlev - 1)), interior.asnumpy(), boundary.asnumpy())
    cases.verify(cartesian_case, testee, interior, boundary, nlev, out=out, ref=ref)


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
def test_lap_like(cartesian_case):
    @gtx.field_operator
    def testee(
        input: cases.IJField, boundary: np.int32, shape: tuple[np.int32, np.int32]
    ) -> cases.IJField:
        # TODO add support for multi-dimensional concat_where masks
        return concat_where(
            (IDim == 0) | (IDim == shape[0] - 1),
            boundary,
            concat_where((JDim == 0) | (JDim == shape[1] - 1), boundary, input),
        )

    out = cases.allocate(cartesian_case, testee, cases.RETURN)()
    input = cases.allocate(
        cartesian_case, testee, "input", domain=out.domain.slice_at[1:-1, 1:-1]
    )()
    boundary = 2

    ref = np.full(out.domain.shape, np.nan)
    ref[0, :] = boundary
    ref[:, 0] = boundary
    ref[-1, :] = boundary
    ref[:, -1] = boundary
    ref[1:-1, 1:-1] = input.asnumpy()
    cases.verify(cartesian_case, testee, input, boundary, out.domain.shape, out=out, ref=ref)


@pytest.mark.uses_frontend_concat_where
@pytest.mark.uses_tuple_returns
def test_with_tuples(cartesian_case):
    @gtx.field_operator
    def testee(
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
        interior0,
        boundary0,
        interior1,
        boundary1,
        out=out,
        ref=(ref0, ref1),
    )


@pytest.mark.uses_frontend_concat_where
def test_nested_conditions_with_empty_branches(cartesian_case):
    @gtx.field_operator
    def testee(interior: cases.IField, boundary: cases.IField, N: gtx.int32) -> cases.IField:
        interior = concat_where(IDim == 0, boundary, interior)
        interior = concat_where((1 <= IDim) & (IDim < N - 1), interior * 2, interior)
        interior = concat_where(IDim == N - 1, boundary, interior)
        return interior

    interior = cases.allocate(cartesian_case, testee, "interior")()
    boundary = cases.allocate(cartesian_case, testee, "boundary")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()
    N = cartesian_case.default_sizes[IDim]

    i = np.arange(0, cartesian_case.default_sizes[IDim])
    ref = np.where(
        (i[:] == 0) | (i[:] == N - 1),
        boundary.asnumpy(),
        interior.asnumpy() * 2,
    )
    cases.verify(cartesian_case, testee, interior, boundary, N, out=out, ref=ref)


@pytest.mark.uses_tuple_returns
def test_with_tuples_different_domain(cartesian_case):
    @gtx.field_operator
    def testee(
        interior0: cases.IJKField,
        boundary0: cases.IJKField,
        interior1: cases.KField,
        boundary1: cases.KField,
    ) -> Tuple[cases.IJKField, cases.IJKField]:
        a, b = concat_where(KDim == 0, (boundary0, boundary1), (interior0, interior1))
        # the broadcast is only needed since we can not return fields on different domains yet
        return a, broadcast(b, (IDim, JDim, KDim))

    interior0 = cases.allocate(cartesian_case, testee, "interior0")()
    boundary0 = cases.allocate(cartesian_case, testee, "boundary0")()
    interior1 = cases.allocate(cartesian_case, testee, "interior1")()
    boundary1 = cases.allocate(cartesian_case, testee, "boundary1")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    k = np.arange(0, cartesian_case.default_sizes[KDim])
    ref0 = np.where(
        k[np.newaxis, np.newaxis, :] == 0,
        boundary0.asnumpy(),
        interior0.asnumpy(),
    )
    ref1 = np.where(
        k == 0,
        boundary1.asnumpy(),
        interior1.asnumpy(),
    )

    cases.verify(
        cartesian_case,
        testee,
        interior0,
        boundary0,
        interior1,
        boundary1,
        out=out,
        ref=(ref0, ref1),
    )
