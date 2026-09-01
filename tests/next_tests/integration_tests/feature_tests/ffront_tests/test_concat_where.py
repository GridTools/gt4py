# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from next_tests.integration_tests.cases import IDim, JDim, KDim, cartesian_case
from gt4py import next as gtx
from gt4py.next import broadcast
from gt4py.next.ffront.experimental import concat_where
from next_tests.integration_tests import cases
from next_tests.integration_tests.cases_utils import (
    exec_alloc_descriptor,
)

pytestmark = pytest.mark.uses_concat_where


@pytest.fixture(params=[False, True], ids=["dynamic_domains", "static_domains"])
def static_domains(request) -> bool:
    """Fixture to select compilation with dynamic or statically known domain bounds."""
    return request.param


def test_concat_where_simple(cartesian_case, static_domains: bool):
    @gtx.field_operator(static_domains=static_domains)
    def concat_where_simple_op(ground: cases.IJKField, air: cases.IJKField) -> cases.IJKField:
        return concat_where(KDim > 0, air, ground)

    k = np.arange(0, cartesian_case.default_sizes[KDim])
    cases.verify_with_default_data(
        cartesian_case,
        concat_where_simple_op,
        lambda ground, air: np.where(k[np.newaxis, np.newaxis, :] == 0, ground, air),
    )


def test_concat_where(cartesian_case, static_domains: bool):
    @gtx.field_operator(static_domains=static_domains)
    def concat_where_op(ground: cases.IJKField, air: cases.IJKField) -> cases.IJKField:
        return concat_where(KDim == 0, ground, air)

    k = np.arange(0, cartesian_case.default_sizes[KDim])
    cases.verify_with_default_data(
        cartesian_case,
        concat_where_op,
        lambda ground, air: np.where(k[np.newaxis, np.newaxis, :] == 0, ground, air),
    )


def test_concat_where_non_overlapping(cartesian_case, static_domains: bool):
    """Fields only defined in their respective region in concat_where."""

    @gtx.field_operator(static_domains=static_domains)
    def concat_where_non_overlapping_op(ground: cases.IJKField, air: cases.IJKField) -> cases.IJKField:
        return concat_where(KDim == 0, ground, air)

    out = cases.allocate(cartesian_case, concat_where_non_overlapping_op, cases.RETURN)()
    ground = cases.allocate(
        cartesian_case, concat_where_non_overlapping_op, "ground", domain=out.domain.slice_at[:, :, 0:1]
    )()
    air = cases.allocate(cartesian_case, concat_where_non_overlapping_op, "air", domain=out.domain.slice_at[:, :, 1:])()

    ref = np.concatenate((ground.asnumpy(), air.asnumpy()), axis=2)
    cases.verify(cartesian_case, concat_where_non_overlapping_op, ground, air, out=out, ref=ref)


def test_concat_where_empty_branch(cartesian_case, static_domains: bool):
    @gtx.field_operator(static_domains=static_domains)
    def concat_where_empty_branch_op(a: cases.IJKField, b: cases.IJKField, N: np.int32) -> cases.IJKField:
        return concat_where(IDim < N, a, b * 2)

    out = cases.allocate(cartesian_case, concat_where_empty_branch_op, cases.RETURN)()
    a = cases.allocate(cartesian_case, concat_where_empty_branch_op, "a")()
    b = cases.allocate(cartesian_case, concat_where_empty_branch_op, "b")()

    N = out.shape[2] + 1
    cases.verify(cartesian_case, concat_where_empty_branch_op, a, b, N, out=out, ref=a.asnumpy())


@pytest.mark.embedded_concat_where_infinite_domain
def test_concat_where_scalar_broadcast(cartesian_case, static_domains: bool):
    @gtx.field_operator(static_domains=static_domains)
    def concat_where_scalar_broadcast_op(a: np.int32, b: cases.IJKField, N: np.int32) -> cases.IJKField:
        return concat_where(KDim < N - 1, a, b)

    a = 3
    b = cases.allocate(cartesian_case, concat_where_scalar_broadcast_op, "b")()
    out = cases.allocate(cartesian_case, concat_where_scalar_broadcast_op, cases.RETURN)()

    ref = np.concatenate(
        (
            np.full((*out.domain.shape[0:2], out.domain.shape[2] - 1), a),
            b.asnumpy()[:, :, -1:],
        ),
        axis=2,
    )
    cases.verify(cartesian_case, concat_where_scalar_broadcast_op, a, b, cartesian_case.default_sizes[KDim], out=out, ref=ref)


@pytest.mark.embedded_concat_where_infinite_domain
def test_concat_where_scalar_broadcast_on_empty_branch(cartesian_case, static_domains: bool):
    """Output domain such that the scalar branch is never active."""

    @gtx.field_operator(static_domains=static_domains)
    def concat_where_scalar_broadcast_on_empty_branch_op(a: np.int32, b: cases.KField, N: np.int32) -> cases.KField:
        return concat_where(KDim < N, a, b)

    a = 3
    b = cases.allocate(cartesian_case, concat_where_scalar_broadcast_on_empty_branch_op, "b")()
    out = cases.allocate(cartesian_case, concat_where_scalar_broadcast_on_empty_branch_op, cases.RETURN, domain=b.domain.slice_at[1:])()

    ref = b.asnumpy()[1:]
    cases.verify(cartesian_case, concat_where_scalar_broadcast_on_empty_branch_op, a, b, 1, out=out, ref=ref)


def test_concat_where_single_level_broadcast(cartesian_case, static_domains: bool):
    @gtx.field_operator(static_domains=static_domains)
    def concat_where_single_level_broadcast_op(a: cases.KField, b: cases.IJKField) -> cases.IJKField:
        return concat_where(KDim == 0, a, b)

    out = cases.allocate(cartesian_case, concat_where_single_level_broadcast_op, cases.RETURN)()
    a = cases.allocate(
        cartesian_case, concat_where_single_level_broadcast_op, "a", domain=gtx.domain({KDim: out.domain.shape[2]})
    )()
    b = cases.allocate(cartesian_case, concat_where_single_level_broadcast_op, "b", domain=out.domain.slice_at[:, :, 1:])()

    ref = np.concatenate(
        (
            np.tile(a.asnumpy()[0], (*b.domain.shape[0:2], 1)),
            b.asnumpy(),
        ),
        axis=2,
    )
    cases.verify(cartesian_case, concat_where_single_level_broadcast_op, a, b, out=out, ref=ref)


def test_concat_where_single_level_restricted_domain_broadcast(
    cartesian_case, static_domains: bool
):
    @gtx.field_operator(static_domains=static_domains)
    def concat_where_single_level_restricted_domain_broadcast_op(a: cases.KField, b: cases.IJKField) -> cases.IJKField:
        return concat_where(KDim == 0, a, b)

    out = cases.allocate(cartesian_case, concat_where_single_level_restricted_domain_broadcast_op, cases.RETURN)()
    # note: this field is only defined on K: 0, 1, i.e., contains only a single value
    a = cases.allocate(cartesian_case, concat_where_single_level_restricted_domain_broadcast_op, "a", domain=gtx.domain({KDim: (0, 1)}))()
    b = cases.allocate(cartesian_case, concat_where_single_level_restricted_domain_broadcast_op, "b", domain=out.domain.slice_at[:, :, 1:])()

    ref = np.concatenate(
        (
            np.tile(a.asnumpy()[0], (*b.domain.shape[0:2], 1)),
            b.asnumpy(),
        ),
        axis=2,
    )
    cases.verify(cartesian_case, concat_where_single_level_restricted_domain_broadcast_op, a, b, out=out, ref=ref)


def test_boundary_single_layer_3d_bc(cartesian_case, static_domains: bool):
    @gtx.field_operator(static_domains=static_domains)
    def boundary_single_layer_3d_bc_op(interior: cases.IJKField, boundary: cases.IJKField) -> cases.IJKField:
        return concat_where(KDim == 0, boundary, interior)

    interior = cases.allocate(cartesian_case, boundary_single_layer_3d_bc_op, "interior")()
    boundary = cases.allocate(cartesian_case, boundary_single_layer_3d_bc_op, "boundary", sizes={KDim: 1})()
    out = cases.allocate(cartesian_case, boundary_single_layer_3d_bc_op, cases.RETURN)()

    k = np.arange(0, cartesian_case.default_sizes[KDim])
    ref = np.where(
        k[np.newaxis, np.newaxis, :] == 0,
        np.broadcast_to(boundary.asnumpy(), interior.shape),
        interior.asnumpy(),
    )

    cases.verify(cartesian_case, boundary_single_layer_3d_bc_op, interior, boundary, out=out, ref=ref)


def test_boundary_single_layer_2d_bc(cartesian_case, static_domains: bool):
    @gtx.field_operator(static_domains=static_domains)
    def boundary_single_layer_2d_bc_op(interior: cases.IJKField, boundary: cases.IJField) -> cases.IJKField:
        return concat_where(KDim == 0, boundary, interior)

    k = np.arange(0, cartesian_case.default_sizes[KDim])
    cases.verify_with_default_data(
        cartesian_case,
        boundary_single_layer_2d_bc_op,
        lambda interior, boundary: np.where(
            k[np.newaxis, np.newaxis, :] == 0, boundary[:, :, np.newaxis], interior
        ),
    )


def test_boundary_single_layer_2d_bc_on_empty_branch(cartesian_case, static_domains: bool):
    @gtx.field_operator(static_domains=static_domains)
    def boundary_single_layer_2d_bc_on_empty_branch_op(interior: cases.IJKField, boundary: cases.IJField) -> cases.IJKField:
        return concat_where(KDim == 0, boundary, interior)

    interior = cases.allocate(cartesian_case, boundary_single_layer_2d_bc_on_empty_branch_op, "interior")()
    boundary = cases.allocate(cartesian_case, boundary_single_layer_2d_bc_on_empty_branch_op, "boundary")()
    out = cases.allocate(
        cartesian_case, boundary_single_layer_2d_bc_on_empty_branch_op, cases.RETURN, domain=interior.domain.slice_at[:, :, 1:]
    )()

    ref = interior.asnumpy()[:, :, 1:]
    cases.verify(cartesian_case, boundary_single_layer_2d_bc_on_empty_branch_op, interior, boundary, out=out, ref=ref)


def test_dimension_two_nested_conditions(cartesian_case, static_domains: bool):
    @gtx.field_operator(static_domains=static_domains)
    def dimension_two_nested_conditions_op(interior: cases.IJKField, boundary: cases.IJKField) -> cases.IJKField:
        return concat_where((KDim < 2), boundary, concat_where((KDim >= 5), boundary, interior))

    k = np.arange(0, cartesian_case.default_sizes[KDim])
    cases.verify_with_default_data(
        cartesian_case,
        dimension_two_nested_conditions_op,
        lambda interior, boundary: np.where(
            (k[np.newaxis, np.newaxis, :] < 2) | (k[np.newaxis, np.newaxis, :] >= 5),
            boundary,
            interior,
        ),
    )


def test_dimension_two_conditions_and(cartesian_case, static_domains: bool):
    @gtx.field_operator(static_domains=static_domains)
    def dimension_two_conditions_and_op(interior: cases.KField, boundary: cases.KField, nlev: np.int32) -> cases.KField:
        return concat_where((0 < KDim) & (KDim < (nlev - 1)), interior, boundary)

    interior = cases.allocate(cartesian_case, dimension_two_conditions_and_op, "interior")()
    boundary = cases.allocate(cartesian_case, dimension_two_conditions_and_op, "boundary")()
    out = cases.allocate(cartesian_case, dimension_two_conditions_and_op, cases.RETURN)()

    nlev = cartesian_case.default_sizes[KDim]
    k = np.arange(0, nlev)
    ref = np.where((0 < k) & (k < (nlev - 1)), interior.asnumpy(), boundary.asnumpy())
    cases.verify(cartesian_case, dimension_two_conditions_and_op, interior, boundary, nlev, out=out, ref=ref)


def test_dimension_eq_in_middle_of_domain(cartesian_case, static_domains: bool):
    @gtx.field_operator(static_domains=static_domains)
    def dimension_eq_in_middle_of_domain_op(interior: cases.KField, boundary: cases.KField) -> cases.KField:
        return concat_where((KDim == 2), interior, boundary)

    k = np.arange(0, cartesian_case.default_sizes[KDim])
    cases.verify_with_default_data(
        cartesian_case, dimension_eq_in_middle_of_domain_op, lambda interior, boundary: np.where(k == 2, interior, boundary)
    )


@pytest.mark.embedded_concat_where_non_contiguous_domain
def test_dimension_not_eq_in_middle_of_domain(cartesian_case, static_domains: bool):
    @gtx.field_operator(static_domains=static_domains)
    def dimension_not_eq_in_middle_of_domain_op(interior: cases.KField, boundary: cases.KField) -> cases.KField:
        return concat_where((KDim != 2), boundary, interior)

    k = np.arange(0, cartesian_case.default_sizes[KDim])
    cases.verify_with_default_data(
        cartesian_case, dimension_not_eq_in_middle_of_domain_op, lambda interior, boundary: np.where(k != 2, boundary, interior)
    )


def test_dimension_less_equal(cartesian_case, static_domains: bool):
    @gtx.field_operator(static_domains=static_domains)
    def dimension_less_equal_op(interior: cases.KField, boundary: cases.KField) -> cases.KField:
        return concat_where((KDim <= 2), boundary, interior)

    k = np.arange(0, cartesian_case.default_sizes[KDim])
    cases.verify_with_default_data(
        cartesian_case, dimension_less_equal_op, lambda interior, boundary: np.where(k <= 2, boundary, interior)
    )


def test_dimension_reverse_greater(cartesian_case, static_domains: bool):
    @gtx.field_operator(static_domains=static_domains)
    def dimension_reverse_greater_op(interior: cases.KField, boundary: cases.KField) -> cases.KField:
        return concat_where((2 > KDim), boundary, interior)

    k = np.arange(0, cartesian_case.default_sizes[KDim])
    cases.verify_with_default_data(
        cartesian_case, dimension_reverse_greater_op, lambda interior, boundary: np.where(2 > k, boundary, interior)
    )


def test_dimension_reverse_greater_equal(cartesian_case, static_domains: bool):
    @gtx.field_operator(static_domains=static_domains)
    def dimension_reverse_greater_equal_op(interior: cases.KField, boundary: cases.KField) -> cases.KField:
        return concat_where((2 >= KDim), boundary, interior)

    k = np.arange(0, cartesian_case.default_sizes[KDim])
    cases.verify_with_default_data(
        cartesian_case, dimension_reverse_greater_equal_op, lambda interior, boundary: np.where(2 >= k, boundary, interior)
    )


def test_dimension_reverse_eq(cartesian_case, static_domains: bool):
    @gtx.field_operator(static_domains=static_domains)
    def dimension_reverse_eq_op(interior: cases.KField, boundary: cases.KField) -> cases.KField:
        return concat_where((2 == KDim), interior, boundary)

    k = np.arange(0, cartesian_case.default_sizes[KDim])
    cases.verify_with_default_data(
        cartesian_case, dimension_reverse_eq_op, lambda interior, boundary: np.where(2 == k, interior, boundary)
    )


@pytest.mark.embedded_concat_where_non_contiguous_domain
def test_dimension_reverse_not_eq(cartesian_case, static_domains: bool):
    @gtx.field_operator(static_domains=static_domains)
    def dimension_reverse_not_eq_op(interior: cases.KField, boundary: cases.KField) -> cases.KField:
        return concat_where((2 != KDim), boundary, interior)

    k = np.arange(0, cartesian_case.default_sizes[KDim])
    cases.verify_with_default_data(
        cartesian_case, dimension_reverse_not_eq_op, lambda interior, boundary: np.where(2 != k, boundary, interior)
    )


@pytest.mark.embedded_concat_where_non_contiguous_domain
def test_dimension_two_conditions_or(cartesian_case, static_domains: bool):
    @gtx.field_operator(static_domains=static_domains)
    def dimension_two_conditions_or_op(interior: cases.KField, boundary: cases.KField) -> cases.KField:
        return concat_where(((KDim < 2) | (KDim >= 5)), boundary, interior)

    k = np.arange(0, cartesian_case.default_sizes[KDim])
    cases.verify_with_default_data(
        cartesian_case,
        dimension_two_conditions_or_op,
        lambda interior, boundary: np.where((k < 2) | (k >= 5), boundary, interior),
    )


def test_lap_like(cartesian_case, static_domains: bool):
    @gtx.field_operator(static_domains=static_domains)
    def lap_like_op(
        inp: cases.IJField, boundary: np.int32, shape: tuple[np.int32, np.int32]
    ) -> cases.IJField:
        # TODO(havogt) add support for multi-dimensional concat_where and non-contiguous unions
        return concat_where(
            (IDim == 0),
            boundary,
            concat_where(
                IDim == shape[0] - 1,
                boundary,
                concat_where(
                    JDim == 0,
                    boundary,
                    concat_where(JDim == shape[1] - 1, boundary, inp),
                ),
            ),
        )

    out = cases.allocate(cartesian_case, lap_like_op, cases.RETURN)()
    inp = cases.allocate(cartesian_case, lap_like_op, "inp", domain=out.domain.slice_at[1:-1, 1:-1])()
    boundary = 2

    ref = np.full(out.domain.shape, np.nan)
    ref[0, :] = boundary
    ref[:, 0] = boundary
    ref[-1, :] = boundary
    ref[:, -1] = boundary
    ref[1:-1, 1:-1] = inp.asnumpy()
    cases.verify(cartesian_case, lap_like_op, inp, boundary, out.domain.shape, out=out, ref=ref)


@pytest.mark.uses_tuple_returns
def test_with_tuples(cartesian_case, static_domains: bool):
    @gtx.field_operator(static_domains=static_domains)
    def with_tuples_op(
        interior0: cases.IJKField,
        boundary0: cases.IJField,
        interior1: cases.IJKField,
        boundary1: cases.IJField,
    ) -> tuple[cases.IJKField, cases.IJKField]:
        return concat_where(KDim == 0, (boundary0, boundary1), (interior0, interior1))

    k = np.arange(0, cartesian_case.default_sizes[KDim])

    def ref(interior0, boundary0, interior1, boundary1):
        return (
            np.where(k[np.newaxis, np.newaxis, :] == 0, boundary0[:, :, np.newaxis], interior0),
            np.where(k[np.newaxis, np.newaxis, :] == 0, boundary1[:, :, np.newaxis], interior1),
        )

    cases.verify_with_default_data(cartesian_case, with_tuples_op, ref)


def test_nested_conditions_with_empty_branches(cartesian_case, static_domains: bool):
    @gtx.field_operator(static_domains=static_domains)
    def nested_conditions_with_empty_branches_op(interior: cases.IField, boundary: cases.IField, N: gtx.int32) -> cases.IField:
        interior = concat_where(IDim == 0, boundary, interior)
        interior = concat_where((1 <= IDim) & (IDim < N - 1), interior * 2, interior)
        interior = concat_where(IDim == N - 1, boundary, interior)
        return interior

    interior = cases.allocate(cartesian_case, nested_conditions_with_empty_branches_op, "interior")()
    boundary = cases.allocate(cartesian_case, nested_conditions_with_empty_branches_op, "boundary")()
    out = cases.allocate(cartesian_case, nested_conditions_with_empty_branches_op, cases.RETURN)()
    N = cartesian_case.default_sizes[IDim]

    i = np.arange(0, cartesian_case.default_sizes[IDim])
    ref = np.where(
        (i[:] == 0) | (i[:] == N - 1),
        boundary.asnumpy(),
        interior.asnumpy() * 2,
    )
    cases.verify(cartesian_case, nested_conditions_with_empty_branches_op, interior, boundary, N, out=out, ref=ref)


@pytest.mark.uses_tuple_returns
def test_with_tuples_different_domain(cartesian_case, static_domains: bool):
    @gtx.field_operator(static_domains=static_domains)
    def with_tuples_different_domain_op(
        interior0: cases.IJKField,
        boundary0: cases.IJKField,
        interior1: cases.KField,
        boundary1: cases.KField,
    ) -> tuple[cases.IJKField, cases.IJKField]:
        a, b = concat_where(KDim == 0, (boundary0, boundary1), (interior0, interior1))
        # the broadcast is only needed since we can not return fields on different domains yet
        return a, broadcast(b, (IDim, JDim, KDim))

    k = np.arange(0, cartesian_case.default_sizes[KDim])

    def ref(interior0, boundary0, interior1, boundary1):
        return (
            np.where(k[np.newaxis, np.newaxis, :] == 0, boundary0, interior0),
            np.where(k == 0, boundary1, interior1),
        )

    cases.verify_with_default_data(cartesian_case, with_tuples_different_domain_op, ref)


def test_concat_where_field_broadcast_on_empty_branch(cartesian_case, static_domains: bool):
    """
    A field branch with fewer dimensions than the expression is implicitly broadcast.

    `b` only has the `K` dimension, but is selected everywhere, so it is broadcast to the
    three-dimensional result. With `static_domains` this also tests pruning: the domain bounds
    are statically known, so `prune_empty_concat_where` decides that `a` is never selected and
    must reintroduce the implicit broadcast of the `concat_where` instead of replacing the
    three-dimensional expression by a one-dimensional one.
    """

    @gtx.field_operator(static_domains=static_domains)
    def concat_where_field_broadcast_on_empty_branch_op(a: cases.IJKField, b: cases.KField) -> cases.IJKField:
        return concat_where(KDim < 0, a, b)

    cases.verify_with_default_data(
        cartesian_case,
        concat_where_field_broadcast_on_empty_branch_op,
        lambda a, b: np.broadcast_to(b[np.newaxis, np.newaxis, :], a.shape),
    )
