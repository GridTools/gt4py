# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import functools
import numpy as np
from typing import Tuple
import pytest
from next_tests.integration_tests.cases import IDim, Ioff, JDim, KDim, Koff, cartesian_case
from gt4py import next as gtx
from gt4py.next import float64, int32
from gt4py.next.ffront.fbuiltins import where, broadcast
from next_tests.integration_tests import cases
from next_tests.integration_tests.cases_utils import (
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
        interior1: cases.IJKField,
        interior2: cases.IJKField,
        boundary0: cases.IJField,
        boundary1: cases.IJField,
        boundary2: cases.IJField,
    ) -> tuple[cases.IJKField, tuple[cases.IJKField, cases.IJKField]]:
        return where(
            broadcast(k, (IDim, JDim, KDim)) == 0,
            (boundary0, (boundary1, boundary2)),
            (interior0, (interior1, interior2)),
        )

    k = cases.allocate(cartesian_case, testee, "k", strategy=cases.IndexInitializer())()
    interiors = tuple(cases.allocate(cartesian_case, testee, f"interior{i}")() for i in range(3))
    boundaries = tuple(cases.allocate(cartesian_case, testee, f"boundary{i}")() for i in range(3))
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    refs = tuple(
        np.where(
            k.asnumpy()[np.newaxis, np.newaxis, :] == 0,
            boundary.asnumpy()[:, :, np.newaxis],
            interior.asnumpy(),
        )
        for boundary, interior in zip(boundaries, interiors)
    )

    cases.verify(
        cartesian_case,
        testee,
        k,
        *interiors,
        *boundaries,
        out=out,
        ref=(refs[0], (refs[1], refs[2])),
    )


@pytest.mark.uses_tuple_returns
def test_conditional_nested_tuple(cartesian_case):
    @gtx.field_operator
    def conditional_nested_tuple(
        mask: cases.IBoolField, a: cases.IFloatField, b: cases.IFloatField
    ) -> tuple[
        tuple[cases.IFloatField, cases.IFloatField], tuple[cases.IFloatField, cases.IFloatField]
    ]:
        return where(mask, ((a, b), (b, a)), ((5.0, 7.0), (7.0, 5.0)))

    size = cartesian_case.default_sizes[IDim]
    mask = cartesian_case.as_field([IDim], np.random.choice(a=[False, True], size=size))
    a = cases.allocate(cartesian_case, conditional_nested_tuple, "a")()
    b = cases.allocate(cartesian_case, conditional_nested_tuple, "b")()

    where_with_mask = functools.partial(np.where, mask.asnumpy())

    cases.verify(
        cartesian_case,
        conditional_nested_tuple,
        mask,
        a,
        b,
        out=cases.allocate(cartesian_case, conditional_nested_tuple, cases.RETURN)(),
        ref=(
            (
                where_with_mask(a.asnumpy(), np.full(size, 5.0)),
                where_with_mask(b.asnumpy(), np.full(size, 7.0)),
            ),
            (
                where_with_mask(b.asnumpy(), np.full(size, 7.0)),
                where_with_mask(a.asnumpy(), np.full(size, 5.0)),
            ),
        ),
    )


def test_conditional(cartesian_case):
    @gtx.field_operator
    def conditional(
        mask: cases.IBoolField, a: cases.IFloatField, b: cases.IFloatField
    ) -> cases.IFloatField:
        return where(mask, a, b)

    size = cartesian_case.default_sizes[IDim]
    mask = cartesian_case.as_field([IDim], np.random.choice(a=[False, True], size=(size)))
    a = cases.allocate(cartesian_case, conditional, "a")()
    b = cases.allocate(cartesian_case, conditional, "b")()
    out = cases.allocate(cartesian_case, conditional, cases.RETURN)()

    cases.verify(
        cartesian_case,
        conditional,
        mask,
        a,
        b,
        out=out,
        ref=np.where(mask.asnumpy(), a.asnumpy(), b.asnumpy()),
    )


def test_conditional_promotion(cartesian_case):
    @gtx.field_operator
    def conditional_promotion(mask: cases.IBoolField, a: cases.IFloatField) -> cases.IFloatField:
        return where(mask, a, 10.0)

    size = cartesian_case.default_sizes[IDim]
    mask = cartesian_case.as_field([IDim], np.random.choice(a=[False, True], size=(size)))
    a = cases.allocate(cartesian_case, conditional_promotion, "a")()
    out = cases.allocate(cartesian_case, conditional_promotion, cases.RETURN)()
    ref = np.where(mask.asnumpy(), a.asnumpy(), 10.0)

    cases.verify(cartesian_case, conditional_promotion, mask, a, out=out, ref=ref)


def test_conditional_compareop(cartesian_case):
    @gtx.field_operator
    def conditional_promotion(a: cases.IFloatField) -> cases.IFloatField:
        return where(a != a, a, 10.0)

    cases.verify_with_default_data(
        cartesian_case, conditional_promotion, ref=lambda a: np.where(a != a, a, 10.0)
    )


@pytest.mark.uses_cartesian_shift
def test_conditional_shifted(cartesian_case):
    @gtx.field_operator
    def conditional_shifted(
        mask: cases.IBoolField, a: cases.IFloatField, b: cases.IFloatField
    ) -> gtx.Field[[IDim], float64]:
        tmp = where(mask, a, b)
        return tmp(Ioff[1])

    @gtx.program
    def conditional_program(
        mask: cases.IBoolField, a: cases.IFloatField, b: cases.IFloatField, out: cases.IFloatField
    ):
        conditional_shifted(mask, a, b, out=out)

    size = cartesian_case.default_sizes[IDim] + 1
    mask = cartesian_case.as_field([IDim], np.random.choice(a=[False, True], size=(size)))
    a = cases.allocate(cartesian_case, conditional_program, "a").extend({IDim: (0, 1)})()
    b = cases.allocate(cartesian_case, conditional_program, "b").extend({IDim: (0, 1)})()
    out = cases.allocate(cartesian_case, conditional_shifted, cases.RETURN)()

    cases.verify(
        cartesian_case,
        conditional_program,
        mask,
        a,
        b,
        out,
        inout=out,
        ref=np.where(mask.asnumpy(), a.asnumpy(), b.asnumpy())[1:],
    )
