# -*- coding: utf-8 -*-
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

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next import broadcast, float64, int32, max_over, min_over, neighbor_sum, where
from gt4py.next.program_processors.runners import gtfn

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import (
    V2E,
    Edge,
    IDim,
    Ioff,
    JDim,
    Joff,
    KDim,
    V2EDim,
    cartesian_case,
    unstructured_case,
)
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    fieldview_backend,
    reduction_setup,
)


@pytest.mark.uses_unstructured_shift
@pytest.mark.parametrize(
    "strategy",
    [cases.UniqueInitializer(1), cases.UniqueInitializer(-100)],
    ids=["positive_values", "negative_values"],
)
def test_maxover_execution_(unstructured_case, strategy):
    if unstructured_case.backend in [
        gtfn.run_gtfn,
        gtfn.run_gtfn_gpu,
        gtfn.run_gtfn_imperative,
        gtfn.run_gtfn_with_temporaries,
    ]:
        pytest.xfail("`maxover` broken in gtfn, see #1289.")

    @gtx.field_operator
    def testee(edge_f: cases.EField) -> cases.VField:
        out = max_over(edge_f(V2E), axis=V2EDim)
        return out

    inp = cases.allocate(unstructured_case, testee, "edge_f", strategy=strategy)()
    out = cases.allocate(unstructured_case, testee, cases.RETURN)()

    v2e_table = unstructured_case.offset_provider["V2E"].table
    ref = np.max(inp.ndarray[v2e_table], axis=1)
    cases.verify(unstructured_case, testee, inp, ref=ref, out=out)


@pytest.mark.uses_unstructured_shift
def test_minover_execution(unstructured_case):
    @gtx.field_operator
    def minover(edge_f: cases.EField) -> cases.VField:
        out = min_over(edge_f(V2E), axis=V2EDim)
        return out

    v2e_table = unstructured_case.offset_provider["V2E"].table
    cases.verify_with_default_data(
        unstructured_case, minover, ref=lambda edge_f: np.min(edge_f[v2e_table], axis=1)
    )


@pytest.mark.uses_unstructured_shift
def test_reduction_execution(unstructured_case):
    @gtx.field_operator
    def reduction(edge_f: cases.EField) -> cases.VField:
        return neighbor_sum(edge_f(V2E), axis=V2EDim)

    @gtx.program
    def fencil(edge_f: cases.EField, out: cases.VField):
        reduction(edge_f, out=out)

    cases.verify_with_default_data(
        unstructured_case,
        fencil,
        ref=lambda edge_f: np.sum(edge_f[unstructured_case.offset_provider["V2E"].table], axis=1),
    )


@pytest.mark.uses_unstructured_shift
@pytest.mark.uses_constant_fields
def test_reduction_expression_in_call(unstructured_case):
    @gtx.field_operator
    def reduce_expr(edge_f: cases.EField) -> cases.VField:
        tmp_nbh_tup = edge_f(V2E), edge_f(V2E)
        tmp_nbh = tmp_nbh_tup[0]
        return 3 * neighbor_sum(-edge_f(V2E) * tmp_nbh * 2, axis=V2EDim)

    @gtx.program
    def fencil(edge_f: cases.EField, out: cases.VField):
        reduce_expr(edge_f, out=out)

    cases.verify_with_default_data(
        unstructured_case,
        fencil,
        ref=lambda edge_f: 3
        * np.sum(-edge_f[unstructured_case.offset_provider["V2E"].table] ** 2 * 2, axis=1),
    )


@pytest.mark.uses_unstructured_shift
def test_reduction_with_common_expression(unstructured_case):
    # TODO(edopao): remove try/catch after uplift of dace module to version > 0.15
    try:
        from gt4py.next.program_processors.runners.dace_iterator import run_dace_gpu

        if unstructured_case.backend == run_dace_gpu:
            # see https://github.com/spcl/dace/pull/1442
            pytest.xfail("requires fix in dace module for cuda codegen")
    except ImportError:
        pass

    @gtx.field_operator
    def testee(flux: cases.EField) -> cases.VField:
        return neighbor_sum(flux(V2E) + flux(V2E), axis=V2EDim)

    cases.verify_with_default_data(
        unstructured_case,
        testee,
        ref=lambda flux: np.sum(flux[unstructured_case.offset_provider["V2E"].table] * 2, axis=1),
    )


@pytest.mark.uses_tuple_returns
def test_conditional_nested_tuple(cartesian_case):
    @gtx.field_operator
    def conditional_nested_tuple(
        mask: cases.IBoolField, a: cases.IFloatField, b: cases.IFloatField
    ) -> tuple[
        tuple[cases.IFloatField, cases.IFloatField],
        tuple[cases.IFloatField, cases.IFloatField],
    ]:
        return where(mask, ((a, b), (b, a)), ((5.0, 7.0), (7.0, 5.0)))

    size = cartesian_case.default_sizes[IDim]
    mask = cartesian_case.as_field([IDim], np.random.choice(a=[False, True], size=size))
    a = cases.allocate(cartesian_case, conditional_nested_tuple, "a")()
    b = cases.allocate(cartesian_case, conditional_nested_tuple, "b")()

    cases.verify(
        cartesian_case,
        conditional_nested_tuple,
        mask,
        a,
        b,
        out=cases.allocate(cartesian_case, conditional_nested_tuple, cases.RETURN)(),
        ref=np.where(
            mask.asnumpy(),
            ((a.asnumpy(), b.asnumpy()), (b.asnumpy(), a.asnumpy())),
            ((np.full(size, 5.0), np.full(size, 7.0)), (np.full(size, 7.0), np.full(size, 5.0))),
        ),
    )


def test_broadcast_simple(cartesian_case):
    @gtx.field_operator
    def simple_broadcast(inp: cases.IField) -> cases.IJField:
        return broadcast(inp, (IDim, JDim))

    cases.verify_with_default_data(
        cartesian_case, simple_broadcast, ref=lambda inp: inp[:, np.newaxis]
    )


def test_broadcast_scalar(cartesian_case):
    size = cartesian_case.default_sizes[IDim]

    @gtx.field_operator
    def scalar_broadcast() -> gtx.Field[[IDim], float64]:
        return broadcast(float(1.0), (IDim,))

    cases.verify_with_default_data(cartesian_case, scalar_broadcast, ref=lambda: np.ones(size))


def test_broadcast_two_fields(cartesian_case):
    @gtx.field_operator
    def broadcast_two_fields(inp1: cases.IField, inp2: gtx.Field[[JDim], int32]) -> cases.IJField:
        a = broadcast(inp1, (IDim, JDim))
        b = broadcast(inp2, (IDim, JDim))
        return a + b

    cases.verify_with_default_data(
        cartesian_case, broadcast_two_fields, ref=lambda a, b: a[:, np.newaxis] + b[np.newaxis, :]
    )


@pytest.mark.uses_cartesian_shift
def test_broadcast_shifted(cartesian_case):
    @gtx.field_operator
    def simple_broadcast(inp: cases.IField) -> cases.IJField:
        bcasted = broadcast(inp, (IDim, JDim))
        return bcasted(Joff[1])

    cases.verify_with_default_data(
        cartesian_case, simple_broadcast, ref=lambda inp: inp[:, np.newaxis]
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
        mask: cases.IBoolField,
        a: cases.IFloatField,
        b: cases.IFloatField,
        out: cases.IFloatField,
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


def test_promotion(unstructured_case):
    @gtx.field_operator
    def promotion(
        inp1: gtx.Field[[Edge, KDim], float64], inp2: gtx.Field[[KDim], float64]
    ) -> gtx.Field[[Edge, KDim], float64]:
        return inp1 / inp2

    cases.verify_with_default_data(unstructured_case, promotion, ref=lambda inp1, inp2: inp1 / inp2)
