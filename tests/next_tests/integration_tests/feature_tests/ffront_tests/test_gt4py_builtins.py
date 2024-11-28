# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import TypeAlias

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next import broadcast, common, float64, int32, max_over, min_over, neighbor_sum, where
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
    Koff,
    V2EDim,
    Vertex,
    cartesian_case,
    unstructured_case,
)
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
    mesh_descriptor,
)


@pytest.mark.uses_unstructured_shift
@pytest.mark.uses_max_over
@pytest.mark.parametrize(
    "strategy",
    [cases.UniqueInitializer(1), cases.UniqueInitializer(-100)],
    ids=["positive_values", "negative_values"],
)
def test_maxover_execution_(unstructured_case, strategy):
    @gtx.field_operator
    def testee(edge_f: cases.EField) -> cases.VField:
        out = max_over(edge_f(V2E), axis=V2EDim)
        return out

    inp = cases.allocate(unstructured_case, testee, "edge_f", strategy=strategy)()
    out = cases.allocate(unstructured_case, testee, cases.RETURN)()

    v2e_table = unstructured_case.offset_provider["V2E"].ndarray
    ref = np.max(
        inp.asnumpy()[v2e_table],
        axis=1,
        initial=np.min(inp.asnumpy()),
        where=v2e_table != common._DEFAULT_SKIP_VALUE,
    )
    cases.verify(unstructured_case, testee, inp, ref=ref, out=out)


@pytest.mark.uses_unstructured_shift
def test_minover_execution(unstructured_case):
    @gtx.field_operator
    def minover(edge_f: cases.EField) -> cases.VField:
        out = min_over(edge_f(V2E), axis=V2EDim)
        return out

    v2e_table = unstructured_case.offset_provider["V2E"].ndarray
    cases.verify_with_default_data(
        unstructured_case,
        minover,
        ref=lambda edge_f: np.min(
            edge_f[v2e_table],
            axis=1,
            initial=np.max(edge_f),
            where=v2e_table != common._DEFAULT_SKIP_VALUE,
        ),
    )


@gtx.field_operator
def reduction_e_field(edge_f: cases.EField) -> cases.VField:
    return neighbor_sum(edge_f(V2E), axis=V2EDim)


@gtx.field_operator
def reduction_ek_field(
    edge_f: common.Field[[Edge, KDim], np.int32],
) -> common.Field[[Vertex, KDim], np.int32]:
    return neighbor_sum(edge_f(V2E), axis=V2EDim)


@gtx.field_operator
def reduction_ke_field(
    edge_f: common.Field[[KDim, Edge], np.int32],
) -> common.Field[[KDim, Vertex], np.int32]:
    return neighbor_sum(edge_f(V2E), axis=V2EDim)


@pytest.mark.uses_unstructured_shift
@pytest.mark.parametrize(
    "fop", [reduction_e_field, reduction_ek_field, reduction_ke_field], ids=lambda fop: fop.__name__
)
def test_neighbor_sum(unstructured_case, fop):
    v2e_table = unstructured_case.offset_provider["V2E"].ndarray

    edge_f = cases.allocate(unstructured_case, fop, "edge_f")()

    local_dim_idx = edge_f.domain.dims.index(Edge) + 1
    adv_indexing = tuple(
        slice(None) if dim is not Edge else v2e_table for dim in edge_f.domain.dims
    )

    broadcast_slice = []
    for dim in edge_f.domain.dims:
        if dim is Edge:
            broadcast_slice.append(slice(None))
            broadcast_slice.append(slice(None))
        else:
            broadcast_slice.append(None)

    broadcasted_table = v2e_table[tuple(broadcast_slice)]
    ref = np.sum(
        edge_f.asnumpy()[adv_indexing],
        axis=local_dim_idx,
        initial=0,
        where=broadcasted_table != common._DEFAULT_SKIP_VALUE,
    )
    cases.verify(
        unstructured_case,
        fop,
        edge_f,
        out=cases.allocate(unstructured_case, fop, cases.RETURN)(),
        ref=ref,
    )


@pytest.mark.uses_unstructured_shift
def test_reduction_execution_with_offset(unstructured_case):
    EKField: TypeAlias = gtx.Field[[Edge, KDim], np.int32]
    VKField: TypeAlias = gtx.Field[[Vertex, KDim], np.int32]

    @gtx.field_operator
    def reduction(edge_f: EKField) -> VKField:
        return neighbor_sum(edge_f(V2E), axis=V2EDim)

    @gtx.field_operator
    def fencil_op(edge_f: EKField) -> VKField:
        red = reduction(edge_f)
        return red(Koff[1])

    @gtx.program
    def fencil(edge_f: EKField, out: VKField):
        fencil_op(edge_f, out=out)

    v2e_table = unstructured_case.offset_provider["V2E"].ndarray
    field = cases.allocate(unstructured_case, fencil, "edge_f", sizes={KDim: 2})()
    out = cases.allocate(unstructured_case, fencil_op, cases.RETURN, sizes={KDim: 1})()

    cases.verify(
        unstructured_case,
        fencil,
        field,
        out,
        inout=out,
        ref=np.sum(
            field.asnumpy()[:, 1][v2e_table],
            axis=1,
            initial=0,
            where=v2e_table != common._DEFAULT_SKIP_VALUE,
        ).reshape(out.shape),
        offset_provider=unstructured_case.offset_provider | {"Koff": KDim},
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

    v2e_table = unstructured_case.offset_provider["V2E"].ndarray
    cases.verify_with_default_data(
        unstructured_case,
        fencil,
        ref=lambda edge_f: 3
        * np.sum(
            -(edge_f[v2e_table] ** 2) * 2,
            axis=1,
            initial=0,
            where=v2e_table != common._DEFAULT_SKIP_VALUE,
        ),
    )


@pytest.mark.uses_unstructured_shift
def test_reduction_with_common_expression(unstructured_case):
    @gtx.field_operator
    def testee(flux: cases.EField) -> cases.VField:
        return neighbor_sum(flux(V2E) + flux(V2E), axis=V2EDim)

    v2e_table = unstructured_case.offset_provider["V2E"].ndarray
    cases.verify_with_default_data(
        unstructured_case,
        testee,
        ref=lambda flux: np.sum(
            flux[v2e_table] * 2, axis=1, initial=0, where=v2e_table != common._DEFAULT_SKIP_VALUE
        ),
    )


@pytest.mark.uses_unstructured_shift
def test_reduction_expression_with_where(unstructured_case):
    @gtx.field_operator
    def testee(mask: cases.VBoolField, inp: cases.EField) -> cases.VField:
        return neighbor_sum(where(mask, inp(V2E), inp(V2E)), axis=V2EDim)

    v2e_table = unstructured_case.offset_provider["V2E"].ndarray

    mask = unstructured_case.as_field(
        [Vertex], np.random.choice(a=[False, True], size=unstructured_case.default_sizes[Vertex])
    )
    inp = cases.allocate(unstructured_case, testee, "inp")()
    out = cases.allocate(unstructured_case, testee, cases.RETURN)()

    cases.verify(
        unstructured_case,
        testee,
        mask,
        inp,
        out=out,
        ref=np.sum(
            inp.asnumpy()[v2e_table],
            axis=1,
            initial=0,
            where=v2e_table != common._DEFAULT_SKIP_VALUE,
        ),
    )


@pytest.mark.uses_unstructured_shift
def test_reduction_expression_with_where_and_tuples(unstructured_case):
    @gtx.field_operator
    def testee(mask: cases.VBoolField, inp: cases.EField) -> cases.VField:
        return neighbor_sum(where(mask, (inp(V2E), inp(V2E)), (inp(V2E), inp(V2E)))[1], axis=V2EDim)

    v2e_table = unstructured_case.offset_provider["V2E"].ndarray

    mask = unstructured_case.as_field(
        [Vertex], np.random.choice(a=[False, True], size=unstructured_case.default_sizes[Vertex])
    )
    inp = cases.allocate(unstructured_case, testee, "inp")()
    out = cases.allocate(unstructured_case, testee, cases.RETURN)()

    cases.verify(
        unstructured_case,
        testee,
        mask,
        inp,
        out=out,
        ref=np.sum(
            inp.asnumpy()[v2e_table],
            axis=1,
            initial=0,
            where=v2e_table != common._DEFAULT_SKIP_VALUE,
        ),
    )


@pytest.mark.uses_unstructured_shift
def test_reduction_expression_with_where_and_scalar(unstructured_case):
    @gtx.field_operator
    def testee(mask: cases.VBoolField, inp: cases.EField) -> cases.VField:
        return neighbor_sum(inp(V2E) + where(mask, inp(V2E), 1), axis=V2EDim)

    v2e_table = unstructured_case.offset_provider["V2E"].ndarray

    mask = unstructured_case.as_field(
        [Vertex], np.random.choice(a=[False, True], size=unstructured_case.default_sizes[Vertex])
    )
    inp = cases.allocate(unstructured_case, testee, "inp")()
    out = cases.allocate(unstructured_case, testee, cases.RETURN)()

    cases.verify(
        unstructured_case,
        testee,
        mask,
        inp,
        out=out,
        ref=np.sum(
            inp.asnumpy()[v2e_table]
            + np.where(np.expand_dims(mask.asnumpy(), 1), inp.asnumpy()[v2e_table], 1),
            axis=1,
            initial=0,
            where=v2e_table != common._DEFAULT_SKIP_VALUE,
        ),
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


def test_promotion(unstructured_case):
    @gtx.field_operator
    def promotion(
        inp1: gtx.Field[[Edge, KDim], float64], inp2: gtx.Field[[KDim], float64]
    ) -> gtx.Field[[Edge, KDim], float64]:
        return inp1 / inp2

    cases.verify_with_default_data(unstructured_case, promotion, ref=lambda inp1, inp2: inp1 / inp2)
