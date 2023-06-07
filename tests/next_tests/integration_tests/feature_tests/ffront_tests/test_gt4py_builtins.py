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
from gt4py.next import broadcast, float64, int64, max_over, min_over, neighbor_sum, where
from gt4py.next.program_processors.runners import gtfn_cpu

from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    Edge,
    IDim,
    Ioff,
    JDim,
    Joff,
    KDim,
    Vertex,
    fieldview_backend,
    reduction_setup,
    size,
)


def test_maxover_execution(reduction_setup, fieldview_backend):
    """Testing max_over functionality."""
    if fieldview_backend in [gtfn_cpu.run_gtfn or fieldview_backend, gtfn_cpu.run_gtfn_imperative]:
        pytest.skip("not yet supported.")

    rs = reduction_setup
    V2EDim = rs.V2EDim
    inp_field = gtx.np_as_located_field(Vertex, V2EDim)(rs.v2e_table)

    @gtx.field_operator(backend=fieldview_backend)
    def maxover_fieldoperator(
        inp_field: gtx.Field[[Vertex, V2EDim], int64]
    ) -> gtx.Field[[Vertex], int64]:
        return max_over(inp_field, axis=V2EDim)

    maxover_fieldoperator(inp_field, out=rs.out, offset_provider=rs.offset_provider)

    ref = np.max(rs.v2e_table, axis=1)
    assert np.allclose(ref, rs.out)


def test_maxover_execution_negatives(reduction_setup, fieldview_backend):
    """Testing max_over functionality for negative values in array."""
    if fieldview_backend in [gtfn_cpu.run_gtfn, gtfn_cpu.run_gtfn_imperative]:
        pytest.skip("not yet supported.")

    rs = reduction_setup
    V2EDim, V2E = rs.V2EDim, rs.V2E
    edge_num = np.max(rs.v2e_table)
    inp_field_arr = np.arange(-edge_num // 2, edge_num // 2 + 1, 1, dtype=int)
    inp_field = gtx.np_as_located_field(Edge)(inp_field_arr)

    @gtx.field_operator(backend=fieldview_backend)
    def maxover_negvals(edge_f: gtx.Field[[Edge], int64]) -> gtx.Field[[Vertex], int64]:
        out = max_over(edge_f(V2E), axis=V2EDim)
        return out

    maxover_negvals(inp_field, out=rs.out, offset_provider=rs.offset_provider)

    ref = np.max(inp_field_arr[rs.v2e_table], axis=1)
    assert np.allclose(ref, rs.out)


def test_minover_execution(reduction_setup, fieldview_backend):
    """Testing the min_over functionality"""
    if fieldview_backend in [gtfn_cpu.run_gtfn, gtfn_cpu.run_gtfn_imperative]:
        pytest.skip("not implemented yet")

    rs = reduction_setup
    V2EDim = rs.V2EDim
    in_field = gtx.np_as_located_field(Vertex, V2EDim)(rs.v2e_table)

    @gtx.field_operator(backend=fieldview_backend)
    def minover_fieldoperator(
        input: gtx.Field[[Vertex, V2EDim], int64]
    ) -> gtx.Field[[Vertex], int64]:
        return min_over(input, axis=V2EDim)

    minover_fieldoperator(in_field, out=rs.out, offset_provider=rs.offset_provider)

    ref = np.min(rs.v2e_table, axis=1)
    assert np.allclose(ref, rs.out)


def test_minover_execution_float(reduction_setup, fieldview_backend):
    """Testing the min_over functionality"""
    if fieldview_backend in [gtfn_cpu.run_gtfn, gtfn_cpu.run_gtfn_imperative]:
        pytest.skip("not implemented yet")

    rs = reduction_setup
    V2EDim = rs.V2EDim
    in_array = np.random.default_rng().uniform(low=-1, high=1, size=rs.v2e_table.shape)
    in_field = gtx.np_as_located_field(Vertex, V2EDim)(in_array)
    out_field = gtx.np_as_located_field(Vertex)(np.zeros(rs.num_vertices))

    @gtx.field_operator(backend=fieldview_backend)
    def minover_fieldoperator(
        input: gtx.Field[[Vertex, V2EDim], float64]
    ) -> gtx.Field[[Vertex], float64]:
        return min_over(input, axis=V2EDim)

    minover_fieldoperator(in_field, out=out_field, offset_provider=rs.offset_provider)

    ref = np.min(in_array, axis=1)
    assert np.allclose(ref, out_field)


def test_reduction_execution(reduction_setup, fieldview_backend):
    """Testing a trivial neighbor sum."""
    rs = reduction_setup
    V2EDim, V2E = rs.V2EDim, rs.V2E

    @gtx.field_operator(backend=fieldview_backend)
    def reduction(edge_f: gtx.Field[[Edge], int64]) -> gtx.Field[[Vertex], int64]:
        return neighbor_sum(edge_f(V2E), axis=V2EDim)

    @gtx.program(backend=fieldview_backend)
    def fencil(edge_f: gtx.Field[[Edge], int64], out: gtx.Field[[Vertex], int64]) -> None:
        reduction(edge_f, out=out)

    fencil(rs.inp, rs.out, offset_provider=rs.offset_provider)

    ref = np.sum(rs.v2e_table, axis=1)
    assert np.allclose(ref, rs.out)


def test_reduction_expression(reduction_setup, fieldview_backend):
    """Test reduction with an expression directly inside the call."""
    if fieldview_backend in [gtfn_cpu.run_gtfn, gtfn_cpu.run_gtfn_imperative]:
        pytest.skip("Has a bug.")

    rs = reduction_setup
    V2EDim, V2E = rs.V2EDim, rs.V2E

    @gtx.field_operator(backend=fieldview_backend)
    def reduce_expr(edge_f: gtx.Field[[Edge], int64]) -> gtx.Field[[Vertex], int64]:
        tmp_nbh_tup = edge_f(V2E), edge_f(V2E)
        tmp_nbh = tmp_nbh_tup[0]
        return 3 * neighbor_sum(-edge_f(V2E) * tmp_nbh * 2, axis=V2EDim)

    @gtx.program(backend=fieldview_backend)
    def fencil(edge_f: gtx.Field[[Edge], int64], out: gtx.Field[[Vertex], int64]) -> None:
        reduce_expr(edge_f, out=out)

    fencil(rs.inp, rs.out, offset_provider=rs.offset_provider)

    ref = 3 * np.sum(-(rs.v2e_table**2) * 2, axis=1)
    assert np.allclose(ref, rs.out.array())


def test_reduction_with_common_expression(reduction_setup, fieldview_backend):
    rs = reduction_setup
    V2EDim, V2E = rs.V2EDim, rs.V2E

    @gtx.field_operator(backend=fieldview_backend)
    def testee(flux: gtx.Field[[Edge], int64]) -> gtx.Field[[Vertex], int64]:
        return neighbor_sum(flux(V2E) + flux(V2E), axis=V2EDim)

    testee(rs.inp, out=rs.out, offset_provider=rs.offset_provider)

    ref = np.sum(rs.v2e_table * 2, axis=1)
    assert np.allclose(ref, rs.out.array())


def test_conditional_nested_tuple(fieldview_backend):
    a_I_float = gtx.np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    b_I_float = gtx.np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    out_I_float = gtx.np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    out_I_float_1 = gtx.np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    mask = gtx.np_as_located_field(IDim)(np.zeros((size,), dtype=bool))

    @gtx.field_operator(backend=fieldview_backend)
    def conditional_nested_tuple(
        mask: gtx.Field[[IDim], bool], a: gtx.Field[[IDim], float64], b: gtx.Field[[IDim], float64]
    ) -> tuple[
        tuple[gtx.Field[[IDim], float64], gtx.Field[[IDim], float64]],
        tuple[gtx.Field[[IDim], float64], gtx.Field[[IDim], float64]],
    ]:
        return where(mask, ((a, b), (b, a)), ((5.0, 7.0), (7.0, 5.0)))

    conditional_nested_tuple(
        mask,
        a_I_float,
        b_I_float,
        out=((out_I_float, out_I_float_1), (out_I_float_1, out_I_float)),
        offset_provider={},
    )

    expected = np.where(
        mask,
        ((a_I_float, b_I_float), (b_I_float, a_I_float)),
        ((np.full(size, 5.0), np.full(size, 7.0)), (np.full(size, 7.0), np.full(size, 5.0))),
    )

    assert np.allclose(expected, ((out_I_float, out_I_float_1), (out_I_float_1, out_I_float)))


def test_broadcast_simple(fieldview_backend):
    a_I_int = gtx.np_as_located_field(IDim)(np.random.randn(size).astype("int64"))
    out_IJ_int = gtx.np_as_located_field(IDim, JDim)(np.zeros((size, size), dtype=int64))

    @gtx.field_operator(backend=fieldview_backend)
    def simple_broadcast(inp: gtx.Field[[IDim], int64]) -> gtx.Field[[IDim, JDim], int64]:
        return broadcast(inp, (IDim, JDim))

    simple_broadcast(a_I_int, out=out_IJ_int, offset_provider={})

    assert np.allclose(a_I_int.array()[:, np.newaxis], out_IJ_int)


def test_broadcast_scalar(fieldview_backend):
    out_I_float = gtx.np_as_located_field(IDim)(np.random.randn(size).astype("float64"))

    @gtx.field_operator(backend=fieldview_backend)
    def scalar_broadcast() -> gtx.Field[[IDim], float64]:
        return broadcast(float(1.0), (IDim,))

    scalar_broadcast(out=out_I_float, offset_provider={})

    assert np.allclose(1, out_I_float)


def test_broadcast_two_fields(fieldview_backend):
    a_I_int = gtx.np_as_located_field(IDim)(np.random.randn(size).astype("int64"))
    b_J_int = gtx.np_as_located_field(JDim)(np.random.randn(size).astype("int64"))
    out_IJ_int = gtx.np_as_located_field(IDim, JDim)(np.zeros((size, size), dtype=int64))

    @gtx.field_operator(backend=fieldview_backend)
    def broadcast_two_fields(
        inp1: gtx.Field[[IDim], int64], inp2: gtx.Field[[JDim], int64]
    ) -> gtx.Field[[IDim, JDim], int64]:
        a = broadcast(inp1, (IDim, JDim))
        b = broadcast(inp2, (IDim, JDim))
        return a + b

    broadcast_two_fields(a_I_int, b_J_int, out=out_IJ_int, offset_provider={})

    expected = a_I_int.array()[:, np.newaxis] + b_J_int.array()[np.newaxis, :]

    assert np.allclose(expected, out_IJ_int)


def test_broadcast_shifted(fieldview_backend):
    a_I_int = gtx.np_as_located_field(IDim)(np.random.randn(size).astype("int64"))
    out_IJ_int = gtx.np_as_located_field(IDim, JDim)(np.zeros((size, size), dtype=int64))

    @gtx.field_operator(backend=fieldview_backend)
    def simple_broadcast(inp: gtx.Field[[IDim], int64]) -> gtx.Field[[IDim, JDim], int64]:
        bcasted = broadcast(inp, (IDim, JDim))
        return bcasted(Joff[1])

    simple_broadcast(a_I_int, out=out_IJ_int, offset_provider={"Joff": JDim})

    assert np.allclose(a_I_int.array()[:, np.newaxis], out_IJ_int)


def test_conditional(fieldview_backend):
    a = gtx.np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    b = gtx.np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    out = gtx.np_as_located_field(IDim)(np.zeros((size,), dtype=np.float64))
    mask = gtx.np_as_located_field(IDim)(np.random.randn(size) > 0)

    @gtx.field_operator(backend=fieldview_backend)
    def conditional(
        mask: gtx.Field[[IDim], bool], a: gtx.Field[[IDim], float64], b: gtx.Field[[IDim], float64]
    ) -> gtx.Field[[IDim], float64]:
        return where(mask, a, b)

    conditional(mask, a, b, out=out, offset_provider={})

    assert np.allclose(np.where(mask, a, b), out)


def test_conditional_promotion(fieldview_backend):
    a_I_float = gtx.np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    out_I_float = gtx.np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    mask = gtx.np_as_located_field(IDim)(np.zeros((size,), dtype=bool))

    @gtx.field_operator(backend=fieldview_backend)
    def conditional_promotion(
        mask: gtx.Field[[IDim], bool], a: gtx.Field[[IDim], float64]
    ) -> gtx.Field[[IDim], float64]:
        return where(mask, a, 10.0)

    conditional_promotion(mask, a_I_float, out=out_I_float, offset_provider={})

    assert np.allclose(np.where(mask, a_I_float, 10), out_I_float)


def test_conditional_compareop(fieldview_backend):
    a_I_float = gtx.np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    out_I_float = gtx.np_as_located_field(IDim)(np.random.randn(size).astype("float64"))

    @gtx.field_operator(backend=fieldview_backend)
    def conditional_promotion(a: gtx.Field[[IDim], float64]) -> gtx.Field[[IDim], float64]:
        return where(a != a, a, 10.0)

    conditional_promotion(a_I_float, out=out_I_float, offset_provider={})

    assert np.allclose(
        np.where(np.asarray(a_I_float) != np.asarray(a_I_float), a_I_float, 10), out_I_float
    )


def test_conditional_shifted(fieldview_backend):
    a_I_float = gtx.np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    b_I_float = gtx.np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    out_I_float = gtx.np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    mask = gtx.np_as_located_field(IDim)(np.zeros((size,), dtype=bool))

    @gtx.field_operator(backend=fieldview_backend)
    def conditional_shifted(
        mask: gtx.Field[[IDim], bool], a: gtx.Field[[IDim], float64], b: gtx.Field[[IDim], float64]
    ) -> gtx.Field[[IDim], float64]:
        tmp = where(mask, a, b)
        return tmp(Ioff[1])

    @gtx.program(backend=fieldview_backend)
    def conditional_program(
        mask: gtx.Field[[IDim], bool],
        a: gtx.Field[[IDim], float64],
        b: gtx.Field[[IDim], float64],
        out: gtx.Field[[IDim], float64],
    ):
        conditional_shifted(mask, a, b, out=out[:-1])

    conditional_program(mask, a_I_float, b_I_float, out_I_float, offset_provider={"Ioff": IDim})

    assert np.allclose(np.where(mask, a_I_float, b_I_float)[1:], out_I_float.array()[:-1])


def test_promotion(fieldview_backend):
    ksize = 5
    a = gtx.np_as_located_field(Edge, KDim)(np.ones((size, ksize)))
    b = gtx.np_as_located_field(KDim)(np.ones((ksize)) * 2)
    c = gtx.np_as_located_field(Edge, KDim)(np.zeros((size, ksize)))

    @gtx.field_operator(backend=fieldview_backend)
    def promotion(
        inp1: gtx.Field[[Edge, KDim], float64], inp2: gtx.Field[[KDim], float64]
    ) -> gtx.Field[[Edge, KDim], float64]:
        return inp1 / inp2

    promotion(a, b, out=c, offset_provider={})

    assert np.allclose((a.array() / b.array()), c)
