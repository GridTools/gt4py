# -*- coding: utf-8 -*-
# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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

#
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import (
    Field,
    broadcast,
    float64,
    int32,
    int64,
    max_over,
    min_over,
    minimum,
    neighbor_sum,
    where,
)

from .ffront_test_utils import *


def test_maxover_execution_sparse(reduction_setup, fieldview_backend):
    """Testing max_over functionality."""
    rs = reduction_setup
    Vertex = rs.Vertex
    V2EDim = rs.V2EDim

    inp_field = np_as_located_field(Vertex, V2EDim)(rs.v2e_table.astype(np.int64))

    @field_operator(backend=fieldview_backend)
    def maxover_fieldoperator(inp_field: Field[[Vertex, V2EDim], int64]) -> Field[[Vertex], int64]:
        return max_over(inp_field, axis=V2EDim)

    maxover_fieldoperator(inp_field, out=rs.out, offset_provider=rs.offset_provider)

    ref = np.max(rs.v2e_table, axis=1)
    assert np.allclose(ref, rs.out)


def test_maxover_execution_negatives(reduction_setup, fieldview_backend):
    """Testing max_over functionality for negative values in array."""
    rs = reduction_setup
    Edge = rs.Edge
    Vertex = rs.Vertex
    V2EDim = rs.V2EDim
    V2E = rs.V2E

    edge_num = np.max(rs.v2e_table)
    inp_field_arr = np.arange(-edge_num // 2, edge_num // 2 + 1, 1, dtype=int)
    inp_field = np_as_located_field(Edge)(inp_field_arr)

    @field_operator(backend=fieldview_backend)
    def maxover_negvals(
        edge_f: Field[[Edge], int64],
    ) -> Field[[Vertex], int64]:
        out = max_over(edge_f(V2E), axis=V2EDim)
        return out

    maxover_negvals(inp_field, out=rs.out, offset_provider=rs.offset_provider)

    ref = np.max(inp_field_arr[rs.v2e_table], axis=1)
    assert np.allclose(ref, rs.out)


def test_minover_execution(reduction_setup, fieldview_backend):
    """Testing the min_over functionality"""
    rs = reduction_setup
    Vertex = rs.Vertex
    V2EDim = rs.V2EDim

    in_field = np_as_located_field(Vertex, V2EDim)(rs.v2e_table.astype(np.int32))
    out_field = np_as_located_field(Vertex)(np.zeros(rs.num_vertices, dtype=np.int32))

    @field_operator
    def minover_fieldoperator(input: Field[[Vertex, V2EDim], int32]) -> Field[[Vertex], int32]:
        return min_over(input, axis=V2EDim)

    minover_fieldoperator(in_field, out=out_field, offset_provider=rs.offset_provider)

    ref = np.min(rs.v2e_table, axis=1)
    assert np.allclose(ref, out_field)


def test_minover_execution_float(reduction_setup, fieldview_backend):
    """Testing the min_over functionality"""
    rs = reduction_setup
    Vertex = rs.Vertex
    V2EDim = rs.V2EDim

    in_array = np.random.default_rng().uniform(low=-1, high=1, size=rs.v2e_table.shape)
    in_field = np_as_located_field(Vertex, V2EDim)(in_array)
    out_field = np_as_located_field(Vertex)(np.zeros(rs.num_vertices))

    @field_operator
    def minover_fieldoperator(input: Field[[Vertex, V2EDim], float64]) -> Field[[Vertex], float64]:
        return min_over(input, axis=V2EDim)

    minover_fieldoperator(in_field, out=out_field, offset_provider=rs.offset_provider)

    ref = np.min(in_array, axis=1)
    assert np.allclose(ref, out_field)


def test_reduction_execution(reduction_setup, fieldview_backend):
    """Testing a trivial neighbor sum."""
    rs = reduction_setup
    Edge = rs.Edge
    Vertex = rs.Vertex
    V2EDim = rs.V2EDim
    V2E = rs.V2E

    @field_operator
    def reduction(edge_f: Field[[Edge], int64]) -> Field[[Vertex], int64]:
        return neighbor_sum(edge_f(V2E) + 1, axis=V2EDim)

    @program(backend=fieldview_backend)
    def fencil(edge_f: Field[[Edge], int64], out: Field[[Vertex], int64]) -> None:
        reduction(edge_f, out=out)

    fencil(rs.inp, rs.out, offset_provider=rs.offset_provider)

    ref = np.sum(rs.v2e_table + np.where(rs.v2e_table != -1, 1, 0), axis=1)
    assert np.allclose(ref, rs.out)


def test_reduction_execution_nb(reduction_setup, fieldview_backend):
    """Testing a neighbor sum on a neighbor field."""
    rs = reduction_setup
    V2EDim = rs.V2EDim

    nb_field = np_as_located_field(rs.Vertex, rs.V2EDim)(rs.v2e_table.astype(np.int64))

    @field_operator(backend=fieldview_backend)
    def reduction(nb_field: Field[[rs.Vertex, rs.V2EDim], int64]) -> Field[[rs.Vertex], int64]:  # type: ignore
        return neighbor_sum(nb_field, axis=V2EDim)

    reduction(nb_field, out=rs.out, offset_provider=rs.offset_provider)

    ref = np.sum(rs.v2e_table, axis=1)
    assert np.allclose(ref, rs.out)


def test_reduction_expression(reduction_setup, fieldview_backend):
    """Test reduction with an expression directly inside the call."""
    rs = reduction_setup
    Vertex = rs.Vertex
    Edge = rs.Edge
    V2EDim = rs.V2EDim
    V2E = rs.V2E

    @field_operator
    def reduce_expr(edge_f: Field[[Edge], int64]) -> Field[[Vertex], int64]:
        tmp_nbh_tup = edge_f(V2E), edge_f(V2E)
        tmp_nbh = tmp_nbh_tup[0]
        return 3 * neighbor_sum(-edge_f(V2E) * tmp_nbh * 2, axis=V2EDim)

    @program(backend=fieldview_backend)
    def fencil(edge_f: Field[[Edge], int64], out: Field[[Vertex], int64]) -> None:
        reduce_expr(edge_f, out=out)

    fencil(rs.inp, rs.out, offset_provider=rs.offset_provider)

    ref = 3 * np.sum(-(rs.v2e_table**2) * 2, axis=1)
    assert np.allclose(ref, rs.out.array())


def test_reduction_expression2(reduction_setup, fieldview_backend):
    # if fieldview_backend == gtfn_cpu.run_gtfn:
    #    pytest.skip("IndexFields are not supported yet.")
    rs = reduction_setup
    Vertex = rs.Vertex
    Edge = rs.Edge
    V2EDim = rs.V2EDim
    V2E = rs.V2E

    vertex_field = np_as_located_field(Vertex)(np.arange(0, rs.num_vertices, 1))
    edge_field = np_as_located_field(Edge)(np.arange(0, rs.num_edges, 1))
    vertex_v2e_field = np_as_located_field(Vertex, V2EDim)(rs.v2e_table.astype(np.int64))
    out = np_as_located_field(Vertex)(np.zeros(rs.num_vertices, dtype=np.int64))

    @field_operator
    def reduction(
        vertex_field: Field[[Vertex], int64],
        edge_field: Field[[Edge], int64],
        vertex_v2e_field: Field[[Vertex, V2EDim], int64],
    ):
        tmp = vertex_field + edge_field(V2E) + vertex_v2e_field
        return neighbor_sum(tmp, axis=V2EDim)

    reduction(
        vertex_field, edge_field, vertex_v2e_field, out=out, offset_provider=rs.offset_provider
    )

    ref = np.sum(
        vertex_field.array()[:, np.newaxis] + edge_field.array()[rs.v2e_table] + rs.v2e_table,
        axis=1,
    )
    assert np.allclose(ref, out.array())


def test_math_builtin_with_sparse_field(reduction_setup, fieldview_backend):
    # if fieldview_backend == gtfn_cpu.run_gtfn:
    #    pytest.skip("IndexFields are not supported yet.")
    rs = reduction_setup
    Vertex = rs.Vertex
    Edge = rs.Edge
    V2EDim = rs.V2EDim
    V2E = rs.V2E

    edge_field = np_as_located_field(Edge)(np.arange(0, rs.num_edges, 1, dtype=np.int32))
    vertex_v2e_field = np_as_located_field(Vertex, V2EDim)(rs.v2e_table.astype(np.int32))
    out = np_as_located_field(Vertex)(np.zeros(rs.num_vertices, dtype=np.int32))

    @field_operator
    def reduction(
        edge_field: Field[[Edge], int32], vertex_v2e_field: Field[[Vertex, V2EDim], int32]
    ):
        return neighbor_sum(minimum(edge_field(V2E), vertex_v2e_field), axis=V2EDim)

    reduction(edge_field, vertex_v2e_field, out=out, offset_provider=rs.offset_provider)

    ref = np.sum(np.minimum(edge_field.array()[rs.v2e_table], rs.v2e_table), axis=1)
    assert np.allclose(ref, out.array())


def test_conditional_nested_tuple():
    a_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    b_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    out_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    out_I_float_1 = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    mask = np_as_located_field(IDim)(np.zeros((size,), dtype=bool))

    @field_operator
    def conditional_nested_tuple(
        mask: Field[[IDim], bool], a: Field[[IDim], float64], b: Field[[IDim], float64]
    ) -> tuple[
        tuple[Field[[IDim], float64], Field[[IDim], float64]],
        tuple[Field[[IDim], float64], Field[[IDim], float64]],
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
    a_I_int = np_as_located_field(IDim)(np.random.randn(size).astype("int64"))
    out_IJ_int = np_as_located_field(IDim, JDim)(np.zeros((size, size), dtype=int64))

    @field_operator(backend=fieldview_backend)
    def simple_broadcast(inp: Field[[IDim], int64]) -> Field[[IDim, JDim], int64]:
        return broadcast(inp, (IDim, JDim))

    simple_broadcast(a_I_int, out=out_IJ_int, offset_provider={})

    assert np.allclose(a_I_int.array()[:, np.newaxis], out_IJ_int)


def test_broadcast_scalar(fieldview_backend):
    out_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))

    @field_operator(backend=fieldview_backend)
    def scalar_broadcast() -> Field[[IDim], float64]:
        return broadcast(float(1.0), (IDim,))

    scalar_broadcast(out=out_I_float, offset_provider={})

    assert np.allclose(1, out_I_float)


def test_broadcast_two_fields(fieldview_backend):
    a_I_int = np_as_located_field(IDim)(np.random.randn(size).astype("int64"))
    b_J_int = np_as_located_field(JDim)(np.random.randn(size).astype("int64"))
    out_IJ_int = np_as_located_field(IDim, JDim)(np.zeros((size, size), dtype=int64))

    @field_operator(backend=fieldview_backend)
    def broadcast_two_fields(
        inp1: Field[[IDim], int64], inp2: Field[[JDim], int64]
    ) -> Field[[IDim, JDim], int64]:
        a = broadcast(inp1, (IDim, JDim))
        b = broadcast(inp2, (IDim, JDim))
        return a + b

    broadcast_two_fields(a_I_int, b_J_int, out=out_IJ_int, offset_provider={})

    expected = a_I_int.array()[:, np.newaxis] + b_J_int.array()[np.newaxis, :]

    assert np.allclose(expected, out_IJ_int)


def test_broadcast_shifted(fieldview_backend):
    a_I_int = np_as_located_field(IDim)(np.random.randn(size).astype("int64"))
    out_IJ_int = np_as_located_field(IDim, JDim)(np.zeros((size, size), dtype=int64))

    @field_operator(backend=fieldview_backend)
    def simple_broadcast(inp: Field[[IDim], int64]) -> Field[[IDim, JDim], int64]:
        bcasted = broadcast(inp, (IDim, JDim))
        return bcasted(Joff[1])

    simple_broadcast(a_I_int, out=out_IJ_int, offset_provider={"Joff": JDim})

    assert np.allclose(a_I_int.array()[:, np.newaxis], out_IJ_int)


def test_conditional(fieldview_backend):
    a_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    b_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    out_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    mask = np_as_located_field(IDim)(np.zeros((size,), dtype=bool))

    @field_operator(backend=fieldview_backend)
    def conditional(
        mask: Field[[IDim], bool], a: Field[[IDim], float64], b: Field[[IDim], float64]
    ) -> Field[[IDim], float64]:
        return where(mask, a, b)

    conditional(mask, a_I_float, b_I_float, out=out_I_float, offset_provider={})

    assert np.allclose(np.where(mask, a_I_float, b_I_float), out_I_float)


def test_conditional_promotion(fieldview_backend):
    a_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    out_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    mask = np_as_located_field(IDim)(np.zeros((size,), dtype=bool))

    @field_operator(backend=fieldview_backend)
    def conditional_promotion(
        mask: Field[[IDim], bool], a: Field[[IDim], float64]
    ) -> Field[[IDim], float64]:
        return where(mask, a, 10.0)

    conditional_promotion(mask, a_I_float, out=out_I_float, offset_provider={})

    assert np.allclose(np.where(mask, a_I_float, 10), out_I_float)


def test_conditional_compareop(fieldview_backend):
    a_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    out_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))

    @field_operator(backend=fieldview_backend)
    def conditional_promotion(a: Field[[IDim], float64]) -> Field[[IDim], float64]:
        return where(a != a, a, 10.0)

    conditional_promotion(a_I_float, out=out_I_float, offset_provider={})

    assert np.allclose(
        np.where(np.asarray(a_I_float) != np.asarray(a_I_float), a_I_float, 10), out_I_float
    )


def test_conditional_shifted(fieldview_backend):
    a_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    b_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    out_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    mask = np_as_located_field(IDim)(np.zeros((size,), dtype=bool))

    @field_operator()
    def conditional_shifted(
        mask: Field[[IDim], bool], a: Field[[IDim], float64], b: Field[[IDim], float64]
    ) -> Field[[IDim], float64]:
        tmp = where(mask, a, b)
        return tmp(Ioff[1])

    @program(backend=fieldview_backend)
    def conditional_program(
        mask: Field[[IDim], bool],
        a: Field[[IDim], float64],
        b: Field[[IDim], float64],
        out: Field[[IDim], float64],
    ):
        conditional_shifted(mask, a, b, out=out[:-1])

    conditional_program(mask, a_I_float, b_I_float, out_I_float, offset_provider={"Ioff": IDim})

    assert np.allclose(np.where(mask, a_I_float, b_I_float)[1:], out_I_float.array()[:-1])


def test_promotion(fieldview_backend):
    ksize = 5
    a = np_as_located_field(Edge, KDim)(np.ones((size, ksize)))
    b = np_as_located_field(KDim)(np.ones((ksize)) * 2)
    c = np_as_located_field(Edge, KDim)(np.zeros((size, ksize)))

    @field_operator(backend=fieldview_backend)
    def promotion(
        inp1: Field[[Edge, KDim], float64], inp2: Field[[KDim], float64]
    ) -> Field[[Edge, KDim], float64]:
        return inp1 / inp2

    promotion(a, b, out=c, offset_provider={})

    assert np.allclose((a.array() / b.array()), c)
