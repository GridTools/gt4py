# -*- coding: utf-8 -*-
#
# GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
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
from collections import namedtuple
from typing import TypeVar

import numpy as np
import pytest

from functional.common import DimensionKind
from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import (
    Dimension,
    Field,
    FieldOffset,
    broadcast,
    float64,
    int64,
    max_over,
    min_over,
    neighbor_sum,
    where,
)
from functional.iterator.embedded import (
    NeighborTableOffsetProvider,
    index_field,
    np_as_located_field,
)
from functional.program_processors.runners import gtfn_cpu, roundtrip


@pytest.fixture(params=[roundtrip.executor, gtfn_cpu.run_gtfn])
def fieldview_backend(request):
    yield request.param


def debug_itir(tree):
    """Compare tree snippets while debugging."""
    from devtools import debug

    from eve.codegen import format_python_source
    from functional.program_processors import EmbeddedDSL

    debug(format_python_source(EmbeddedDSL.apply(tree)))


DimsType = TypeVar("DimsType")
DType = TypeVar("DType")

IDim = Dimension("IDim")
JDim = Dimension("JDim")

size = 10
mask = np_as_located_field(IDim)(np.zeros((size,), dtype=bool))
mask.array()[size // 2] = True
a_int = np_as_located_field(IDim)(np.random.randn(size).astype("int64"))
b_int = np_as_located_field(JDim)(np.random.randn(size).astype("int64"))
out_int = np_as_located_field(IDim, JDim)(np.zeros((size, size), dtype=int64))
a_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
b_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
out_float = np_as_located_field(IDim)(np.zeros((size), dtype=float64))


@pytest.fixture
def reduction_setup():
    num_vertices = 9
    edge = Dimension("Edge")
    vertex = Dimension("Vertex")
    v2edim = Dimension("V2E", kind=DimensionKind.LOCAL)
    e2vdim = Dimension("E2V", kind=DimensionKind.LOCAL)

    v2e_arr = np.array(
        [
            [0, 15, 2, 9],  # 0
            [1, 16, 0, 10],
            [2, 17, 1, 11],
            [3, 9, 5, 12],  # 3
            [4, 10, 3, 13],
            [5, 11, 4, 14],
            [6, 12, 8, 15],  # 6
            [7, 13, 6, 16],
            [8, 14, 7, 17],
        ]
    )

    # create e2v connectivity by inverting v2e
    num_edges = np.max(v2e_arr) + 1
    e2v_arr = [[] for _ in range(0, num_edges)]
    for v in range(0, v2e_arr.shape[0]):
        for e in v2e_arr[v]:
            e2v_arr[e].append(v)
    assert all(len(row) == 2 for row in e2v_arr)
    e2v_arr = np.asarray(e2v_arr)

    yield namedtuple(
        "ReductionSetup",
        [
            "num_vertices",
            "num_edges",
            "Edge",
            "Vertex",
            "V2EDim",
            "E2VDim",
            "V2E",
            "E2V",
            "inp",
            "out",
            "offset_provider",
            "v2e_table",
            "e2v_table",
        ],
    )(
        num_vertices=num_vertices,
        num_edges=num_edges,
        Edge=edge,
        Vertex=vertex,
        V2EDim=v2edim,
        E2VDim=e2vdim,
        V2E=FieldOffset("V2E", source=edge, target=(vertex, v2edim)),
        E2V=FieldOffset("E2V", source=vertex, target=(edge, e2vdim)),
        inp=index_field(edge, dtype=np.int64),
        out=np_as_located_field(vertex)(np.zeros([num_vertices], dtype=np.int64)),
        offset_provider={
            "V2E": NeighborTableOffsetProvider(v2e_arr, vertex, edge, 4),
            "E2V": NeighborTableOffsetProvider(e2v_arr, edge, vertex, 2, has_skip_values=False),
        },
        v2e_table=v2e_arr,
        e2v_table=e2v_arr,
    )  # type: ignore


def test_maxover_execution(reduction_setup, fieldview_backend):
    """Testing max_over functionality."""
    if fieldview_backend == gtfn_cpu.run_gtfn:
        pytest.skip("not yet supported.")

    rs = reduction_setup
    Vertex, V2EDim = rs.Vertex, rs.V2EDim
    inp_field = np_as_located_field(Vertex, V2EDim)(rs.v2e_table)

    @field_operator(backend=fieldview_backend)
    def maxover_fieldoperator(inp_field: Field[[Vertex, V2EDim], int64]) -> Field[[Vertex], int64]:
        return max_over(inp_field, axis=V2EDim)

    maxover_fieldoperator(inp_field, out=rs.out, offset_provider=rs.offset_provider)

    ref = np.max(rs.v2e_table, axis=1)
    assert np.allclose(ref, rs.out)


def test_maxover_execution_negatives(reduction_setup, fieldview_backend):
    """Testing max_over functionality for negative values in array."""
    if fieldview_backend == gtfn_cpu.run_gtfn:
        pytest.skip("not yet supported.")

    rs = reduction_setup
    Vertex, V2EDim, V2E = rs.Vertex, rs.V2EDim, rs.V2E
    Edge = rs.Edge
    edge_num = np.max(rs.v2e_table)
    inp_field_arr = np.arange(-edge_num // 2, edge_num // 2 + 1, 1, dtype=int)
    inp_field = np_as_located_field(Edge)(inp_field_arr)

    @field_operator(backend=fieldview_backend)
    def maxover_negvals(edge_f: Field[[Edge], int64]) -> Field[[Vertex], int64]:
        out = max_over(edge_f(V2E), axis=V2EDim)
        return out

    maxover_negvals(inp_field, out=rs.out, offset_provider=rs.offset_provider)

    ref = np.max(inp_field_arr[rs.v2e_table], axis=1)
    assert np.allclose(ref, rs.out)


def test_minover_execution(reduction_setup, fieldview_backend):
    """Testing the min_over functionality"""
    if fieldview_backend == gtfn_cpu.run_gtfn:
        pytest.skip("not implemented yet")

    rs = reduction_setup
    Vertex, V2EDim = rs.Vertex, rs.V2EDim
    in_field = np_as_located_field(Vertex, V2EDim)(rs.v2e_table)

    @field_operator
    def minover_fieldoperator(input: Field[[Vertex, V2EDim], int64]) -> Field[[Vertex], int64]:
        return min_over(input, axis=V2EDim)

    minover_fieldoperator(in_field, out=rs.out, offset_provider=rs.offset_provider)

    ref = np.min(rs.v2e_table, axis=1)
    assert np.allclose(ref, rs.out)


def test_minover_execution_float(reduction_setup, fieldview_backend):
    """Testing the min_over functionality"""
    if fieldview_backend == gtfn_cpu.run_gtfn:
        pytest.skip("not implemented yet")

    rs = reduction_setup
    Vertex, V2EDim = rs.Vertex, rs.V2EDim
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
    if fieldview_backend == gtfn_cpu.run_gtfn:
        pytest.skip("IndexFields are not supported yet.")

    rs = reduction_setup
    Edge = rs.Edge
    Vertex, V2EDim, V2E = rs.Vertex, rs.V2EDim, rs.V2E

    @field_operator
    def reduction(edge_f: Field[[Edge], int64]) -> Field[[Vertex], int64]:
        return neighbor_sum(edge_f(V2E), axis=V2EDim)

    @program(backend=fieldview_backend)
    def fencil(edge_f: Field[[Edge], int64], out: Field[[Vertex], int64]) -> None:
        reduction(edge_f, out=out)

    fencil(rs.inp, rs.out, offset_provider=rs.offset_provider)

    ref = np.sum(rs.v2e_table, axis=1)
    assert np.allclose(ref, rs.out)


def test_reduction_execution_nb(reduction_setup, fieldview_backend):
    """Testing a neighbor sum on a neighbor field."""
    if fieldview_backend == gtfn_cpu.run_gtfn:
        pytest.skip("not yet supported.")

    rs = reduction_setup
    Vertex, V2EDim = rs.Vertex, rs.V2EDim
    nb_field = np_as_located_field(Vertex, V2EDim)(rs.v2e_table)

    @field_operator(backend=fieldview_backend)
    def reduction(nb_field: Field[[Vertex, V2EDim], int64]) -> Field[[Vertex], int64]:
        return neighbor_sum(nb_field, axis=V2EDim)

    reduction(nb_field, out=rs.out, offset_provider=rs.offset_provider)

    ref = np.sum(rs.v2e_table, axis=1)
    assert np.allclose(ref, rs.out)


def test_reduction_expression(reduction_setup, fieldview_backend):
    """Test reduction with an expression directly inside the call."""
    if fieldview_backend == gtfn_cpu.run_gtfn:
        pytest.skip("IndexFields are not supported yet.")

    rs = reduction_setup
    Vertex, V2EDim, V2E = rs.Vertex, rs.V2EDim, rs.V2E
    Edge = rs.Edge

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


def test_conditional_nested_tuple():
    b = np_as_located_field(IDim)(np.ones((size)))
    out_float_2 = np_as_located_field(IDim)(np.zeros((size), dtype=float64))

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
        a_float,
        b,
        out=((out_float, out_float_2), (out_float_2, out_float)),
        offset_provider={},
    )

    expected = np.where(
        mask,
        ((a_float, b), (b, a_float)),
        ((np.full(size, 5.0), np.full(size, 7.0)), (np.full(size, 7.0), np.full(size, 5.0))),
    )

    assert np.allclose(expected, ((out_float, out_float_2), (out_float_2, out_float)))


def test_broadcast_simple(fieldview_backend):
    @field_operator(backend=fieldview_backend)
    def simple_broadcast(inp: Field[[IDim], int64]) -> Field[[IDim, JDim], int64]:
        return broadcast(inp, (IDim, JDim))

    simple_broadcast(a_int, out=out_int, offset_provider={})

    assert np.allclose(a_int.array()[:, np.newaxis], out_int)


def test_broadcast_scalar(fieldview_backend):
    @field_operator(backend=fieldview_backend)
    def scalar_broadcast() -> Field[[IDim], float64]:
        return broadcast(float(1.0), (IDim,))

    scalar_broadcast(out=out_float, offset_provider={})

    assert np.allclose(1, out_float)


def test_broadcast_two_fields(fieldview_backend):
    @field_operator(backend=fieldview_backend)
    def broadcast_two_fields(
        inp1: Field[[IDim], int64], inp2: Field[[JDim], int64]
    ) -> Field[[IDim, JDim], int64]:
        a = broadcast(inp1, (IDim, JDim))
        b = broadcast(inp2, (IDim, JDim))
        return a + b

    broadcast_two_fields(a_int, b_int, out=out_int, offset_provider={})

    expected = a_int.array()[:, np.newaxis] + b_int.array()[np.newaxis, :]

    assert np.allclose(expected, out_int)


def test_broadcast_shifted(fieldview_backend):
    Joff = FieldOffset("Joff", source=JDim, target=(JDim,))

    @field_operator(backend=fieldview_backend)
    def simple_broadcast(inp: Field[[IDim], int64]) -> Field[[IDim, JDim], int64]:
        bcasted = broadcast(inp, (IDim, JDim))
        return bcasted(Joff[1])

    simple_broadcast(a_int, out=out_int, offset_provider={"Joff": JDim})

    assert np.allclose(a_int.array()[:, np.newaxis], out_int)


def test_conditional(fieldview_backend):
    @field_operator(backend=fieldview_backend)
    def conditional(
        mask: Field[[IDim], bool], a: Field[[IDim], float64], b: Field[[IDim], float64]
    ) -> Field[[IDim], float64]:
        return where(mask, a, b)

    conditional(mask, a_float, b_float, out=out_float, offset_provider={})

    assert np.allclose(np.where(mask, a_float, b_float), out_float)


def test_conditional_promotion(fieldview_backend):
    @field_operator(backend=fieldview_backend)
    def conditional_promotion(
        mask: Field[[IDim], bool], a: Field[[IDim], float64]
    ) -> Field[[IDim], float64]:
        return where(mask, a, 10.0)

    conditional_promotion(mask, a_float, out=out_float, offset_provider={})

    assert np.allclose(np.where(mask, a_float, 10), out_float)


def test_conditional_compareop(fieldview_backend):
    @field_operator(backend=fieldview_backend)
    def conditional_promotion(a: Field[[IDim], float64]) -> Field[[IDim], float64]:
        return where(a != a, a, 10.0)

    conditional_promotion(a_float, out=out_float, offset_provider={})

    assert np.allclose(np.where(np.asarray(a_float) != np.asarray(a_float), a_float, 10), out_float)


def test_conditional_shifted(fieldview_backend):
    Ioff = FieldOffset("Ioff", source=IDim, target=(IDim,))

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

    conditional_program(mask, a_float, b_float, out_float, offset_provider={"Ioff": IDim})

    assert np.allclose(np.where(mask, a_float, b_float)[1:], out_float.array()[:-1])
