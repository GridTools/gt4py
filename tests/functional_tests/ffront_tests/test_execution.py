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
from functional.ffront.decorator import field_operator, program, scan_operator
from functional.ffront.fbuiltins import (
    Dimension,
    Field,
    FieldOffset,
    broadcast,
    float64,
    int32,
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


def test_copy(fieldview_backend):
    size = 10
    a = np_as_located_field(IDim)(np.ones((size)))
    b = np_as_located_field(IDim)(np.zeros((size)))

    @field_operator(backend=fieldview_backend)
    def copy(inp: Field[[IDim], float64]) -> Field[[IDim], float64]:
        return inp

    copy(a, out=b, offset_provider={})

    assert np.allclose(a, b)


@pytest.mark.skip(reason="no lowering for returning a tuple of fields exists yet.")
def test_multicopy(fieldview_backend):
    size = 10
    a = np_as_located_field(IDim)(np.ones((size)))
    b = np_as_located_field(IDim)(np.ones((size)) * 3)
    c = np_as_located_field(IDim)(np.zeros((size)))
    d = np_as_located_field(IDim)(np.zeros((size)))

    @field_operator(backend=fieldview_backend)
    def multicopy(
        inp1: Field[[IDim], float64], inp2: Field[[IDim], float64]
    ) -> tuple[Field[[IDim], float64], Field[[IDim], float64]]:
        return inp1, inp2

    multicopy(a, b, out=(c, d), offset_provider={})

    assert np.allclose(a, c)
    assert np.allclose(b, d)


def test_arithmetic(fieldview_backend):
    size = 10
    a = np_as_located_field(IDim)(np.ones((size)))
    b = np_as_located_field(IDim)(np.ones((size)) * 2)
    c = np_as_located_field(IDim)(np.zeros((size)))

    @field_operator(backend=fieldview_backend)
    def arithmetic(
        inp1: Field[[IDim], float64], inp2: Field[[IDim], float64]
    ) -> Field[[IDim], float64]:
        return (inp1 + inp2) * 2.0

    arithmetic(a, b, out=c, offset_provider={})

    assert np.allclose((a.array() + b.array()) * 2.0, c)


def test_power(fieldview_backend):
    if fieldview_backend == gtfn_cpu.run_gtfn:
        pytest.xfail("gtfn does not yet support math builtins")

    size = 10
    a = np_as_located_field(IDim)(np.random.randn((size)))
    b = np_as_located_field(IDim)(np.zeros((size)))

    @field_operator(backend=fieldview_backend)
    def pow(inp1: Field[[IDim], float64]) -> Field[[IDim], float64]:
        return inp1**2

    pow(a, out=b, offset_provider={})

    assert np.allclose(a.array() ** 2, b)


def test_power_arithmetic(fieldview_backend):
    if fieldview_backend == gtfn_cpu.run_gtfn:
        pytest.xfail("gtfn does not yet support math builtins")

    size = 10
    a = np_as_located_field(IDim)(np.random.randn((size)))
    b = np_as_located_field(IDim)(np.zeros((size)))
    c = np_as_located_field(IDim)(np.random.randn((size)))

    @field_operator(backend=fieldview_backend)
    def power_arithmetic(
        inp1: Field[[IDim], float64], inp2: Field[[IDim], float64]
    ) -> Field[[IDim], float64]:
        return inp2 + ((inp1 + inp2) ** 2)

    power_arithmetic(a, c, out=b, offset_provider={})

    assert np.allclose(c.array() + ((c.array() + a.array()) ** 2), b)


def test_bit_logic(fieldview_backend):
    size = 10
    a = np_as_located_field(IDim)(np.full((size), True))
    b_data = np.full((size), True)
    b_data[5] = False
    b = np_as_located_field(IDim)(b_data)
    c = np_as_located_field(IDim)(np.full((size), False))

    @field_operator(backend=fieldview_backend)
    def bit_and(inp1: Field[[IDim], bool], inp2: Field[[IDim], bool]) -> Field[[IDim], bool]:
        return inp1 & inp2 & True

    bit_and(a, b, out=c, offset_provider={})

    assert np.allclose(a.array() & b.array(), c)


def test_unary_neg(fieldview_backend):
    size = 10
    a = np_as_located_field(IDim)(np.ones((size), dtype=int32))
    b = np_as_located_field(IDim)(np.zeros((size), dtype=int32))

    @field_operator(backend=fieldview_backend)
    def uneg(inp: Field[[IDim], int32]) -> Field[[IDim], int32]:
        return -inp

    uneg(a, out=b, offset_provider={})

    assert np.allclose(b, np.full((size), -1, dtype=int32))


def test_shift(fieldview_backend):
    size = 10
    Ioff = FieldOffset("Ioff", source=IDim, target=(IDim,))
    a = np_as_located_field(IDim)(np.arange(size + 1, dtype=np.float64))
    b = np_as_located_field(IDim)(np.zeros((size)))

    @field_operator
    def shift_by_one(inp: Field[[IDim], float64]) -> Field[[IDim], float64]:
        return inp(Ioff[1])

    @program(backend=fieldview_backend)
    def fencil(inp: Field[[IDim], float64], out: Field[[IDim], float64]) -> None:
        shift_by_one(inp, out=out)

    fencil(a, b, offset_provider={"Ioff": IDim})

    assert np.allclose(b.array(), np.arange(1, 11))


def test_fold_shifts(fieldview_backend):
    """Shifting the result of an addition should work."""
    size = 10
    Ioff = FieldOffset("Ioff", source=IDim, target=(IDim,))
    a = np_as_located_field(IDim)(np.arange(size + 1, dtype=np.float64))
    b = np_as_located_field(IDim)(np.ones((size + 2)) * 2)
    c = np_as_located_field(IDim)(np.zeros((size)))

    @field_operator
    def auto_lift(
        inp1: Field[[IDim], float64], inp2: Field[[IDim], float64]
    ) -> Field[[IDim], float64]:
        tmp = inp1 + inp2(Ioff[1])
        return tmp(Ioff[1])

    @program(backend=fieldview_backend)
    def fencil(
        inp1: Field[[IDim], float64], inp2: Field[[IDim], float64], out: Field[[IDim], float64]
    ) -> None:
        auto_lift(inp1, inp2, out=out)

    fencil(a, b, c, offset_provider={"Ioff": IDim})

    assert np.allclose(a[1:] + b[2:], c)


def test_tuples(fieldview_backend):
    if fieldview_backend == gtfn_cpu.run_gtfn:
        pytest.skip("Tuples are not supported yet.")
    size = 10
    a = np_as_located_field(IDim)(np.ones((size)))
    b = np_as_located_field(IDim)(np.ones((size)) * 2)
    c = np_as_located_field(IDim)(np.zeros((size)))

    @field_operator
    def tuples(
        inp1: Field[[IDim], float64], inp2: Field[[IDim], float64]
    ) -> Field[[IDim], float64]:
        inps = inp1, inp2
        scalars = 1.3, float64(5.0), float64("3.4")
        return (inps[0] * scalars[0] + inps[1] * scalars[1]) * scalars[2]

    @program(backend=fieldview_backend)
    def fencil(
        inp1: Field[[IDim], float64], inp2: Field[[IDim], float64], out: Field[[IDim], float64]
    ) -> None:
        tuples(inp1, inp2, out=out)

    fencil(a, b, c, offset_provider={})

    assert np.allclose((a.array() * 1.3 + b.array() * 5.0) * 3.4, c)


def test_promotion(fieldview_backend):
    Edge = Dimension("Edge")
    K = Dimension("K")

    size = 10
    ksize = 5
    a = np_as_located_field(Edge, K)(np.ones((size, ksize)))
    b = np_as_located_field(K)(np.ones((ksize)) * 2)
    c = np_as_located_field(Edge, K)(np.zeros((size, ksize)))

    @field_operator(backend=fieldview_backend)
    def promotion(
        inp1: Field[[Edge, K], float64], inp2: Field[[K], float64]
    ) -> Field[[Edge, K], float64]:
        return inp1 / inp2

    promotion(a, b, out=c, offset_provider={})

    assert np.allclose((a.array() / b.array()), c)


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


def test_maxover_execution_sparse(reduction_setup, fieldview_backend):
    """Testing max_over functionality."""
    if fieldview_backend == gtfn_cpu.run_gtfn:
        pytest.skip("not yet supported.")
    rs = reduction_setup
    Vertex = rs.Vertex
    V2EDim = rs.V2EDim

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
    if fieldview_backend == gtfn_cpu.run_gtfn:
        pytest.skip("not implemented yet")
    rs = reduction_setup
    Vertex = rs.Vertex
    V2EDim = rs.V2EDim

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
    if fieldview_backend == gtfn_cpu.run_gtfn:
        pytest.skip("IndexFields are not supported yet.")
    rs = reduction_setup
    Edge = rs.Edge
    Vertex = rs.Vertex
    V2EDim = rs.V2EDim
    V2E = rs.V2E

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
    V2EDim = rs.V2EDim

    nb_field = np_as_located_field(rs.Vertex, rs.V2EDim)(rs.v2e_table)

    @field_operator(backend=fieldview_backend)
    def reduction(nb_field: Field[[rs.Vertex, rs.V2EDim], int64]) -> Field[[rs.Vertex], int64]:  # type: ignore
        return neighbor_sum(nb_field, axis=V2EDim)

    reduction(nb_field, out=rs.out, offset_provider=rs.offset_provider)

    ref = np.sum(rs.v2e_table, axis=1)
    assert np.allclose(ref, rs.out)


def test_reduction_expression(reduction_setup, fieldview_backend):
    """Test reduction with an expression directly inside the call."""
    if fieldview_backend == gtfn_cpu.run_gtfn:
        pytest.skip("IndexFields are not supported yet.")
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


def test_scalar_arg(fieldview_backend):
    """Test scalar argument being turned into 0-dim field."""
    if fieldview_backend == gtfn_cpu.run_gtfn:
        pytest.skip("ConstantFields are not supported yet.")
    Vertex = Dimension("Vertex")
    size = 5
    inp = 5.0
    out = np_as_located_field(Vertex)(np.zeros([size]))

    @field_operator(backend=fieldview_backend)
    def scalar_arg(scalar_inp: float64) -> Field[[Vertex], float64]:
        return broadcast(scalar_inp + 1.0, (Vertex,))

    scalar_arg(inp, out=out, offset_provider={})

    ref = np.full([size], 6.0)
    assert np.allclose(ref, out.array())


def test_conditional_tuple_1():
    size = 10
    mask = np_as_located_field(IDim)(np.zeros((size,), dtype=bool))
    mask.array()[0 : (size // 2)] = True
    a = np_as_located_field(IDim)(np.ones((size,)))
    c = np_as_located_field(IDim)(np.zeros((size,)))
    d = np_as_located_field(IDim)(np.zeros((size,)))

    @field_operator
    def conditional_tuple(
        mask: Field[[IDim], bool], a: Field[[IDim], float64]
    ) -> tuple[Field[[IDim], float64], Field[[IDim], float64]]:
        return where(mask, (a, 3.0), (2.0, 7.0))

    @program
    def conditional_tuple_p(
        mask: Field[[IDim], bool],
        a: Field[[IDim], float64],
        c: Field[[IDim], float64],
        d: Field[[IDim], float64],
    ):
        conditional_tuple(mask, a, out=(c, d))

    conditional_tuple_p(mask, a, c, d, offset_provider={})

    assert np.allclose(
        np.where(mask, (a, np.full(size, 3.0)), (np.full(size, 2.0), np.full(size, 7.0))), (c, d)
    )


def test_conditional_nested_tuple():
    size = 10
    mask = np_as_located_field(IDim)(np.zeros((size,), dtype=bool))
    mask.array()[0 : (size // 2)] = True
    a = np_as_located_field(IDim)(np.ones((size,)))
    b = np_as_located_field(IDim)(np.ones((size,)))
    c = np_as_located_field(IDim)(np.zeros((size,)))
    d = np_as_located_field(IDim)(np.zeros((size,)))

    @field_operator
    def conditional_tuple_3_field_op(
        mask: Field[[IDim], bool], a: Field[[IDim], float64], b: Field[[IDim], float64]
    ) -> tuple[
        tuple[Field[[IDim], float64], Field[[IDim], float64]],
        tuple[Field[[IDim], float64], Field[[IDim], float64]],
    ]:
        return where(mask, ((a, b), (b, a)), ((5.0, 7.0), (7.0, 5.0)))

    @program
    def conditional_tuple_3_p(
        mask: Field[[IDim], bool],
        a: Field[[IDim], float64],
        b: Field[[IDim], float64],
        c: Field[[IDim], float64],
        d: Field[[IDim], float64],
    ):
        conditional_tuple_3_field_op(mask, a, b, out=((c, d), (d, c)))

    conditional_tuple_3_p(mask, a, b, c, d, offset_provider={})

    assert np.allclose(
        np.where(
            mask,
            ((a, b), (b, a)),
            ((np.full(size, 5.0), np.full(size, 7.0)), (np.full(size, 7.0), np.full(size, 5.0))),
        ),
        ((c, d), (d, c)),
    )


def test_nested_scalar_arg(fieldview_backend):
    if fieldview_backend == gtfn_cpu.run_gtfn:
        pytest.skip("ConstantFields are not supported yet.")
    Vertex = Dimension("Vertex")
    size = 5
    inp = 5.0
    out = np_as_located_field(Vertex)(np.zeros([size]))

    @field_operator(backend=fieldview_backend)
    def scalar_arg_inner(scalar_inp: float64) -> Field[[Vertex], float64]:
        return broadcast(scalar_inp + 1.0, (Vertex,))

    @field_operator(backend=fieldview_backend)
    def scalar_arg(scalar_inp: float64) -> Field[[Vertex], float64]:
        return scalar_arg_inner(scalar_inp + 1.0)

    scalar_arg(inp, out=out, offset_provider={})

    ref = np.full([size], 7.0)
    assert np.allclose(ref, out.array())


def test_scalar_arg_with_field(fieldview_backend):
    if fieldview_backend == gtfn_cpu.run_gtfn:
        pytest.skip("IndexFields and ConstantFields are not supported yet.")
    Edge = Dimension("Edge")
    EdgeOffset = FieldOffset("EdgeOffset", source=Edge, target=(Edge,))
    size = 5
    inp = index_field(Edge, dtype=float64)
    factor = 3.0
    out = np_as_located_field(Edge)(np.zeros((size), dtype=np.float64))

    @field_operator
    def scalar_and_field_args(
        inp: Field[[Edge], float64], factor: float64
    ) -> Field[[Edge], float64]:
        tmp = factor * inp
        return tmp(EdgeOffset[1])

    @program(backend=fieldview_backend)
    def fencil(out: Field[[Edge], float64], inp: Field[[Edge], float64], factor: float64) -> None:
        scalar_and_field_args(inp, factor, out=out)

    fencil(out, inp, factor, offset_provider={"EdgeOffset": Edge})

    ref = np.arange(1, size + 1) * factor
    assert np.allclose(ref, out.array())


def test_broadcast_simple(fieldview_backend):
    size = 10
    a = np_as_located_field(IDim)(np.arange(0, size, 1, dtype=int64))
    out = np_as_located_field(IDim, JDim)(np.zeros((size, size), dtype=int64))

    @field_operator(backend=fieldview_backend)
    def simple_broadcast(inp: Field[[IDim], int64]) -> Field[[IDim, JDim], int64]:
        return broadcast(inp, (IDim, JDim))

    simple_broadcast(a, out=out, offset_provider={})

    assert np.allclose(a.array()[:, np.newaxis], out)


def test_broadcast_scalar(fieldview_backend):
    size = 10
    out = np_as_located_field(IDim)(np.zeros((size)))

    @field_operator(backend=fieldview_backend)
    def scalar_broadcast() -> Field[[IDim], float64]:
        return broadcast(float(1.0), (IDim,))

    scalar_broadcast(out=out, offset_provider={})

    assert np.allclose(1, out)


def test_broadcast_two_fields(fieldview_backend):
    size = 10
    a = np_as_located_field(IDim)(np.arange(0, size, 1, dtype=int64))
    b = np_as_located_field(JDim)(np.arange(0, size, 1, dtype=int64))

    out = np_as_located_field(IDim, JDim)(np.zeros((size, size), dtype=int64))

    @field_operator(backend=fieldview_backend)
    def broadcast_two_fields(
        inp1: Field[[IDim], int64], inp2: Field[[JDim], int64]
    ) -> Field[[IDim, JDim], int64]:
        a = broadcast(inp1, (IDim, JDim))
        b = broadcast(inp2, (IDim, JDim))
        return a + b

    broadcast_two_fields(a, b, out=out, offset_provider={})

    expected = a.array()[:, np.newaxis] + b.array()[np.newaxis, :]

    assert np.allclose(expected, out)


def test_broadcast_shifted(fieldview_backend):
    Joff = FieldOffset("Joff", source=JDim, target=(JDim,))

    size = 10
    a = np_as_located_field(IDim)(np.arange(0, size, 1, dtype=int))
    out = np_as_located_field(IDim, JDim)(np.zeros((size, size), dtype=int64))

    @field_operator(backend=fieldview_backend)
    def simple_broadcast(inp: Field[[IDim], int64]) -> Field[[IDim, JDim], int64]:
        bcasted = broadcast(inp, (IDim, JDim))
        return bcasted(Joff[1])

    simple_broadcast(a, out=out, offset_provider={"Joff": JDim})

    assert np.allclose(a.array()[:, np.newaxis], out)


def test_conditional(fieldview_backend):
    size = 10
    mask = np_as_located_field(IDim)(np.zeros((size,), dtype=bool))
    mask.array()[0 : (size // 2)] = True
    a = np_as_located_field(IDim)(np.ones((size,)))
    b = np_as_located_field(IDim)(2 * np.ones((size,)))
    out = np_as_located_field(IDim)(np.zeros((size,)))

    @field_operator(backend=fieldview_backend)
    def conditional(
        mask: Field[[IDim], bool], a: Field[[IDim], float64], b: Field[[IDim], float64]
    ) -> Field[[IDim], float64]:
        return where(mask, a, b)

    conditional(mask, a, b, out=out, offset_provider={})

    assert np.allclose(np.where(mask, a, b), out)


def test_conditional_promotion(fieldview_backend):
    size = 10
    mask = np_as_located_field(IDim)(np.zeros((size,), dtype=bool))
    mask.array()[0 : (size // 2)] = True
    a = np_as_located_field(IDim)(np.ones((size,)))
    out = np_as_located_field(IDim)(np.zeros((size,)))

    @field_operator(backend=fieldview_backend)
    def conditional_promotion(
        mask: Field[[IDim], bool], a: Field[[IDim], float64]
    ) -> Field[[IDim], float64]:
        return where(mask, a, 10.0)

    conditional_promotion(mask, a, out=out, offset_provider={})

    assert np.allclose(np.where(mask, a, 10), out)


def test_conditional_compareop(fieldview_backend):
    size = 10
    a = np_as_located_field(IDim)(np.ones((size,)))
    out = np_as_located_field(IDim)(np.zeros((size,)))

    @field_operator(backend=fieldview_backend)
    def conditional_promotion(a: Field[[IDim], float64]) -> Field[[IDim], float64]:
        return where(a != a, a, 10.0)

    conditional_promotion(a, out=out, offset_provider={})

    assert np.allclose(np.where(np.asarray(a) != np.asarray(a), a, 10), out)


def test_conditional_shifted(fieldview_backend):
    Ioff = FieldOffset("Ioff", source=IDim, target=(IDim,))

    size = 10
    mask = np_as_located_field(IDim)(np.zeros((size,), dtype=bool))
    mask.array()[size // 2] = True
    a = np_as_located_field(IDim)(np.arange(0, size, 1, dtype=float64))
    b = np_as_located_field(IDim)(np.zeros((size,)))
    out = np_as_located_field(IDim)(np.zeros((size,)))

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

    conditional_program(mask, a, b, out, offset_provider={"Ioff": IDim})

    assert np.allclose(np.where(mask, a, b)[1:], out.array()[:-1])


def test_nested_tuple_return():
    size = 10
    a = np_as_located_field(IDim)(np.ones((size,)))
    b = np_as_located_field(IDim)(2 * np.ones((size,)))
    out = np_as_located_field(IDim)(np.zeros((size,)))

    @field_operator
    def pack_tuple(
        a: Field[[IDim], float64], b: Field[[IDim], float64]
    ) -> tuple[Field[[IDim], float64], tuple[Field[[IDim], float64], Field[[IDim], float64]]]:
        return (a, (a, b))

    @field_operator
    def combine(a: Field[[IDim], float64], b: Field[[IDim], float64]) -> Field[[IDim], float64]:
        packed = pack_tuple(a, b)
        return packed[0] + packed[1][0] + packed[1][1]

    combine(a, b, out=out, offset_provider={})

    assert np.allclose(2 * a.array() + b.array(), out)


def test_tuple_return_2(reduction_setup):
    rs = reduction_setup
    Edge = rs.Edge
    Vertex = rs.Vertex
    V2EDim = rs.V2EDim
    V2E = rs.V2E

    @field_operator
    def reduction_tuple(
        a: Field[[Edge], int64], b: Field[[Edge], int64]
    ) -> tuple[Field[[Vertex], int64], Field[[Vertex], int64]]:
        a = neighbor_sum(a(V2E), axis=V2EDim)
        b = neighbor_sum(b(V2E), axis=V2EDim)
        return a, b

    @field_operator
    def combine_tuple(a: Field[[Edge], int64], b: Field[[Edge], int64]) -> Field[[Vertex], int64]:
        packed = reduction_tuple(a, b)
        return packed[0] + packed[1]

    combine_tuple(rs.inp, rs.inp, out=rs.out, offset_provider=rs.offset_provider)

    ref = np.sum(rs.v2e_table, axis=1) * 2
    assert np.allclose(ref, rs.out)


@pytest.mark.xfail(raises=NotImplementedError)
def test_tuple_with_local_field_in_reduction_shifted(reduction_setup):
    rs = reduction_setup
    Edge = rs.Edge
    Vertex = rs.Vertex
    V2EDim = rs.V2EDim
    V2E = rs.V2E
    E2V = rs.E2V

    num_vertices = rs.num_vertices
    num_edges = rs.num_edges

    size = 10
    # TODO(tehrengruber): use different values per location
    a = np_as_located_field(Edge)(np.ones((num_edges,)))
    b = np_as_located_field(Vertex)(2 * np.ones((num_vertices,)))
    out = np_as_located_field(Edge)(np.zeros((num_edges,)))

    @field_operator
    def reduce_tuple_element(
        edge_field: Field[[Edge], float64], vertex_field: Field[[Vertex], float64]
    ) -> Field[[Edge], float64]:
        tup = edge_field(V2E), vertex_field
        # the shift inside the reduction fails as tup is a tuple of iterators
        #  (as it contains a local field) which can not be shifted
        red = neighbor_sum(tup[0] + vertex_field, axis=V2EDim)
        # even if the above is fixed we need to be careful with a subsequent
        #  shift as the lifted lambda will contain tup as an argument which -
        #  again - can not be shifted.
        return red(E2V[0])

    reduce_tuple_element(a, b, out=out, offset_provider=rs.offset_provider)

    # conn table used is inverted here on purpose
    red = np.sum(np.asarray(a)[rs.v2e_table] + np.asarray(b)[:, np.newaxis], axis=1)
    expected = red[rs.e2v_table][:, 0]

    assert np.allclose(expected, out)


def test_tuple_arg(fieldview_backend):
    if fieldview_backend == gtfn_cpu.run_gtfn:
        pytest.skip("Tuple arguments are not supported in gtfn yet.")
    size = 10
    a = np_as_located_field(IDim)(np.ones((size,)))
    b = np_as_located_field(IDim)(2 * np.ones((size,)))
    out = np_as_located_field(IDim)(np.zeros((size,)))

    @field_operator(backend=fieldview_backend)
    def unpack_tuple(
        inp: tuple[tuple[Field[[IDim], float64], Field[[IDim], float64]], Field[[IDim], float64]]
    ) -> Field[[IDim], float64]:
        return 3.0 * inp[0][0] + inp[0][1] + inp[1]

    unpack_tuple(((a, b), a), out=out, offset_provider={})

    assert np.allclose(3 * a.array() + b.array() + a.array(), out)


@pytest.mark.parametrize("forward", [True, False])
def test_simple_scan(fieldview_backend, forward):
    if fieldview_backend == gtfn_cpu.run_gtfn:
        pytest.xfail("gtfn does not yet support scan pass.")

    KDim = Dimension("K", kind=DimensionKind.VERTICAL)
    size = 10
    init = 1.0
    out = np_as_located_field(KDim)(np.zeros((size,)))
    expected = np.arange(init + 1.0, init + 1.0 + size, 1)
    if not forward:
        expected = np.flip(expected)

    @scan_operator(axis=KDim, forward=forward, init=init, backend=fieldview_backend)
    def simple_scan_operator(carry: float) -> float:
        return carry + 1.0

    simple_scan_operator(out=out, offset_provider={})

    assert np.allclose(expected, out)


def test_solve_triag(fieldview_backend):
    if fieldview_backend == gtfn_cpu.run_gtfn:
        pytest.xfail("gtfn does not yet support scan pass.")

    KDim = Dimension("K", kind=DimensionKind.VERTICAL)
    shape = (3, 7, 5)
    rng = np.random.default_rng()
    a_np, b_np, c_np, d_np = (rng.normal(size=shape) for _ in range(4))
    b_np *= 2
    a, b, c, d = (
        np_as_located_field(IDim, JDim, KDim)(np_arr) for np_arr in [a_np, b_np, c_np, d_np]
    )
    out = np_as_located_field(IDim, JDim, KDim)(np.zeros(shape))

    # compute reference
    matrices = np.zeros(shape + shape[-1:])
    i = np.arange(shape[2])
    matrices[:, :, i[1:], i[:-1]] = a_np[:, :, 1:]
    matrices[:, :, i, i] = b_np
    matrices[:, :, i[:-1], i[1:]] = c_np[:, :, :-1]
    expected = np.linalg.solve(matrices, d_np)

    @scan_operator(axis=KDim, forward=True, init=(0.0, 0.0))
    def tridiag_forward(
        state: tuple[float, float], a: float, b: float, c: float, d: float
    ) -> tuple[float, float]:
        return (c / (b - a * state[0]), (d - a * state[1]) / (b - a * state[0]))

    @scan_operator(axis=KDim, forward=False, init=0.0)
    def tridiag_backward(x_kp1: float, cp: float, dp: float) -> float:
        return dp - cp * x_kp1

    @field_operator(backend=fieldview_backend)
    def solve_tridiag(
        a: Field[[IDim, JDim, KDim], float],
        b: Field[[IDim, JDim, KDim], float],
        c: Field[[IDim, JDim, KDim], float],
        d: Field[[IDim, JDim, KDim], float],
    ) -> Field[[IDim, JDim, KDim], float]:
        cp, dp = tridiag_forward(a, b, c, d)
        return tridiag_backward(cp, dp)

    solve_tridiag(a, b, c, d, out=out, offset_provider={})

    np.allclose(expected, out)


def test_ternary_operator():
    size = 10

    a = np_as_located_field(IDim)(2 * np.ones((size,)))
    b = np_as_located_field(IDim)(2 * np.ones((size,)))
    out = np_as_located_field(IDim)(np.zeros((size,)))

    left = 2.0
    right = 3.0

    @field_operator
    def ternary_field_op(
        a: Field[[IDim], float], b: Field[[IDim], float], left: float, right: float
    ) -> Field[[IDim], float]:
        return a if left < right else b

    ternary_field_op(a, b, left, right, out=out, offset_provider={})
    e = np.asarray(a) if left < right else np.asarray(b)
    np.allclose(e, out)

    @field_operator
    def ternary_field_op_scalars(left: float, right: float) -> Field[[IDim], float]:
        return broadcast(3.0, (IDim,)) if left > right else broadcast(4.0, (IDim,))

    ternary_field_op_scalars(left, right, out=out, offset_provider={})
    e = np.full(e.shape, 3.0) if left > right else e
    np.allclose(e, out)


def test_ternary_operator_tuple():
    size = 10
    a = np_as_located_field(IDim)(np.ones((size,)))
    b = np_as_located_field(IDim)(2 * np.ones((size,)))
    out_1 = np_as_located_field(IDim)(np.zeros((size,)))
    out_2 = np_as_located_field(IDim)(np.zeros((size,)))

    left = 2.0
    right = 3.0

    @field_operator
    def ternary_field_op(
        a: Field[[IDim], float], b: Field[[IDim], float], left: float, right: float
    ) -> tuple[Field[[IDim], float], Field[[IDim], float]]:
        return (a, b) if left < right else (b, a)

    # TODO(tehrengruber): directly call field operator when the generated programs support `out` being a tuple
    @program
    def ternary_field(
        a: Field[[IDim], float],
        b: Field[[IDim], float],
        left: float,
        right: float,
        out_1: Field[[IDim], float],
        out_2: Field[[IDim], float],
    ):
        ternary_field_op(a, b, left, right, out=(out_1, out_2))

    ternary_field(a, b, left, right, out_1, out_2, offset_provider={})

    e, f = (np.asarray(a), np.asarray(b)) if left < right else (np.asarray(b), np.asarray(a))
    np.allclose(e, out_1)
    np.allclose(f, out_2)


def test_ternary_builtin_neighbor_sum(reduction_setup):
    rs = reduction_setup
    Edge = rs.Edge
    Vertex = rs.Vertex
    V2EDim = rs.V2EDim
    V2E = rs.V2E

    num_vertices = rs.num_vertices
    num_edges = rs.num_edges

    a = np_as_located_field(Edge)(np.ones((num_edges,)))
    b = np_as_located_field(Edge)(2 * np.ones((num_edges,)))
    out = np_as_located_field(Vertex)(np.zeros((num_vertices,)))

    @field_operator
    def ternary_reduce(a: Field[[Edge], float], b: Field[[Edge], float]) -> Field[[Vertex], float]:
        out = neighbor_sum(b(V2E) if 2 < 3 else a(V2E), axis=V2EDim)
        return out

    ternary_reduce(a, b, out=out, offset_provider=rs.offset_provider)

    expected = (
        np.sum(np.asarray(b)[rs.v2e_table], axis=1)
        if 2 < 3
        else np.sum(np.asarray(a)[rs.v2e_table], axis=1)
    )

    assert np.allclose(expected, out)


def test_ternary_scan():
    KDim = Dimension("K", kind=DimensionKind.VERTICAL)
    size = 10
    init = 0.0
    a_float = 4
    a = np_as_located_field(KDim)(a_float * np.ones((size,)))
    out = np_as_located_field(KDim)(np.zeros((size,)))
    expected = np.asarray([i if i <= a_float else a_float + 1 for i in range(1, size + 1)])

    @scan_operator(axis=KDim, forward=True, init=init)
    def simple_scan_operator(carry: float, a: float) -> float:
        return carry if carry > a else carry + 1.0

    simple_scan_operator(a, out=out, offset_provider={})

    assert np.allclose(expected, out)


def test_scan_tuple_output(fieldview_backend):
    if fieldview_backend == gtfn_cpu.run_gtfn:
        pytest.xfail("gtfn does not yet support scan pass.")

    KDim = Dimension("K", kind=DimensionKind.VERTICAL)
    size = 10
    init = (0.0, 1.0)
    inp = np_as_located_field(KDim)(np.arange(0, size, 1.0))
    out1 = np_as_located_field(KDim)(np.zeros((size,)))
    out2 = np_as_located_field(KDim)(np.zeros((size,)))
    expected = np.arange(init[1] + 1.0, init[1] + 1.0 + size, 1)

    @scan_operator(axis=KDim, forward=True, init=init, backend=fieldview_backend)
    def simple_scan_operator(carry: tuple[float, float], x: float) -> tuple[float, float]:
        return (x, carry[1] + 1.0)

    # TODO(tehrengruber): directly call scan operator when this is supported
    #  for tuple outputs
    @program
    def simple_scan_operator_program(
        x: Field[[KDim], float], out1: Field[[KDim], float], out2: Field[[KDim], float]
    ) -> None:
        simple_scan_operator(x, out=(out1, out2))

    simple_scan_operator_program(inp, out1, out2, offset_provider={})

    assert np.allclose(inp, out1)
    assert np.allclose(expected, out2)


def test_docstring():
    size = 10
    a = np_as_located_field(IDim)(np.ones((size,)))

    @field_operator
    def fieldop_with_docstring(a: Field[[IDim], float64]) -> Field[[IDim], float64]:
        """My docstring."""
        return a

    @program
    def test_docstring(a: Field[[IDim], float64]) -> Field[[IDim], float64]:
        """My docstring."""
        fieldop_with_docstring(a, out=a)

    test_docstring(a, offset_provider={})


def test_domain(fieldview_backend):
    size = 10
    a = np_as_located_field(IDim, JDim)(np.ones((size, size)))

    @field_operator(backend=fieldview_backend)
    def fieldop_domain(a: Field[[IDim, JDim], float64]) -> Field[[IDim, JDim], float64]:
        return a + a

    @program
    def program_domain(a: Field[[IDim, JDim], float64]):
        fieldop_domain(a, out=a, domain={IDim: (1, 9), JDim: (4, 6)})

    program_domain(a, offset_provider={})

    expected = np.asarray(a)
    expected[1:9, 4:6] = 1 + 1

    assert np.allclose(expected, a)


def test_domain_input_bounds(fieldview_backend):
    if fieldview_backend == gtfn_cpu.run_gtfn:
        pytest.skip("FloorDiv and Power not fully supported in gtfn.")

    size = 10
    a = np_as_located_field(IDim, JDim)(np.ones((size, size)))
    lower_i = 1
    upper_i = 9
    lower_j = 4
    upper_j = 6

    @field_operator(backend=fieldview_backend)
    def fieldop_domain(a: Field[[IDim, JDim], float64]) -> Field[[IDim, JDim], float64]:
        return a + a

    @program
    def program_domain(
        a: Field[[IDim, JDim], float64],
        lower_i: int64,
        upper_i: int64,
        lower_j: int64,
        upper_j: int64,
    ):
        fieldop_domain(
            a,
            out=a,
            domain={IDim: (lower_i, upper_i // 1), JDim: (lower_j**1, upper_j)},
        )

    program_domain(a, lower_i, upper_i, lower_j, upper_j, offset_provider={})

    expected = np.asarray(a)
    expected[1:9, 4:6] = 1 + 1

    assert np.allclose(expected, a)


def test_domain_input_bounds_1(fieldview_backend):
    size = 10
    a = np_as_located_field(IDim, JDim)(np.ones((size, size)) * 2)
    lower_i = 1
    upper_i = 9
    lower_j = 4
    upper_j = 6

    @field_operator(backend=fieldview_backend)
    def fieldop_domain(a: Field[[IDim, JDim], float64]) -> Field[[IDim, JDim], float64]:
        return a * a

    @program
    def program_domain(
        a: Field[[IDim, JDim], float64],
        lower_i: int64,
        upper_i: int64,
        lower_j: int64,
        upper_j: int64,
    ):
        fieldop_domain(
            a,
            out=a,
            domain={IDim: (1 * lower_i, upper_i + 0), JDim: (lower_j - 0, upper_j)},
        )

    program_domain(a, lower_i, upper_i, lower_j, upper_j, offset_provider={})

    expected = np.asarray(a)
    expected[1:9, 4:6] = 2 * 2

    assert np.allclose(expected, a)


def test_domain_tuple(fieldview_backend):
    size = 10
    a = np_as_located_field(IDim, JDim)(np.ones((size, size)))
    b = np_as_located_field(IDim, JDim)(np.ones((size, size)))

    @field_operator(backend=fieldview_backend)
    def fieldop_domain_tuple(
        a: Field[[IDim, JDim], float64]
    ) -> tuple[Field[[IDim, JDim], float64], Field[[IDim, JDim], float64]]:
        return (a + a, a)

    @program
    def program_domain_tuple(a: Field[[IDim, JDim], float64], b: Field[[IDim, JDim], float64]):
        fieldop_domain_tuple(a, out=(b, a), domain={IDim: (1, 9), JDim: (4, 6)})

    program_domain_tuple(a, b, offset_provider={})

    expected = np.asarray(a)
    expected[1:9, 4:6] = 1 + 1

    assert np.allclose(np.asarray(a), a)
    assert np.allclose(expected, b)


def test_where_k_offset(fieldview_backend):
    if fieldview_backend == gtfn_cpu.run_gtfn:
        pytest.skip("IndexFields are not supported yet.")
    size = 10
    KDim = Dimension("K", kind=DimensionKind.VERTICAL)
    Koff = FieldOffset("Koff", source=KDim, target=(KDim,))
    a = np_as_located_field(IDim, KDim)(np.ones((size, size)))
    out = np_as_located_field(IDim, KDim)(np.zeros((size, size)))
    k_index = index_field(KDim)

    @field_operator(backend=fieldview_backend)
    def fieldop_where_k_offset(
        a: Field[[IDim, KDim], float64],
        k_index: Field[[KDim], int64],
    ) -> Field[[IDim, KDim], float64]:
        return where(k_index > 0, a(Koff[-1]), 2.0)

    fieldop_where_k_offset(a, k_index, out=out, offset_provider={"Koff": KDim})

    expected = np.where(np.arange(0, size, 1)[np.newaxis, :] > 0.0, a, 2.0)

    assert np.allclose(np.asarray(out), expected)


def test_undefined_symbols():
    from functional.ffront.foast_passes.type_deduction import FieldOperatorTypeDeductionError

    with pytest.raises(FieldOperatorTypeDeductionError, match="Undeclared symbol"):

        @field_operator
        def return_undefined():
            return undefined_symbol
