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
from functools import reduce
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

    ternary_field_op(a, b, left, right, out=(out_1, out_2), offset_provider={})

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


@pytest.mark.parametrize("forward", [True, False])
def test_scan_nested_tuple_output(fieldview_backend, forward):
    if fieldview_backend == gtfn_cpu.run_gtfn:
        pytest.xfail("gtfn does not yet support scan pass or tuple out arguments.")

    KDim = Dimension("K", kind=DimensionKind.VERTICAL)
    size = 10
    init = (1.0, (2.0, 3.0))
    out1, out2, out3 = (np_as_located_field(KDim)(np.zeros((size,))) for _ in range(3))
    expected = np.arange(1.0, 1.0 + size, 1)
    if not forward:
        expected = np.flip(expected)

    @scan_operator(axis=KDim, forward=forward, init=init, backend=fieldview_backend)
    def simple_scan_operator(
        carry: tuple[float, tuple[float, float]]
    ) -> tuple[float, tuple[float, float]]:
        return (carry[0] + 1.0, (carry[1][0] + 1.0, carry[1][1] + 1.0))

    simple_scan_operator(out=(out1, (out2, out3)), offset_provider={})

    assert np.allclose(expected + 1.0, out1)
    assert np.allclose(expected + 2.0, out2)
    assert np.allclose(expected + 3.0, out3)


@pytest.mark.parametrize("forward", [True, False])
def test_scan_nested_tuple_input(fieldview_backend, forward):
    if fieldview_backend == gtfn_cpu.run_gtfn:
        pytest.xfail("gtfn does not yet support scan pass or tuple arguments.")

    KDim = Dimension("K", kind=DimensionKind.VERTICAL)
    size = 10
    init = 1.0
    inp1 = np_as_located_field(KDim)(np.ones(size))
    inp2 = np_as_located_field(KDim)(np.arange(0.0, size, 1))
    out = np_as_located_field(KDim)(np.zeros((size,)))

    prev_levels_iterator = lambda i: range(i + 1) if forward else range(size - 1, i - 1, -1)
    expected = np.asarray(
        [
            reduce(lambda prev, i: prev + inp1[i] + inp2[i], prev_levels_iterator(i), init)
            for i in range(size)
        ]
    )

    @scan_operator(axis=KDim, forward=forward, init=init, backend=fieldview_backend)
    def simple_scan_operator(carry: float, a: tuple[float, float]) -> float:
        return carry + a[0] + a[1]

    simple_scan_operator((inp1, inp2), out=out, offset_provider={})

    assert np.allclose(expected, out)


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
        pytest.skip("FloorDiv not fully supported in gtfn.")

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


def test_constant_closure_vars():
    from eve.utils import FrozenNamespace

    constants = FrozenNamespace(
        PI=np.float32(3.142),
        E=np.float32(2.718),
    )

    @field_operator
    def consume_constants(input: Field[[IDim], np.float32]) -> Field[[IDim], np.float32]:
        return constants.PI * constants.E * input

    input = np_as_located_field(IDim)(np.ones((1,), dtype=np.float32))
    output = np_as_located_field(IDim)(np.zeros((1,), dtype=np.float32))
    consume_constants(input, out=output, offset_provider={})
    assert np.allclose(np.asarray(output), constants.PI * constants.E)
