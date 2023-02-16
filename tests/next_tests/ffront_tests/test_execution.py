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
import warnings
from functools import reduce

import numpy as np
import pytest as pytest

from gt4py.next.ffront.decorator import field_operator, program, scan_operator
from gt4py.next.ffront.fbuiltins import (
    Dimension,
    Field,
    FieldOffset,
    astype,
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
from gt4py.next.ffront.foast_passes.type_deduction import FieldOperatorTypeDeductionError
from gt4py.next.iterator.builtins import float32
from gt4py.next.iterator.embedded import (
    NeighborTableOffsetProvider,
    index_field,
    np_as_located_field,
)
from gt4py.next.program_processors.runners import gtfn_cpu, roundtrip

from .ffront_test_utils import *


def test_copy(fieldview_backend):
    a_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    b_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))

    @field_operator(backend=fieldview_backend)
    def copy(inp: Field[[IDim], float64]) -> Field[[IDim], float64]:
        field_tuple = (inp, inp)
        field_0 = field_tuple[0]
        field_1 = field_tuple[1]
        return field_0

    copy(a_I_float, out=b_I_float, offset_provider={})

    assert np.allclose(a_I_float, b_I_float)


@pytest.mark.skip(reason="no lowering for returning a tuple of fields exists yet.")
def test_multicopy(fieldview_backend):
    a_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    b_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    out_I_float = np_as_located_field(IDim)(np.zeros((size), dtype=float64))
    out_I_float_1 = np_as_located_field(IDim)(np.zeros((size), dtype=float64))

    @field_operator(backend=fieldview_backend)
    def multicopy(
        inp1: Field[[IDim], float64], inp2: Field[[IDim], float64]
    ) -> tuple[Field[[IDim], float64], Field[[IDim], float64]]:
        return inp1, inp2

    assert np.allclose(a_I_float, out_I_float)
    assert np.allclose(b_I_float, out_I_float_1)


def test_cartesian_shift(fieldview_backend):
    a = np_as_located_field(IDim)(np.arange(size + 1, dtype=np.float64))
    out_I_float = np_as_located_field(IDim)(np.zeros((size), dtype=float64))

    @field_operator
    def shift_by_one(inp: Field[[IDim], float64]) -> Field[[IDim], float64]:
        return inp(Ioff[1])

    @program(backend=fieldview_backend)
    def fencil(inp: Field[[IDim], float64], out: Field[[IDim], float64]) -> None:
        shift_by_one(inp, out=out)

    fencil(a, out_I_float, offset_provider={"Ioff": IDim})

    assert np.allclose(out_I_float.array(), np.arange(1, 11))


def test_unstructured_shift(reduction_setup, fieldview_backend):
    Vertex = reduction_setup.Vertex
    Edge = reduction_setup.Edge
    E2V = reduction_setup.E2V

    a = np_as_located_field(Vertex)(np.zeros(reduction_setup.num_vertices))
    b = np_as_located_field(Edge)(np.zeros(reduction_setup.num_edges))

    @field_operator(backend=fieldview_backend)
    def shift_by_one(inp: Field[[Vertex], float64]) -> Field[[Edge], float64]:
        return inp(E2V[0])

    shift_by_one(a, out=b, offset_provider={"E2V": reduction_setup.offset_provider["E2V"]})

    ref = np.asarray(a)[reduction_setup.offset_provider["E2V"].table[slice(0, None), 0]]

    assert np.allclose(b, ref)


def test_fold_shifts(fieldview_backend):
    """Shifting the result of an addition should work."""
    a = np_as_located_field(IDim)(np.arange(size + 2, dtype=np.float64))
    b = np_as_located_field(IDim)(np.ones((size + 3)) * 2)
    out_I_float = np_as_located_field(IDim)(np.zeros((size), dtype=float64))

    @field_operator
    def auto_lift(
        inp1: Field[[IDim], float64], inp2: Field[[IDim], float64]
    ) -> Field[[IDim], float64]:
        return (inp1 + inp2(Ioff[1]))(Ioff[2])

    @program(backend=fieldview_backend)
    def fencil(
        inp1: Field[[IDim], float64], inp2: Field[[IDim], float64], out: Field[[IDim], float64]
    ) -> None:
        auto_lift(inp1, inp2, out=out)

    fencil(a, b, out_I_float, offset_provider={"Ioff": IDim})

    assert np.allclose(a[2:] + b[3:], out_I_float)


def test_tuples(fieldview_backend):
    size = 10
    a = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    b = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    c = np_as_located_field(IDim)(np.zeros((size), dtype=float64))

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


def test_scalar_arg(fieldview_backend):
    """Test scalar argument being turned into 0-dim field."""
    inp = 5.0
    out = np_as_located_field(Vertex)(np.zeros([size]))

    @field_operator(backend=fieldview_backend)
    def scalar_arg(scalar_inp: float64) -> Field[[Vertex], float64]:
        return broadcast(scalar_inp + 1.0, (Vertex,))

    scalar_arg(inp, out=out, offset_provider={})

    ref = np.full([size], 6.0)
    assert np.allclose(ref, out.array())


def test_nested_scalar_arg(fieldview_backend):
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
    if fieldview_backend in [gtfn_cpu.run_gtfn, gtfn_cpu.run_gtfn_imperative]:
        pytest.skip("IndexFields and ConstantFields are not supported yet.")

    inp = index_field(Edge, dtype=float64)
    if fieldview_backend == gtfn_cpu.run_gtfn:
        warnings.warn(
            "IndexFields not supported in gtfn backend. Using a memory backed field instead."
        )
        # TODO(tehrengruber): if we choose the wrong size here the gtfn backend
        #  will happily executy, but give wrong results. we should implement
        #  checks for such cases at some point.
        inp = np_as_located_field(Edge)(np.array([inp[i] for i in range(size + 1)]))
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


def test_scalar_in_domain_spec_and_fo_call(fieldview_backend):
    if fieldview_backend in [gtfn_cpu.run_gtfn, gtfn_cpu.run_gtfn_imperative]:
        pytest.skip(
            "Scalar arguments not supported to be used in both domain specification "
            "and as an argument to a field operator."
        )

    size = 10
    out = np_as_located_field(Vertex)(np.zeros(10, dtype=int))

    @field_operator
    def foo(size: int) -> Field[[Vertex], int]:
        return broadcast(size, (Vertex,))

    @program(backend=fieldview_backend)
    def bar(size: int, out: Field[[Vertex], int]):
        foo(size, out=out, domain={Vertex: (0, size)})

    bar(size, out, offset_provider={})

    assert (out.array() == size).all()


def test_scalar_scan():
    size = 10
    KDim = Dimension("K", kind=DimensionKind.VERTICAL)
    qc = np_as_located_field(IDim, KDim)(np.zeros((size, size)))
    scalar = 1.0
    expected = np.full((size, size), np.arange(start=1, stop=11, step=1).astype(float64))

    @scan_operator(axis=KDim, forward=True, init=(0.0))
    def _scan_scalar(carry: float, qc_in: float, scalar: float) -> float:
        qc = qc_in + carry + scalar
        return qc

    @program
    def scan_scalar(qc: Field[[IDim, KDim], float], scalar: float):
        _scan_scalar(qc, scalar, out=qc)

    scan_scalar(qc, scalar, offset_provider={})
    assert np.allclose(np.asarray(qc), expected)


def test_tuple_scalar_scan():
    size = 10
    KDim = Dimension("K", kind=DimensionKind.VERTICAL)
    qc = np_as_located_field(IDim, KDim)(np.zeros((size, size)))
    tuple_scalar = (1.0, (1.0, 0.0))
    expected = np.full((size, size), np.arange(start=1, stop=11, step=1).astype(float64))

    @scan_operator(axis=KDim, forward=True, init=0.0)
    def _scan_tuple_scalar(
        state: float, qc_in: float, tuple_scalar: tuple[float, tuple[float, float]]
    ) -> float:
        return (qc_in + state + tuple_scalar[1][0] + tuple_scalar[1][1]) / tuple_scalar[0]

    @field_operator
    def scan_tuple_scalar(
        qc: Field[[IDim, KDim], float], tuple_scalar: tuple[float, tuple[float, float]]
    ) -> Field[[IDim, KDim], float]:
        return _scan_tuple_scalar(qc, tuple_scalar)

    scan_tuple_scalar(qc, tuple_scalar, out=qc, offset_provider={})
    assert np.allclose(np.asarray(qc), expected)


def test_astype_int(fieldview_backend):
    size = 10
    b_float_64 = np_as_located_field(IDim)(np.ones((size), dtype=np.float64))
    c_int64 = np_as_located_field(IDim)(np.ones((size,), dtype=np.int64))
    out_int_64 = np_as_located_field(IDim)(np.zeros((size,), dtype=np.int64))

    @field_operator(backend=fieldview_backend)
    def astype_fieldop_int(b: Field[[IDim], float64]) -> Field[[IDim], int64]:
        d = astype(b, int64)
        return d

    astype_fieldop_int(b_float_64, out=out_int_64, offset_provider={})
    assert np.allclose(c_int64.array(), out_int_64)


def test_astype_bool(fieldview_backend):
    b_float_64 = np_as_located_field(IDim)(np.ones((size), dtype=np.float64))
    c_bool = np_as_located_field(IDim)(np.ones((size,), dtype=bool))
    out_bool = np_as_located_field(IDim)(np.zeros((size,), dtype=bool))

    @field_operator(backend=fieldview_backend)
    def astype_fieldop_bool(b: Field[[IDim], float64]) -> Field[[IDim], bool]:
        d = astype(b, bool)
        return d

    astype_fieldop_bool(b_float_64, out=out_bool, offset_provider={})
    assert np.allclose(c_bool, out_bool)


def test_astype_float(fieldview_backend):
    c_int64 = np_as_located_field(IDim)(np.ones((size,), dtype=np.int64))
    c_int32 = np_as_located_field(IDim)(np.ones((size,), dtype=np.int32))
    out_int_32 = np_as_located_field(IDim)(np.zeros((size,), dtype=np.int32))

    @field_operator(backend=fieldview_backend)
    def astype_fieldop_float(b: Field[[IDim], int64]) -> Field[[IDim], int32]:
        d = astype(b, int32)
        return d

    astype_fieldop_float(c_int64, out=out_int_32, offset_provider={})
    assert np.allclose(c_int32.array(), out_int_32)


def test_nested_tuple_return(fieldview_backend):
    if fieldview_backend == gtfn_cpu.run_gtfn:
        pytest.skip("Tuple return values are not supported in gtfn yet.")

    a_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    b_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    out_I_float = np_as_located_field(IDim)(np.zeros((size), dtype=float64))

    @field_operator
    def pack_tuple(
        a: Field[[IDim], float64], b: Field[[IDim], float64]
    ) -> tuple[Field[[IDim], float64], tuple[Field[[IDim], float64], Field[[IDim], float64]]]:
        return (a, (a, b))

    @field_operator
    def combine(a: Field[[IDim], float64], b: Field[[IDim], float64]) -> Field[[IDim], float64]:
        packed = pack_tuple(a, b)
        return packed[0] + packed[1][0] + packed[1][1]

    combine(a_I_float, b_I_float, out=out_I_float, offset_provider={})

    assert np.allclose(2 * a_I_float.array() + b_I_float.array(), out_I_float)


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
    E2VDim = rs.E2VDim
    V2E = rs.V2E
    E2V = rs.E2V

    num_vertices = rs.num_vertices
    num_edges = rs.num_edges

    # TODO(tehrengruber): use different values per location
    a = np_as_located_field(Edge)(np.random.randn(num_edges))
    b = np_as_located_field(Vertex)(np.random.randn(num_vertices))
    out = np_as_located_field(Edge)(np.zeros((num_edges,)))

    @field_operator
    def reduce_tuple_element(
        edge_field: Field[[Edge], float64], vertex_field: Field[[Vertex], float64]
    ) -> Field[[Edge], float64]:
        tup = edge_field(V2E), vertex_field
        red = neighbor_sum(tup[0] + vertex_field, axis=V2EDim)
        return neighbor_sum(red(E2V), axis=E2VDim)

    reduce_tuple_element(a, b, out=out, offset_provider=rs.offset_provider)

    red = np.sum(np.asarray(a)[rs.v2e_table] + np.asarray(b)[:, np.newaxis], axis=1)
    expected = np.sum(red[rs.e2v_table], axis=1)

    assert np.allclose(expected, out)


def test_tuple_arg(fieldview_backend):
    a_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    b_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    out_I_float = np_as_located_field(IDim)(np.zeros((size), dtype=float64))

    if fieldview_backend in [gtfn_cpu.run_gtfn, gtfn_cpu.run_gtfn_imperative]:
        pytest.skip("Tuple arguments are not supported in gtfn yet.")

    @field_operator(backend=fieldview_backend)
    def unpack_tuple(
        inp: tuple[tuple[Field[[IDim], float64], Field[[IDim], float64]], Field[[IDim], float64]]
    ) -> Field[[IDim], float64]:
        return 3.0 * inp[0][0] + inp[0][1] + inp[1]

    unpack_tuple(((a_I_float, b_I_float), a_I_float), out=out_I_float, offset_provider={})

    assert np.allclose(3 * a_I_float.array() + b_I_float.array() + a_I_float.array(), out_I_float)


@pytest.mark.parametrize("forward", [True, False])
def test_fieldop_from_scan(fieldview_backend, forward):
    # TODO(tehrengruber): broken fix
    if fieldview_backend in [gtfn_cpu.run_gtfn, gtfn_cpu.run_gtfn_imperative]:
        pytest.xfail("gtfn does not yet support scan pass.")
    init = 1.0
    out = np_as_located_field(KDim)(np.zeros((size,)))
    expected = np.arange(init + 1.0, init + 1.0 + size, 1)
    if not forward:
        expected = np.flip(expected)

    @field_operator
    def add(carry: float, foo: float) -> float:
        return carry + foo

    @scan_operator(axis=KDim, forward=forward, init=init, backend=fieldview_backend)
    def simple_scan_operator(carry: float) -> float:
        return add(carry, 1.0)

    simple_scan_operator(out=out, offset_provider={})

    assert np.allclose(expected, out)


def test_solve_triag(fieldview_backend):
    if fieldview_backend in [gtfn_cpu.run_gtfn, gtfn_cpu.run_gtfn_imperative]:
        pytest.xfail("gtfn does not yet support scan pass.")

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


def test_ternary_operator(fieldview_backend):
    a_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    b_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    out_I_float = np_as_located_field(IDim)(np.zeros((size), dtype=float64))

    left = 2.0
    right = 3.0

    @field_operator(backend=fieldview_backend)
    def ternary_field_op(
        a: Field[[IDim], float], b: Field[[IDim], float], left: float, right: float
    ) -> Field[[IDim], float]:
        return a if left < right else b

    ternary_field_op(a_I_float, b_I_float, left, right, out=out_I_float, offset_provider={})
    e = np.asarray(a_I_float) if left < right else np.asarray(b_I_float)
    np.allclose(e, out_I_float)

    @field_operator(backend=fieldview_backend)
    def ternary_field_op_scalars(left: float, right: float) -> Field[[IDim], float]:
        return broadcast(3.0, (IDim,)) if left > right else broadcast(4.0, (IDim,))

    ternary_field_op_scalars(left, right, out=out_I_float, offset_provider={})
    e = np.full(e.shape, 3.0) if left > right else e
    np.allclose(e, out_I_float)


def test_ternary_operator_tuple(fieldview_backend):
    if fieldview_backend in [gtfn_cpu.run_gtfn, gtfn_cpu.run_gtfn_imperative]:
        pytest.skip("Tuple return values are not supported in gtfn yet.")

    a_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    b_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))
    out_I_float = np_as_located_field(IDim)(np.zeros((size), dtype=float64))
    out_I_float_1 = np_as_located_field(IDim)(np.zeros((size), dtype=float64))

    left = 2.0
    right = 3.0

    @field_operator(backend=fieldview_backend)
    def ternary_field_op(
        a: Field[[IDim], float], b: Field[[IDim], float], left: float, right: float
    ) -> tuple[Field[[IDim], float], Field[[IDim], float]]:
        return (a, b) if left < right else (b, a)

    ternary_field_op(
        a_I_float, b_I_float, left, right, out=(out_I_float, out_I_float_1), offset_provider={}
    )

    e, f = (
        (np.asarray(a_I_float), np.asarray(b_I_float))
        if left < right
        else (np.asarray(b_I_float), np.asarray(a_I_float))
    )
    np.allclose(e, out_I_float)
    np.allclose(f, out_I_float_1)


def test_ternary_builtin_neighbor_sum(fieldview_backend, reduction_setup):
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

    @field_operator(backend=fieldview_backend)
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


def test_ternary_scan(fieldview_backend):
    init = 0.0
    a_float = 4
    a = np_as_located_field(KDim)(a_float * np.ones((size,)))
    out = np_as_located_field(KDim)(np.zeros((size,)))
    expected = np.asarray([i if i <= a_float else a_float + 1 for i in range(1, size + 1)])

    @scan_operator(backend=fieldview_backend, axis=KDim, forward=True, init=init)
    def simple_scan_operator(carry: float, a: float) -> float:
        return carry if carry > a else carry + 1.0

    simple_scan_operator(a, out=out, offset_provider={})

    assert np.allclose(expected, out)


@pytest.mark.parametrize("forward", [True, False])
def test_scan_nested_tuple_output(fieldview_backend, forward):
    if fieldview_backend in [gtfn_cpu.run_gtfn, gtfn_cpu.run_gtfn_imperative]:
        pytest.xfail("gtfn does not yet support scan pass or tuple out arguments.")

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
    if fieldview_backend in [gtfn_cpu.run_gtfn, gtfn_cpu.run_gtfn_imperative]:
        pytest.xfail("gtfn does not yet support scan pass or tuple arguments.")

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
    a_I_float = np_as_located_field(IDim)(np.random.randn(size).astype("float64"))

    @field_operator
    def fieldop_with_docstring(a: Field[[IDim], float64]) -> Field[[IDim], float64]:
        """My docstring."""
        return a

    @program
    def test_docstring(a: Field[[IDim], float64]):
        """My docstring."""
        fieldop_with_docstring(a, out=a)

    test_docstring(a_I_float, offset_provider={})


def test_domain(fieldview_backend):
    a_IJ_float = np_as_located_field(IDim, JDim)(np.ones((size, size), dtype=float64))

    @field_operator(backend=fieldview_backend)
    def fieldop_domain(a: Field[[IDim, JDim], float64]) -> Field[[IDim, JDim], float64]:
        return a + a

    @program
    def program_domain(a: Field[[IDim, JDim], float64]):
        fieldop_domain(a, out=a, domain={IDim: (1, 9), JDim: (4, 6)})

    program_domain(a_IJ_float, offset_provider={})

    expected = np.asarray(a_IJ_float)
    expected[1:9, 4:6] = 1 + 1

    assert np.allclose(expected, a_IJ_float)


def test_domain_input_bounds(fieldview_backend):
    if fieldview_backend in [gtfn_cpu.run_gtfn, gtfn_cpu.run_gtfn_imperative]:
        pytest.skip("FloorDiv not fully supported in gtfn.")
    a_IJ_float = np_as_located_field(IDim, JDim)(np.ones((size, size), dtype=float64))

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

    program_domain(a_IJ_float, lower_i, upper_i, lower_j, upper_j, offset_provider={})

    expected = np.asarray(a_IJ_float)
    expected[1:9, 4:6] = 1 + 1

    assert np.allclose(expected, a_IJ_float)


def test_domain_input_bounds_1(fieldview_backend):
    a_IJ_float = np_as_located_field(IDim, JDim)(np.ones((size, size), dtype=float64))

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

    program_domain(a_IJ_float, lower_i, upper_i, lower_j, upper_j, offset_provider={})

    expected = np.asarray(a_IJ_float)
    expected[1:9, 4:6] = 2 * 2

    assert np.allclose(expected, a_IJ_float)


def test_domain_tuple(fieldview_backend):
    a_IJ_float = np_as_located_field(IDim, JDim)(np.ones((size, size), dtype=float64))
    b2d_float = a_IJ_float

    @field_operator(backend=fieldview_backend)
    def fieldop_domain_tuple(
        a: Field[[IDim, JDim], float64]
    ) -> tuple[Field[[IDim, JDim], float64], Field[[IDim, JDim], float64]]:
        return (a + a, a)

    @program
    def program_domain_tuple(a: Field[[IDim, JDim], float64], b: Field[[IDim, JDim], float64]):
        fieldop_domain_tuple(a, out=(b, a), domain={IDim: (1, 9), JDim: (4, 6)})

    program_domain_tuple(a_IJ_float, b2d_float, offset_provider={})

    expected = np.asarray(a_IJ_float)
    expected[1:9, 4:6] = 1 + 1

    assert np.allclose(np.asarray(a_IJ_float), a_IJ_float)
    assert np.allclose(expected, b2d_float)


def test_where_k_offset(fieldview_backend):
    a = np_as_located_field(IDim, KDim)(np.ones((size, size)))
    out = np_as_located_field(IDim, KDim)(np.zeros((size, size)))
    k_index = index_field(KDim)
    if fieldview_backend in [gtfn_cpu.run_gtfn, gtfn_cpu.run_gtfn_imperative]:
        warnings.warn(
            "IndexFields not supported in gtfn backend. Using a memory backed field instead."
        )
        # TODO(tehrengruber): if we choose the wrong size here the gtfn backend
        #  will happily executy, but give wrong results. we should implement
        #  checks for such cases at some point.
        k_index = np_as_located_field(KDim)(np.array([k_index.field_getitem(i) for i in range(size)]))

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
    with pytest.raises(FieldOperatorTypeDeductionError, match="Undeclared symbol"):

        @field_operator
        def return_undefined():
            return undefined_symbol


def test_zero_dims_fields(fieldview_backend):
    if fieldview_backend in [gtfn_cpu.run_gtfn, gtfn_cpu.run_gtfn_imperative]:
        pytest.skip("Implicit broadcast are not supported yet.")

    inp = np_as_located_field()(np.array(1.0))
    out = np_as_located_field()(np.array(0.0))

    @field_operator(backend=fieldview_backend)
    def implicit_broadcast_scalar(inp: Field[[], float]):
        return inp

    implicit_broadcast_scalar(inp, out=out, offset_provider={})
    assert np.allclose(out, np.array(1.0))


def test_implicit_broadcast_mixed_dims(fieldview_backend):
    if fieldview_backend == gtfn_cpu.run_gtfn:
        pytest.skip("Implicit broadcast are not supported yet.")

    input1 = np_as_located_field(IDim)(np.ones((10,)))
    inp = np_as_located_field()(np.array(1.0))
    out = np_as_located_field(IDim)(np.ones((10,)))

    @field_operator(backend=fieldview_backend)
    def fieldop_implicit_broadcast(
        zero_dim_inp: Field[[], float], inp: Field[[IDim], float], scalar: float
    ) -> Field[[IDim], float]:
        return inp + zero_dim_inp * scalar

    @field_operator(backend=fieldview_backend)
    def fieldop_implicit_broadcast_2(inp: Field[[IDim], float]) -> Field[[IDim], float]:
        fi = fieldop_implicit_broadcast(1.0, inp, 1.0)
        return fi

    fieldop_implicit_broadcast_2(input1, out=out, offset_provider={})
    assert np.allclose(out, np.asarray(inp) * 2)


def test_tuple_unpacking(fieldview_backend):
    size = 10
    inp = np_as_located_field(IDim)(np.ones((size)))
    out1 = np_as_located_field(IDim)(np.ones((size)))
    out2 = np_as_located_field(IDim)(np.ones((size)))
    out3 = np_as_located_field(IDim)(np.ones((size)))
    out4 = np_as_located_field(IDim)(np.ones((size)))

    @field_operator
    def unpack(
        inp: Field[[IDim], float64],
    ) -> tuple[
        Field[[IDim], float64],
        Field[[IDim], float64],
        Field[[IDim], float64],
        Field[[IDim], float64],
    ]:
        a, b, c, d = (inp + 2.0, inp + 3.0, inp + 5.0, inp + 7.0)
        return a, b, c, d

    unpack(inp, out=(out1, out2, out3, out4), offset_provider={})

    arr = inp.array()

    assert np.allclose(out1, arr + 2.0)
    assert np.allclose(out2, arr + 3.0)
    assert np.allclose(out3, arr + 5.0)
    assert np.allclose(out4, arr + 7.0)


def test_tuple_unpacking_star_multi(fieldview_backend):
    size = 10
    inp = np_as_located_field(IDim)(np.ones((size)))
    out = tuple(np_as_located_field(IDim)(np.ones(size) * i) for i in range(3 * 4))

    OutType = tuple[
        Field[[IDim], float64],
        Field[[IDim], float64],
        Field[[IDim], float64],
        Field[[IDim], float64],
        Field[[IDim], float64],
        Field[[IDim], float64],
        Field[[IDim], float64],
        Field[[IDim], float64],
        Field[[IDim], float64],
        Field[[IDim], float64],
        Field[[IDim], float64],
        Field[[IDim], float64],
    ]

    @field_operator
    def unpack(
        inp: Field[[IDim], float64],
    ) -> OutType:
        *a, a2, a3 = (inp, inp + 1.0, inp + 2.0, inp + 3.0)
        b1, *b, b3 = (inp + 4.0, inp + 5.0, inp + 6.0, inp + 7.0)
        c1, c2, *c = (inp + 8.0, inp + 9.0, inp + 10.0, inp + 11.0)

        return (a[0], a[1], a2, a3, b1, b[0], b[1], b3, c1, c2, c[0], c[1])

    unpack(inp, out=out, offset_provider={})

    for i in range(3 * 4):
        assert np.allclose(out[i], inp.array() + i)


def test_tuple_unpacking_too_many_values(fieldview_backend):
    with pytest.raises(
        FieldOperatorTypeDeductionError,
        match=(r"Could not deduce type: Too many values to unpack \(expected 3\)"),
    ):

        @field_operator
        def _star_unpack() -> tuple[int, float64, int]:
            a, b, c = (1, 2.0, 3, 4, 5, 6, 7.0)
            return a, b, c


def test_tuple_unpacking_too_many_values(fieldview_backend):
    with pytest.raises(
        FieldOperatorTypeDeductionError, match=(r"Assignment value must be of type tuple!")
    ):

        @field_operator
        def _invalid_unpack() -> tuple[int, float64, int]:
            a, b, c = 1
            return a


def test_constant_closure_vars():
    from gt4py.eve.utils import FrozenNamespace

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
