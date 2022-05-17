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

from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import Field, FieldOffset, float64, int32, neighbor_sum
from functional.iterator.embedded import (
    NeighborTableOffsetProvider,
    index_field,
    np_as_located_field,
)
from functional.iterator.runtime import CartesianAxis


def debug_itir(tree):
    """Compare tree snippets while debugging."""
    from devtools import debug

    from eve.codegen import format_python_source
    from functional.iterator.backends.roundtrip import EmbeddedDSL

    debug(format_python_source(EmbeddedDSL.apply(tree)))


DimsType = TypeVar("DimsType")
DType = TypeVar("DType")

IDim = CartesianAxis("IDim")


def test_copy():
    size = 10
    a = np_as_located_field(IDim)(np.ones((size)))
    b = np_as_located_field(IDim)(np.zeros((size)))

    @field_operator(backend="roundtrip")
    def copy(inp: Field[[IDim], float64]) -> Field[[IDim], float64]:
        return inp

    copy(a, out=b, offset_provider={})

    assert np.allclose(a, b)


@pytest.mark.skip(reason="no lowering for returning a tuple of fields exists yet.")
def test_multicopy():
    size = 10
    a = np_as_located_field(IDim)(np.ones((size)))
    b = np_as_located_field(IDim)(np.ones((size)) * 3)
    c = np_as_located_field(IDim)(np.zeros((size)))
    d = np_as_located_field(IDim)(np.zeros((size)))

    @field_operator(backend="roundtrip")
    def multicopy(
        inp1: Field[[IDim], float64], inp2: Field[[IDim], float64]
    ) -> tuple[Field[[IDim], float64], Field[[IDim], float64]]:
        return inp1, inp2

    multicopy(a, b, out=(c, d), offset_provider={})

    assert np.allclose(a, c)
    assert np.allclose(b, d)


def test_arithmetic():
    size = 10
    a = np_as_located_field(IDim)(np.ones((size)))
    b = np_as_located_field(IDim)(np.ones((size)) * 2)
    c = np_as_located_field(IDim)(np.zeros((size)))

    @field_operator(backend="roundtrip")
    def arithmetic(
        inp1: Field[[IDim], float64], inp2: Field[[IDim], float64]
    ) -> Field[[IDim], float64]:
        return (inp1 + inp2) * 2.0

    arithmetic(a, b, out=c, offset_provider={})

    assert np.allclose((a.array() + b.array()) * 2.0, c)


def test_bit_logic():
    size = 10
    a = np_as_located_field(IDim)(np.full((size), True))
    b_data = np.full((size), True)
    b_data[5] = False
    b = np_as_located_field(IDim)(b_data)
    c = np_as_located_field(IDim)(np.full((size), False))

    @field_operator(backend="roundtrip")
    def bit_and(inp1: Field[[IDim], bool], inp2: Field[[IDim], bool]) -> Field[[IDim], bool]:
        return inp1 & inp2 & True

    bit_and(a, b, out=c, offset_provider={})

    assert np.allclose(a.array() & b.array(), c)


def test_unary_neg():
    size = 10
    a = np_as_located_field(IDim)(np.ones((size), dtype=int32))
    b = np_as_located_field(IDim)(np.zeros((size), dtype=int32))

    @field_operator(backend="roundtrip")
    def uneg(inp: Field[[IDim], int32]) -> Field[[IDim], int32]:
        return -inp

    uneg(a, out=b, offset_provider={})

    assert np.allclose(b, np.full((size), -1, dtype=int32))


def test_shift():
    size = 10
    Ioff = FieldOffset("Ioff", source=IDim, target=[IDim])
    a = np_as_located_field(IDim)(np.arange(size + 1))
    b = np_as_located_field(IDim)(np.zeros((size)))

    @field_operator
    def shift_by_one(inp: Field[[IDim], float64]) -> Field[[IDim], float64]:
        return inp(Ioff[1])

    @program(backend="roundtrip")
    def fencil(inp: Field[[IDim], float64], out: Field[[IDim], float64]) -> None:
        shift_by_one(inp, out=out)

    fencil(a, b, offset_provider={"Ioff": IDim})

    assert np.allclose(b.array(), np.arange(1, 11))


def test_fold_shifts():
    """Shifting the result of an addition should work."""
    size = 10
    Ioff = FieldOffset("Ioff", source=IDim, target=[IDim])
    a = np_as_located_field(IDim)(np.arange(size + 1))
    b = np_as_located_field(IDim)(np.ones((size + 2)) * 2)
    c = np_as_located_field(IDim)(np.zeros((size)))

    @field_operator
    def auto_lift(
        inp1: Field[[IDim], float64], inp2: Field[[IDim], float64]
    ) -> Field[[IDim], float64]:
        tmp = inp1 + inp2(Ioff[1])
        return tmp(Ioff[1])

    @program(backend="roundtrip")
    def fencil(
        inp1: Field[[IDim], float64], inp2: Field[[IDim], float64], out: Field[[IDim], float64]
    ) -> None:
        auto_lift(inp1, inp2, out=out)

    fencil(a, b, c, offset_provider={"Ioff": IDim})

    assert np.allclose(a[1:] + b[2:], c)


def test_tuples():
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

    @program(backend="roundtrip")
    def fencil(
        inp1: Field[[IDim], float64], inp2: Field[[IDim], float64], out: Field[[IDim], float64]
    ) -> None:
        tuples(inp1, inp2, out=out)

    fencil(a, b, c, offset_provider={})

    assert np.allclose((a.array() * 1.3 + b.array() * 5.0) * 3.4, c)


def test_broadcasting():
    Edge = CartesianAxis("Edge")
    K = CartesianAxis("K")

    size = 10
    ksize = 5
    a = np_as_located_field(Edge, K)(np.ones((size, ksize)))
    b = np_as_located_field(K)(np.ones((ksize)) * 2)
    c = np_as_located_field(Edge, K)(np.zeros((size, ksize)))

    @field_operator(backend="roundtrip")
    def broadcast(
        inp1: Field[[Edge, K], float64], inp2: Field[[K], float64]
    ) -> Field[[Edge, K], float64]:
        return inp1 / inp2

    broadcast(a, b, out=c, offset_provider={})

    assert np.allclose((a.array() / b.array()), c)


@pytest.fixture
def reduction_setup():

    size = 9
    edge = CartesianAxis("Edge")
    vertex = CartesianAxis("Vertex")
    v2edim = CartesianAxis("V2E")

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

    yield namedtuple(
        "ReductionSetup",
        ["size", "Edge", "Vertex", "V2EDim", "V2E", "inp", "out", "v2e_table", "offset_provider"],
    )(
        size=9,
        Edge=edge,
        Vertex=vertex,
        V2EDim=v2edim,
        V2E=FieldOffset("V2E", source=edge, target=(vertex, v2edim)),
        inp=index_field(edge),
        out=np_as_located_field(vertex)(np.zeros([size])),
        offset_provider={"V2E": NeighborTableOffsetProvider(v2e_arr, vertex, edge, 4)},
        v2e_table=v2e_arr,
    )  # type: ignore


def test_reduction_execution(reduction_setup):
    """Testing a trivial neighbor sum."""
    rs = reduction_setup
    Edge = rs.Edge
    Vertex = rs.Vertex
    V2EDim = rs.V2EDim
    V2E = rs.V2E

    @field_operator
    def reduction(edge_f: Field[[Edge], "float64"]) -> Field[[Vertex], float64]:
        return neighbor_sum(edge_f(V2E), axis=V2EDim)

    @program(backend="roundtrip")
    def fencil(edge_f: Field[[Edge], float64], out: Field[[Vertex], float64]) -> None:
        reduction(edge_f, out=out)

    fencil(rs.inp, rs.out, offset_provider=rs.offset_provider)

    ref = np.sum(rs.v2e_table, axis=1)
    assert np.allclose(ref, rs.out)


def test_reduction_expression(reduction_setup):
    """Test reduction with an expression directly inside the call."""
    rs = reduction_setup
    Vertex = rs.Vertex
    Edge = rs.Edge
    V2EDim = rs.V2EDim
    V2E = rs.V2E

    @field_operator
    def reduce_expr(edge_f: Field[[Edge], "float64"]) -> Field[[Vertex], float64]:
        tmp_nbh_tup = edge_f(V2E), edge_f(V2E)
        tmp_nbh = tmp_nbh_tup[0]
        return neighbor_sum(-edge_f(V2E) * tmp_nbh * 2.0, axis=V2EDim)

    @program(backend="roundtrip")
    def fencil(edge_f: Field[[Edge], float64], out: Field[[Vertex], float64]) -> None:
        reduce_expr(edge_f, out=out)

    fencil(rs.inp, rs.out, offset_provider=rs.offset_provider)

    ref = np.sum(-(rs.v2e_table**2) * 2, axis=1)
    assert np.allclose(ref, rs.out.array())


def test_scalar_arg():
    """Test scalar argument being turned into 0-dim field."""
    Vertex = CartesianAxis("Vertex")
    size = 5
    inp = 5.0
    out = np_as_located_field(Vertex)(np.zeros([size]))

    @field_operator(backend="roundtrip")
    def scalar_arg(scalar_inp: float64) -> Field[[Vertex], float64]:
        return scalar_inp + 1.0

    scalar_arg(inp, out=out, offset_provider={})

    ref = np.full([size], 6.0)
    assert np.allclose(ref, out.array())


def test_scalar_arg_with_field():
    Edge = CartesianAxis("Edge")
    EdgeOffset = FieldOffset("EdgeOffset", source=Edge, target=[Edge])
    size = 5
    inp = index_field(Edge)
    factor = 3
    out = np_as_located_field(Edge)(np.zeros([size]))

    @field_operator
    def scalar_and_field_args(
        inp: Field[[Edge], float64], factor: float64
    ) -> Field[[Edge], float64]:
        tmp = factor * inp
        return tmp(EdgeOffset[1])

    @program(backend="roundtrip")
    def fencil(out: Field[[Edge], float64], inp: Field[[Edge], float64], factor: float64) -> None:
        scalar_and_field_args(inp, factor, out=out)

    fencil(out, inp, factor, offset_provider={"EdgeOffset": Edge})

    ref = np.arange(1, size + 1) * factor
    assert np.allclose(ref, out.array())
