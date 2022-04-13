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

from functional.ffront.fbuiltins import Field, FieldOffset, float64, neighbor_sum
from functional.ffront.foast_to_itir import FieldOperatorLowering
from functional.ffront.func_to_foast import FieldOperatorParser
from functional.iterator import ir as itir
from functional.iterator.backends import roundtrip
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


def make_domain(dim_name: str, lower: int, upper: int) -> itir.FunCall:
    return itir.FunCall(
        fun=itir.SymRef(id="domain"),
        args=[
            itir.FunCall(
                fun=itir.SymRef(id="named_range"),
                args=[
                    itir.AxisLiteral(value=dim_name),
                    itir.IntLiteral(value=lower),
                    itir.IntLiteral(value=upper),
                ],
            )
        ],
    )


def closure_from_fop(
    node: itir.FunctionDefinition, out_name: str, domain: itir.FunCall
) -> itir.StencilClosure:
    return itir.StencilClosure(
        stencil=itir.SymRef(id=node.id),
        inputs=[itir.SymRef(id=sym.id) for sym in node.params],
        output=itir.SymRef(id=out_name),
        domain=domain,
    )


# TODO(tehrengruber): dim and size are implicitly given by out_names. Get values from there
def fencil_from_fop(
    node: itir.FunctionDefinition, out_name: str, dim: CartesianAxis, size: int
) -> itir.FencilDefinition:
    domain = make_domain(dim.value, 0, size)
    closure = closure_from_fop(node, out_name=out_name, domain=domain)
    return itir.FencilDefinition(
        id=node.id + "_fencil",
        function_definitions=[node],
        params=[itir.Sym(id=inp.id) for inp in closure.inputs] + [itir.Sym(id=closure.output.id)],
        closures=[closure],
    )


# TODO(tehrengruber): dim and size are implicitly given bys out_names. Get values from there
def fencil_from_function(
    func, dim: CartesianAxis, size: int, out_name: str = "foo"
) -> itir.FencilDefinition:
    return fencil_from_fop(
        node=FieldOperatorLowering.apply(FieldOperatorParser.apply_to_function(func)),
        out_name=out_name,
        dim=dim,
        size=size,
    )


DimsType = TypeVar("DimsType")
DType = TypeVar("DType")

IDim = CartesianAxis("IDim")


def test_copy():
    size = 10
    a = np_as_located_field(IDim)(np.ones((size)))
    b = np_as_located_field(IDim)(np.zeros((size)))

    def copy(inp: Field[[IDim], float64]):
        return inp

    fencil = fencil_from_function(copy, dim=IDim, size=size)

    roundtrip.executor(fencil, a, b, offset_provider={})

    assert np.allclose(a, b)


@pytest.mark.skip(reason="no lowering for returning a tuple of fields exists yet.")
def test_multicopy():
    size = 10
    a = np_as_located_field(IDim)(np.ones((size)))
    b = np_as_located_field(IDim)(np.ones((size)) * 3)
    c = np_as_located_field(IDim)(np.zeros((size)))
    d = np_as_located_field(IDim)(np.zeros((size)))

    def multicopy(inp1: Field[[IDim], float64], inp2: Field[[IDim], float64]):
        return inp1, inp2

    fencil = fencil_from_function(multicopy, dim=IDim, size=size)
    roundtrip.executor(fencil, a, b, (c, d), offset_provider={})

    assert np.allclose(a, c)
    assert np.allclose(b, d)


def test_arithmetic():
    size = 10
    a = np_as_located_field(IDim)(np.ones((size)))
    b = np_as_located_field(IDim)(np.ones((size)) * 2)
    c = np_as_located_field(IDim)(np.zeros((size)))

    def arithmetic(inp1: Field[[IDim], float64], inp2: Field[[IDim], float64]):
        return inp1 + inp2

    fencil = fencil_from_function(arithmetic, dim=IDim, size=size)
    roundtrip.executor(fencil, a, b, c, offset_provider={})

    assert np.allclose(a.array() + b.array(), c)


def test_bit_logic():
    size = 10
    a = np_as_located_field(IDim)(np.full((size), True))
    b_data = np.full((size), True)
    b_data[5] = False
    b = np_as_located_field(IDim)(b_data)
    c = np_as_located_field(IDim)(np.full((size), False))

    def bit_and(inp1: Field[[IDim], bool], inp2: Field[[IDim], bool]):
        return inp1 & inp2

    fencil = fencil_from_function(bit_and, dim=IDim, size=size)
    roundtrip.executor(fencil, a, b, c, offset_provider={})

    assert np.allclose(a.array() & b.array(), c)


def test_unary_neg():
    size = 10
    a = np_as_located_field(IDim)(np.ones((size)))
    b = np_as_located_field(IDim)(np.zeros((size)))

    def uneg(inp: Field[[IDim], int]):
        return -inp

    fencil = fencil_from_function(uneg, dim=IDim, size=size)
    roundtrip.executor(fencil, a, b, offset_provider={})

    assert np.allclose(b, np.full((size), -1))


def test_shift():
    size = 10
    Ioff = FieldOffset("Ioff", source=IDim, target=[IDim])
    a = np_as_located_field(IDim)(np.arange(size + 1))
    b = np_as_located_field(IDim)(np.zeros((size)))

    def shift_by_one(inp: Field[[IDim], float64]):
        return inp(Ioff[1])

    fencil = fencil_from_function(shift_by_one, dim=IDim, size=size)
    roundtrip.executor(fencil, a, b, offset_provider={"Ioff": IDim})

    assert np.allclose(b.array(), np.arange(1, 11))


def test_fold_shifts():
    """Shifting the result of an addition should work."""
    size = 10
    Ioff = FieldOffset("Ioff", source=IDim, target=[IDim])
    a = np_as_located_field(IDim)(np.arange(size + 1))
    b = np_as_located_field(IDim)(np.ones((size + 2)) * 2)
    c = np_as_located_field(IDim)(np.zeros((size)))

    def auto_lift(inp1: Field[[IDim], float64], inp2: Field[[IDim], float64]):
        tmp = inp1 + inp2(Ioff[1])
        return tmp(Ioff[1])

    fencil = fencil_from_function(auto_lift, dim=IDim, size=size)
    roundtrip.executor(fencil, a, b, c, offset_provider={"Ioff": IDim})

    assert np.allclose(a[1:] + b[2:], c)


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
    )


def test_reduction_execution(reduction_setup):
    """Testing a trivial neighbor sum."""
    rs = reduction_setup
    V2EDim = rs.V2EDim
    V2E = rs.V2E

    def reduction(edge_f: Field[[rs.Edge], "float64"]):
        return neighbor_sum(edge_f(V2E), axis=V2EDim)

    fencil = fencil_from_function(reduction, dim=rs.Vertex, size=rs.size)
    roundtrip.executor(
        fencil,
        rs.inp,
        rs.out,
        offset_provider=rs.offset_provider,
    )

    ref = np.sum(rs.v2e_table, axis=1)
    assert np.allclose(ref, rs.out)


def test_reduction_expression(reduction_setup):
    """Test reduction with an expression directly inside the call."""
    rs = reduction_setup
    V2EDim = rs.V2EDim
    V2E = rs.V2E

    def reduce_expr(edge_f: Field[[rs.Edge], "float64"]):
        tmp_nbh_tup = edge_f(V2E), edge_f(V2E)
        tmp_nbh = tmp_nbh_tup[0]
        return neighbor_sum(-edge_f(V2E) * tmp_nbh, axis=V2EDim)

    fencil = fencil_from_function(reduce_expr, dim=rs.Vertex, size=rs.size)
    roundtrip.executor(
        fencil,
        rs.inp,
        rs.out,
        offset_provider=rs.offset_provider,
    )

    ref = np.sum(-(rs.v2e_table**2), axis=1)
    assert np.allclose(ref, rs.out.array())
