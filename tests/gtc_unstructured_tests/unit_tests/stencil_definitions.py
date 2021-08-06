# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2020, CSCS - Swiss National Supercomputing Center, ETH Zurich
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
# ignore flake8 error: local variable '...' is assigned to but never used
# flake8: noqa: F841
import types

from gtc_unstructured.frontend.gtscript import (
    FORWARD,
    Cell,
    Connectivity,
    Edge,
    Field,
    LocalField,
    SparseField,
    Vertex,
    computation,
    edges,
    interval,
    location,
    vertices,
)
from gtc_unstructured.irs import common


dtype = common.DataType.FLOAT64
E2V = types.new_class("E2V", (Connectivity[Edge, Vertex, 2, False],))
V2E = types.new_class("V2E", (Connectivity[Vertex, Edge, 7, True],))
C2C = types.new_class("C2C", (Connectivity[Cell, Cell, 4, False],))

valid_stencils = [
    "copy",
    "copy2",
    "edge_reduction",
    "sparse_ex",
    "nested",
    "temporary_field",
    "fvm_nabla",
    "weights",
    "sparse_field_assign",
    "native_functions",
]


def copy(field_in: Field[Vertex, dtype], field_out: Field[Vertex, dtype]):
    with computation(FORWARD), interval(0, None), location(Vertex) as v:
        field_out = field_in


def copy2(field_in: Field[Vertex, dtype], field_out: Field[Vertex, dtype]):
    with computation(FORWARD), interval(0, None), location(Vertex) as v:
        field_in[v] = field_out[v]


def edge_reduction(e2v: E2V, edge_field: Field[Edge, dtype], vertex_field: Field[Vertex, dtype]):
    with computation(FORWARD), interval(0, None), location(Edge) as e:
        edge_field = 0.5 * sum(vertex_field[v] for v in e2v[e])


def sparse_ex(e2v: E2V, edge_field: Field[Edge, dtype], sparse_field: SparseField[E2V, dtype]):
    with computation(FORWARD), interval(0, None), location(Edge) as e:
        edge_field = sum(sparse_field[e, v] for v in e2v[e])


def nested(f_1: Field[Edge, dtype], f_2: Field[Vertex, dtype], f_3: Field[Edge, dtype]):
    with computation(FORWARD), interval(0, None):
        with location(Edge) as e:
            f_1 = 1
        with location(Vertex) as v:
            f_2 = 2
    with computation(FORWARD), interval(0, None), location(Edge) as e:
        f_3 = 3


def temporary_field(out: Field[Vertex, dtype]):
    with computation(FORWARD), interval(0, None), location(Vertex) as e:
        tmp = 1
    with computation(FORWARD), interval(0, None), location(Vertex) as v:
        out = tmp


def fvm_nabla(
    v2e: V2E,
    e2v: E2V,
    S_MXX: Field[Edge, dtype],
    S_MYY: Field[Edge, dtype],
    pp: Field[Vertex, dtype],
    pnabla_MXX: Field[Vertex, dtype],
    pnabla_MYY: Field[Vertex, dtype],
    vol: Field[Vertex, dtype],
    sign: SparseField[V2E, dtype],
):
    with computation(FORWARD), interval(0, None):
        with location(Edge) as e:
            zavg = 0.5 * sum(pp[v] for v in e2v[e])
            zavgS_MXX = S_MXX * zavg
            zavgS_MYY = S_MYY * zavg
        with location(Vertex) as v:
            pnabla_MXX = sum(zavgS_MXX[e] * sign[v, e] for e in v2e[v])
            pnabla_MYY = sum(zavgS_MYY[e] * sign[v, e] for e in v2e[v])
            pnabla_MXX = pnabla_MXX / vol
            pnabla_MYY = pnabla_MYY / vol


def weights(e2v: E2V, in_field: Field[Vertex, dtype], out_field: Field[Edge, dtype]):
    with computation(FORWARD), interval(0, None):
        with location(Edge) as e:
            weights = LocalField[E2V, dtype]([2, 1])
            out_field = sum(in_field[v] * weights[e, v] for v in e2v[e])


def sparse_field_assign(
    e2v: E2V, in_sparse_field: SparseField[E2V, dtype], out_sparse_field: SparseField[E2V, dtype]
):
    with computation(FORWARD), interval(0, None):
        with location(Edge) as e:
            # TODO: maybe support slicing for lhs: out_sparse_field[e,:]
            out_sparse_field = (in_sparse_field[e, v] for v in e2v[e])
            # TODO: Fix silently generates invalid code
            # out_sparse_field = in_sparse_field


def native_functions(c2c: C2C, field_in: Field[Cell, dtype], field_out: Field[Cell, dtype]):
    with computation(FORWARD), location(Cell) as c1:
        field_out[c1] = (
            sqrt(field_in) + max(1, 2) + sum(max(field_in[c1], field_in[c2]) for c2 in c2c[c1])
        )
