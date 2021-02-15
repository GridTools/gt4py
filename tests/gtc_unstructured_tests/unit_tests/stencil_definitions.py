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
    Edge,
    Field,
    Local,
    Connectivity,
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
V2E = types.new_class("V2E", (Connectivity[Vertex, Edge, 7, False],))

valid_stencils = ["copy", "edge_reduction", "sparse_ex", "nested", "temporary_field", "fvm_nabla"]


def copy(field_in: Field[Vertex, dtype], field_out: Field[Vertex, dtype]):
    with computation(FORWARD), interval(0, None), location(Vertex) as v:
        field_in = field_out


def edge_reduction(v2e: V2E, edge_field: Field[Edge, dtype], vertex_field: Field[Vertex, dtype]):
    with computation(FORWARD), interval(0, None), location(Edge) as e:
        edge_field = 0.5 * sum(vertex_field[v] for v in v2e[e])


def sparse_ex(
    e2v: E2V, edge_field: Field[Edge, dtype], sparse_field: Field[Edge, Local[Vertex], dtype]
):
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
    sign: Field[Vertex, Local[Edge], dtype],
):
    with computation(FORWARD), interval(0, None):
        with location(Edge) as e:
            zavg = 0.5 * sum(pp[v] for v in e2v[e])
            zavg = sum(pp[v] for v in e2v[e])
            zavgS_MXX = S_MXX * zavg
            zavgS_MYY = S_MYY * zavg
        with location(Vertex) as v:
            pnabla_MXX = sum(zavgS_MXX[e] * sign[v, e] for e in v2e[v])
            pnabla_MYY = sum(zavgS_MYY[e] * sign[v, e] for e in v2e[v])
            pnabla_MXX = pnabla_MXX / vol
            pnabla_MYY = pnabla_MYY / vol
