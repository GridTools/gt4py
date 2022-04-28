# GT4Py New Semantic Model - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.  GT4Py
# New Semantic Model is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or any later version.
# See the LICENSE.txt file at the top-level directory of this distribution for
# a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pytest


pytest.importorskip("atlas4py")

from functional.iterator import library
from functional.iterator.atlas_utils import AtlasTable
from functional.iterator.builtins import *
from functional.iterator.embedded import (
    NeighborTableOffsetProvider,
    index_field,
    np_as_located_field,
)
from functional.iterator.runtime import *

from .fvm_nabla_setup import assert_close, nabla_setup


Vertex = CartesianAxis("Vertex")
Edge = CartesianAxis("Edge")

V2E = offset("V2E")
E2V = offset("E2V")


@fundef
def compute_zavgS(pp, S_M):
    zavg = 0.5 * (deref(shift(E2V, 0)(pp)) + deref(shift(E2V, 1)(pp)))
    # zavg = 0.5 * reduce(lambda a, b: a + b, 0)(shift(E2V)(pp))
    # zavg = 0.5 * library.sum()(shift(E2V)(pp))
    return deref(S_M) * zavg


@fendef
def compute_zavgS_fencil(
    n_edges,
    out,
    pp,
    S_M,
):
    closure(
        domain(named_range(Edge, 0, n_edges)),
        compute_zavgS,
        out,
        [pp, S_M],
    )


@fundef
def compute_pnabla(pp, S_M, sign, vol):
    zavgS = lift(compute_zavgS)(pp, S_M)
    # pnabla_M = reduce(lambda a, b, c: a + b * c, 0)(shift(V2E)(zavgS), sign)
    # pnabla_M = library.sum(lambda a, b: a * b)(shift(V2E)(zavgS), sign)
    pnabla_M = library.dot(shift(V2E)(zavgS), sign)
    return pnabla_M / deref(vol)


@fundef
def pnabla(pp, S_MXX, S_MYY, sign, vol):
    return make_tuple(compute_pnabla(pp, S_MXX, sign, vol), compute_pnabla(pp, S_MYY, sign, vol))


@fundef
def compute_zavgS2(pp, S_M):
    zavg = 0.5 * (deref(shift(E2V, 0)(pp)) + deref(shift(E2V, 1)(pp)))
    s = deref(S_M)
    return make_tuple(tuple_get(0, s) * zavg, tuple_get(1, s) * zavg)


@fundef
def tuple_dot_fun(acc, zavgS, sign):
    return make_tuple(
        tuple_get(0, acc) + tuple_get(0, zavgS) * sign,
        tuple_get(1, acc) + tuple_get(1, zavgS) * sign,
    )


@fundef
def tuple_dot(a, b):
    return reduce(tuple_dot_fun, make_tuple(0.0, 0.0))(a, b)


@fundef
def compute_pnabla2(pp, S_M, sign, vol):
    zavgS = lift(compute_zavgS2)(pp, S_M)
    pnabla_M = tuple_dot(shift(V2E)(zavgS), sign)
    return make_tuple(tuple_get(0, pnabla_M) / deref(vol), tuple_get(1, pnabla_M) / deref(vol))


@fendef
def nabla(
    n_nodes,
    out,
    pp,
    S_MXX,
    S_MYY,
    sign,
    vol,
):
    closure(
        domain(named_range(Vertex, 0, n_nodes)),
        pnabla,
        out,
        [pp, S_MXX, S_MYY, sign, vol],
    )


def test_compute_zavgS(backend, use_tmps):
    if use_tmps:
        pytest.xfail("use_tmps currently only supported for cartesian")
    backend, validate = backend
    setup = nabla_setup()

    pp = np_as_located_field(Vertex)(setup.input_field)
    S_MXX, S_MYY = tuple(map(np_as_located_field(Edge), setup.S_fields))

    zavgS = np_as_located_field(Edge)(np.zeros((setup.edges_size)))

    e2v = NeighborTableOffsetProvider(AtlasTable(setup.edges2node_connectivity), Edge, Vertex, 2)

    compute_zavgS_fencil(
        setup.edges_size,
        zavgS,
        pp,
        S_MXX,
        offset_provider={"E2V": e2v},
        backend=backend,
        use_tmps=use_tmps,
    )

    if validate:
        assert_close(-199755464.25741270, min(zavgS))
        assert_close(388241977.58389181, max(zavgS))

    compute_zavgS_fencil(
        setup.edges_size,
        zavgS,
        pp,
        S_MYY,
        offset_provider={"E2V": e2v},
        backend=backend,
        use_tmps=use_tmps,
    )
    if validate:
        assert_close(-1000788897.3202186, min(zavgS))
        assert_close(1000788897.3202186, max(zavgS))


@fendef
def compute_zavgS2_fencil(
    n_edges,
    out,
    pp,
    S_M,
):
    closure(
        domain(named_range(Edge, 0, n_edges)),
        compute_zavgS2,
        out,
        [pp, S_M],
    )


def test_compute_zavgS2(backend, use_tmps):
    if use_tmps:
        pytest.xfail("use_tmps currently only supported for cartesian")
    backend, validate = backend
    setup = nabla_setup()

    pp = np_as_located_field(Vertex)(setup.input_field)

    S = np_as_located_field(Edge)(
        np.array([(a, b) for a, b in zip(*(setup.S_fields[0], setup.S_fields[1]))], dtype="d,d")
    )

    zavgS = (
        np_as_located_field(Edge)(np.zeros((setup.edges_size))),
        np_as_located_field(Edge)(np.zeros((setup.edges_size))),
    )

    e2v = NeighborTableOffsetProvider(AtlasTable(setup.edges2node_connectivity), Edge, Vertex, 2)

    compute_zavgS2_fencil(
        setup.edges_size,
        zavgS,
        pp,
        S,
        offset_provider={"E2V": e2v},
        backend=backend,
        use_tmps=use_tmps,
    )

    if validate:
        assert_close(-199755464.25741270, min(zavgS[0]))
        assert_close(388241977.58389181, max(zavgS[0]))

        assert_close(-1000788897.3202186, min(zavgS[1]))
        assert_close(1000788897.3202186, max(zavgS[1]))


def test_nabla(backend, use_tmps):
    if use_tmps:
        pytest.xfail("use_tmps currently only supported for cartesian")
    backend, validate = backend
    setup = nabla_setup()

    sign = np_as_located_field(Vertex, V2E)(setup.sign_field)
    pp = np_as_located_field(Vertex)(setup.input_field)
    S_MXX, S_MYY = tuple(map(np_as_located_field(Edge), setup.S_fields))
    vol = np_as_located_field(Vertex)(setup.vol_field)

    pnabla_MXX = np_as_located_field(Vertex)(np.zeros((setup.nodes_size)))
    pnabla_MYY = np_as_located_field(Vertex)(np.zeros((setup.nodes_size)))

    e2v = NeighborTableOffsetProvider(AtlasTable(setup.edges2node_connectivity), Edge, Vertex, 2)
    v2e = NeighborTableOffsetProvider(AtlasTable(setup.nodes2edge_connectivity), Vertex, Edge, 7)

    nabla(
        setup.nodes_size,
        (pnabla_MXX, pnabla_MYY),
        pp,
        S_MXX,
        S_MYY,
        sign,
        vol,
        offset_provider={"E2V": e2v, "V2E": v2e},
        backend=backend,
        use_tmps=use_tmps,
    )

    if validate:
        assert_close(-3.5455427772566003e-003, min(pnabla_MXX))
        assert_close(3.5455427772565435e-003, max(pnabla_MXX))
        assert_close(-3.3540113705465301e-003, min(pnabla_MYY))
        assert_close(3.3540113705465301e-003, max(pnabla_MYY))


@fendef
def nabla2(
    n_nodes,
    out,
    pp,
    S,
    sign,
    vol,
):
    closure(
        domain(named_range(Vertex, 0, n_nodes)),
        compute_pnabla2,
        out,
        [pp, S, sign, vol],
    )


def test_nabla2(backend, use_tmps):
    if use_tmps:
        pytest.xfail("use_tmps currently only supported for cartesian")
    backend, validate = backend
    setup = nabla_setup()

    sign = np_as_located_field(Vertex, V2E)(setup.sign_field)
    pp = np_as_located_field(Vertex)(setup.input_field)
    S_M = np_as_located_field(Edge)(
        np.array([(a, b) for a, b in zip(*(setup.S_fields[0], setup.S_fields[1]))], dtype="d,d")
    )
    vol = np_as_located_field(Vertex)(setup.vol_field)

    pnabla_MXX = np_as_located_field(Vertex)(np.zeros((setup.nodes_size)))
    pnabla_MYY = np_as_located_field(Vertex)(np.zeros((setup.nodes_size)))

    e2v = NeighborTableOffsetProvider(AtlasTable(setup.edges2node_connectivity), Edge, Vertex, 2)
    v2e = NeighborTableOffsetProvider(AtlasTable(setup.nodes2edge_connectivity), Vertex, Edge, 7)

    nabla2(
        setup.nodes_size,
        (pnabla_MXX, pnabla_MYY),
        pp,
        S_M,
        sign,
        vol,
        offset_provider={"E2V": e2v, "V2E": v2e},
        backend=backend,
        use_tmps=use_tmps,
    )

    if validate:
        assert_close(-3.5455427772566003e-003, min(pnabla_MXX))
        assert_close(3.5455427772565435e-003, max(pnabla_MXX))
        assert_close(-3.3540113705465301e-003, min(pnabla_MYY))
        assert_close(3.3540113705465301e-003, max(pnabla_MYY))


@fundef
def sign(node_indices, is_pole_edge):
    node_index = deref(node_indices)

    @fundef
    def sign_impl(node_index):
        def impl2(node_indices, is_pole_edge):
            return if_(
                or_(deref(is_pole_edge), eq(node_index, deref(shift(E2V, 0)(node_indices)))),
                1.0,
                -1.0,
            )

        return impl2

    return shift(V2E)(lift(sign_impl(node_index))(node_indices, is_pole_edge))


@fundef
def compute_pnabla_sign(pp, S_M, vol, node_index, is_pole_edge):
    zavgS = lift(compute_zavgS)(pp, S_M)
    pnabla_M = library.dot(shift(V2E)(zavgS), sign(node_index, is_pole_edge))

    return pnabla_M / deref(vol)


@fendef
def nabla_sign(n_nodes, out_MXX, out_MYY, pp, S_MXX, S_MYY, vol, node_index, is_pole_edge):
    # TODO replace by single stencil which returns tuple
    closure(
        domain(named_range(Vertex, 0, n_nodes)),
        compute_pnabla_sign,
        out_MXX,
        [pp, S_MXX, vol, node_index, is_pole_edge],
    )
    closure(
        domain(named_range(Vertex, 0, n_nodes)),
        compute_pnabla_sign,
        out_MYY,
        [pp, S_MYY, vol, node_index, is_pole_edge],
    )


def test_nabla_sign(backend, use_tmps):
    if use_tmps:
        pytest.xfail("use_tmps currently only supported for cartesian")

    backend, validate = backend
    setup = nabla_setup()

    # sign = np_as_located_field(Vertex, V2E)(setup.sign_field)
    is_pole_edge = np_as_located_field(Edge)(setup.is_pole_edge_field)
    pp = np_as_located_field(Vertex)(setup.input_field)
    S_MXX, S_MYY = tuple(map(np_as_located_field(Edge), setup.S_fields))
    vol = np_as_located_field(Vertex)(setup.vol_field)

    pnabla_MXX = np_as_located_field(Vertex)(np.zeros((setup.nodes_size)))
    pnabla_MYY = np_as_located_field(Vertex)(np.zeros((setup.nodes_size)))

    e2v = NeighborTableOffsetProvider(AtlasTable(setup.edges2node_connectivity), Edge, Vertex, 2)
    v2e = NeighborTableOffsetProvider(AtlasTable(setup.nodes2edge_connectivity), Vertex, Edge, 7)

    nabla_sign(
        setup.nodes_size,
        pnabla_MXX,
        pnabla_MYY,
        pp,
        S_MXX,
        S_MYY,
        vol,
        index_field(Vertex),
        is_pole_edge,
        offset_provider={"E2V": e2v, "V2E": v2e},
        backend=backend,
        use_tmps=use_tmps,
    )

    if validate:
        assert_close(-3.5455427772566003e-003, min(pnabla_MXX))
        assert_close(3.5455427772565435e-003, max(pnabla_MXX))
        assert_close(-3.3540113705465301e-003, min(pnabla_MYY))
        assert_close(3.3540113705465301e-003, max(pnabla_MYY))
