# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest


pytest.importorskip("atlas4py")

import gt4py.next as gtx
from gt4py.next.iterator import library
from gt4py.next.iterator.atlas_utils import AtlasTable
from gt4py.next.iterator.builtins import (
    deref,
    eq,
    if_,
    lift,
    list_get,
    make_tuple,
    named_range,
    neighbors,
    or_,
    reduce,
    tuple_get,
    unstructured_domain,
    as_fieldop,
)
from gt4py.next.iterator.runtime import set_at, fendef, fundef, offset

from next_tests.integration_tests.multi_feature_tests.fvm_nabla_setup import (
    assert_close,
    nabla_setup,
)
from next_tests.unit_tests.conftest import program_processor, run_processor


Vertex = gtx.Dimension("Vertex")
Edge = gtx.Dimension("Edge")
V2EDim = gtx.Dimension("V2E", kind=gtx.DimensionKind.LOCAL)

V2E = offset("V2E")
E2V = offset("E2V")


@fundef
def compute_zavgS(pp, S_M):
    pp_neighs = neighbors(E2V, pp)
    zavg = 0.5 * (list_get(0, pp_neighs) + list_get(1, pp_neighs))
    return deref(S_M) * zavg


@fendef
def compute_zavgS_fencil(n_edges, out, pp, S_M):
    domain = unstructured_domain(named_range(Edge, 0, n_edges))
    set_at(as_fieldop(compute_zavgS, domain)(pp, S_M), domain, out)


@fundef
def compute_pnabla(pp, S_M, sign, vol):
    zavgS = lift(compute_zavgS)(pp, S_M)
    pnabla_M = library.dot(neighbors(V2E, zavgS), deref(sign))
    return pnabla_M / deref(vol)


@fundef
def pnabla(pp, S_MXX, S_MYY, sign, vol):
    return make_tuple(compute_pnabla(pp, S_MXX, sign, vol), compute_pnabla(pp, S_MYY, sign, vol))


@fundef
def compute_zavgS2(pp, S_M):
    pp_neighs = neighbors(E2V, pp)
    zavg = 0.5 * (list_get(0, pp_neighs) + list_get(1, pp_neighs))
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
    pnabla_M = tuple_dot(neighbors(V2E, zavgS), deref(sign))
    return make_tuple(tuple_get(0, pnabla_M) / deref(vol), tuple_get(1, pnabla_M) / deref(vol))


@fendef
def nabla(n_nodes, out, pp, S_MXX, S_MYY, sign, vol):
    domain = unstructured_domain(named_range(Vertex, 0, n_nodes))
    set_at(as_fieldop(pnabla, domain)(pp, S_MXX, S_MYY, sign, vol), domain, out)


@pytest.mark.requires_atlas
def test_compute_zavgS(program_processor):
    program_processor, validate = program_processor
    setup = nabla_setup(allocator=None)

    zavgS = gtx.as_field([Edge], np.zeros((setup.edges_size)))

    run_processor(
        compute_zavgS_fencil,
        program_processor,
        setup.edges_size,
        zavgS,
        setup.input_field,
        setup.S_fields[0],
        offset_provider={"E2V": setup.edges2node_connectivity},
    )

    if validate:
        assert_close(-199755464.25741270, np.min(zavgS.asnumpy()))
        assert_close(388241977.58389181, np.max(zavgS.asnumpy()))

    run_processor(
        compute_zavgS_fencil,
        program_processor,
        setup.edges_size,
        zavgS,
        setup.input_field,
        setup.S_fields[1],
        offset_provider={"E2V": setup.edges2node_connectivity},
    )
    if validate:
        assert_close(-1000788897.3202186, np.min(zavgS.asnumpy()))
        assert_close(1000788897.3202186, np.max(zavgS.asnumpy()))


@fendef
def compute_zavgS2_fencil(n_edges, out, pp, S_M):
    domain = unstructured_domain(named_range(Edge, 0, n_edges))
    set_at(as_fieldop(compute_zavgS2, domain)(pp, S_M), domain, out)


@pytest.mark.requires_atlas
def test_compute_zavgS2(program_processor):
    program_processor, validate = program_processor
    setup = nabla_setup(allocator=None)

    zavgS = (
        gtx.as_field([Edge], np.zeros((setup.edges_size))),
        gtx.as_field([Edge], np.zeros((setup.edges_size))),
    )

    run_processor(
        compute_zavgS2_fencil,
        program_processor,
        setup.edges_size,
        zavgS,
        setup.input_field,
        setup.S_fields,
        offset_provider={"E2V": setup.edges2node_connectivity},
    )

    if validate:
        assert_close(-199755464.25741270, np.min(zavgS[0].asnumpy()))
        assert_close(388241977.58389181, np.max(zavgS[0].asnumpy()))

        assert_close(-1000788897.3202186, np.min(zavgS[1].asnumpy()))
        assert_close(1000788897.3202186, np.max(zavgS[1].asnumpy()))


@pytest.mark.requires_atlas
def test_nabla(program_processor):
    program_processor, validate = program_processor

    setup = nabla_setup(allocator=None)

    S_MXX, S_MYY = setup.S_fields

    pnabla_MXX = gtx.as_field([Vertex], np.zeros((setup.nodes_size)))
    pnabla_MYY = gtx.as_field([Vertex], np.zeros((setup.nodes_size)))

    run_processor(
        nabla,
        program_processor,
        setup.nodes_size,
        (pnabla_MXX, pnabla_MYY),
        setup.input_field,
        S_MXX,
        S_MYY,
        setup.sign_field,
        setup.vol_field,
        offset_provider={
            "E2V": setup.edges2node_connectivity,
            "V2E": setup.nodes2edge_connectivity,
        },
    )

    if validate:
        assert_close(-3.5455427772566003e-003, np.min(pnabla_MXX.asnumpy()))
        assert_close(3.5455427772565435e-003, np.max(pnabla_MXX.asnumpy()))
        assert_close(-3.3540113705465301e-003, np.min(pnabla_MYY.asnumpy()))
        assert_close(3.3540113705465301e-003, np.max(pnabla_MYY.asnumpy()))


@fendef
def nabla2(n_nodes, out, pp, S, sign, vol):
    domain = unstructured_domain(named_range(Vertex, 0, n_nodes))
    set_at(as_fieldop(compute_pnabla2, domain)(pp, S, sign, vol), domain, out)


@pytest.mark.requires_atlas
def test_nabla2(program_processor):
    program_processor, validate = program_processor
    setup = nabla_setup(allocator=None)

    pnabla_MXX = gtx.as_field([Vertex], np.zeros((setup.nodes_size)))
    pnabla_MYY = gtx.as_field([Vertex], np.zeros((setup.nodes_size)))

    run_processor(
        nabla2,
        program_processor,
        setup.nodes_size,
        (pnabla_MXX, pnabla_MYY),
        setup.input_field,
        setup.S_fields,
        setup.sign_field,
        setup.vol_field,
        offset_provider={
            "E2V": setup.edges2node_connectivity,
            "V2E": setup.nodes2edge_connectivity,
        },
    )

    if validate:
        assert_close(-3.5455427772566003e-003, np.min(pnabla_MXX.asnumpy()))
        assert_close(3.5455427772565435e-003, np.max(pnabla_MXX.asnumpy()))
        assert_close(-3.3540113705465301e-003, np.min(pnabla_MYY.asnumpy()))
        assert_close(3.3540113705465301e-003, np.max(pnabla_MYY.asnumpy()))


@fundef
def sign(node_indices, is_pole_edge):
    def impl(node_indices2, is_pole_edge):
        return if_(
            or_(
                deref(is_pole_edge),
                eq(deref(node_indices), list_get(0, neighbors(E2V, node_indices2))),
            ),
            1.0,
            -1.0,
        )

    return neighbors(V2E, lift(impl)(node_indices, is_pole_edge))


@fundef
def compute_pnabla_sign(pp, S_M, vol, node_index, is_pole_edge):
    zavgS = lift(compute_zavgS)(pp, S_M)
    pnabla_M = library.dot(neighbors(V2E, zavgS), sign(node_index, is_pole_edge))

    return pnabla_M / deref(vol)


@fendef
def nabla_sign(n_nodes, out_MXX, out_MYY, pp, S_MXX, S_MYY, vol, node_index, is_pole_edge):
    # TODO replace by single stencil which returns tuple
    domain = unstructured_domain(named_range(Vertex, 0, n_nodes))
    set_at(
        as_fieldop(compute_pnabla_sign, domain)(pp, S_MXX, vol, node_index, is_pole_edge),
        domain,
        out_MXX,
    )
    set_at(
        as_fieldop(compute_pnabla_sign, domain)(pp, S_MYY, vol, node_index, is_pole_edge),
        domain,
        out_MYY,
    )


@pytest.mark.requires_atlas
def test_nabla_sign(program_processor):
    program_processor, validate = program_processor

    setup = nabla_setup(allocator=None)

    S_MXX, S_MYY = setup.S_fields

    pnabla_MXX = gtx.as_field([Vertex], np.zeros((setup.nodes_size)))
    pnabla_MYY = gtx.as_field([Vertex], np.zeros((setup.nodes_size)))

    run_processor(
        nabla_sign,
        program_processor,
        setup.nodes_size,
        pnabla_MXX,
        pnabla_MYY,
        setup.input_field,
        S_MXX,
        S_MYY,
        setup.vol_field,
        gtx.index_field(Vertex),
        setup.is_pole_edge_field,
        offset_provider={
            "E2V": setup.edges2node_connectivity,
            "V2E": setup.nodes2edge_connectivity,
        },
    )

    if validate:
        assert_close(-3.5455427772566003e-003, np.min(pnabla_MXX.asnumpy()))
        assert_close(3.5455427772565435e-003, np.max(pnabla_MXX.asnumpy()))
        assert_close(-3.3540113705465301e-003, np.min(pnabla_MYY.asnumpy()))
        assert_close(3.3540113705465301e-003, np.max(pnabla_MYY.asnumpy()))
