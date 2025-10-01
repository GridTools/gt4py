# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import numpy as np
import copy

dace = pytest.importorskip("dace")
from dace.sdfg import nodes as dace_nodes, graph as dace_graph
from dace import data as dace_data
from dace.transformation import dataflow as dace_dftrafo

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
)

from . import util

import dace


def _create_simple_fusable_sdfg() -> tuple[
    dace.SDFG,
    dace.SDFGState,
    dace_nodes.MapEntry,
    dace_nodes.AccessNode,
    dace_nodes.AccessNode,
    dace_nodes.AccessNode,
    dace_graph.MultiConnectorEdge[dace.Memlet],
]:
    sdfg = dace.SDFG(util.unique_name(f"simple_fusable_sdfg"))
    state = sdfg.add_state(is_start_block=True)

    for name in "abc":
        sdfg.add_array(
            name,
            shape=(10, 10),
            dtype=dace.float64,
            transient=False,
        )
    sdfg.arrays["b"].transient = True

    a, b, c = (state.add_access(name) for name in "abc")

    state.add_mapped_tasklet(
        "map1",
        map_ranges={
            "i1": "0:10",
            "j1": "0:10",
        },
        inputs={"__in": dace.Memlet("a[i1, j1]")},
        code="__out = __in + 1.0",
        outputs={"__out": dace.Memlet("b[i1, j1]")},
        input_nodes={a},
        output_nodes={b},
        external_edges=True,
    )
    _, me2, _ = state.add_mapped_tasklet(
        "map2",
        map_ranges={
            "i2": "0:10",
            "j2": "0:10",
        },
        inputs={"__in": dace.Memlet("b[i2, j2]")},
        code="__out = __in + 1.1",
        outputs={"__out": dace.Memlet("c[i2, j2]")},
        input_nodes={b},
        output_nodes={c},
        external_edges=True,
    )
    sdfg.validate()

    return sdfg, state, me2, a, b, c, next(iter(state.out_edges(me2)))


def test_inline_fuse_simple_case():
    sdfg, state, second_map_entry, a, b, c, edge_to_replace = _create_simple_fusable_sdfg()
    assert util.count_nodes(sdfg, dace_nodes.NestedSDFG) == 0

    nsdfg, output_node = gtx_transformations.inline_dataflow_into_map(
        sdfg=sdfg,
        state=state,
        edge=edge_to_replace,
    )
    sdfg.validate()

    assert all(oedge.dst is nsdfg for oedge in state.out_edges(second_map_entry))
    assert state.out_degree(b) == 0
    assert str(nsdfg.symbol_mapping["i1"]) == "i2"
    assert str(nsdfg.symbol_mapping["j1"]) == "j2"
    assert util.count_nodes(sdfg, dace_nodes.NestedSDFG) == 1

    nested_ac = util.count_nodes(nsdfg.sdfg, dace_nodes.AccessNode, True)
    assert len(nested_ac) == 2
    assert {a.data, output_node.data} == {ac.data for ac in nested_ac}

    nested_tlet = util.count_nodes(nsdfg.sdfg, dace_nodes.Tasklet, True)
    assert len(nested_tlet) == 1
    assert nested_tlet[0].code.as_string == "__out = (__in + 1.0)"

    ref, res = util.make_sdfg_args(sdfg)
    ref["c"] = ref["a"] + 2.1

    util.compile_and_run_sdfg(sdfg, **res)
    assert util.compare_sdfg_res(ref, res)


def _make_laplap_sdfg_1() -> tuple[
    dace.SDFG,
    dace.SDFGState,
    dace_nodes.AccessNode,
    dace_nodes.MapEntry,
    dace_graph.MultiConnectorEdge[dace.Memlet],
    dace_graph.MultiConnectorEdge[dace.Memlet],
    dace_nodes.Tasklet,
]:
    sdfg = dace.SDFG(util.unique_name(f"laplap1"))
    state = sdfg.add_state(is_start_block=True)

    sdfg.add_array(
        "a",
        shape=(10,),
        dtype=dace.float64,
        transient=False,
    )
    sdfg.add_array(
        "lap",
        shape=(8,),
        dtype=dace.float64,
        transient=True,
    )
    sdfg.add_array(
        "laplap",
        shape=(6,),
        dtype=dace.float64,
        transient=False,
    )
    lap = state.add_access("lap")

    state.add_mapped_tasklet(
        "lap",
        map_ranges={"__i": "0:8"},
        inputs={
            "__in1": dace.Memlet("a[__i]"),
            "__in2": dace.Memlet("a[__i + 2]"),
        },
        code="__out = __in2 - __in1",
        outputs={"__out": dace.Memlet("lap[__i]")},
        output_nodes={lap},
        external_edges=True,
    )
    laplap_tlet, laplap_me, _ = state.add_mapped_tasklet(
        "laplap",
        map_ranges={"__i": "0:6"},
        inputs={
            "__in1": dace.Memlet("lap[__i]"),
            "__in2": dace.Memlet("lap[__i + 2]"),
        },
        code="__out = __in2 - __in1",
        outputs={"__out": dace.Memlet("laplap[__i]")},
        input_nodes={lap},
        external_edges=True,
    )
    sdfg.validate()

    edge_m1 = next(iter(oedge for oedge in state.out_edges(laplap_me) if oedge.dst_conn == "__in1"))
    edge_p1 = next(iter(oedge for oedge in state.out_edges(laplap_me) if oedge.dst_conn == "__in2"))
    return sdfg, state, lap, laplap_me, edge_m1, edge_p1, laplap_tlet


def test_laplap1():
    sdfg, state, lap, laplap_me, edge_m1, edge_p1, laplap_tlet = _make_laplap_sdfg_1()

    ref, res = util.make_sdfg_args(sdfg)
    util.compile_and_run_sdfg(sdfg, **ref)

    nsdfg_m1, output_node_m1 = gtx_transformations.inline_dataflow_into_map(
        sdfg=sdfg,
        state=state,
        edge=edge_m1,
    )
    sdfg.validate()
    nsdfg_p1, output_node_p1 = gtx_transformations.inline_dataflow_into_map(
        sdfg=sdfg,
        state=state,
        edge=edge_p1,
    )

    assert state.out_degree(lap) == 0
    assert state.in_degree(laplap_tlet) == 2

    assert state.out_degree(output_node_m1) == 1
    assert next(iter(state.in_edges_by_connector(laplap_tlet, "__in1"))).src is output_node_m1
    assert str(nsdfg_m1.symbol_mapping["__i"]) == "__i"

    assert state.out_degree(output_node_p1) == 1
    assert next(iter(state.in_edges_by_connector(laplap_tlet, "__in2"))).src is output_node_p1
    assert str(nsdfg_p1.symbol_mapping["__i"]) == "__i + 2"

    util.compile_and_run_sdfg(sdfg, **res)
    assert util.compare_sdfg_res(ref=ref, res=res)
