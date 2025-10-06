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
from dace import data as dace_data, subsets as dace_sbs
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


def _make_laplap_sdfg(
    zero_based_index: bool,
) -> tuple[
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
        map_ranges={"__i": "0:8" if zero_based_index else "1:9"},
        inputs={
            "__in1": dace.Memlet(f"a[{'__i' if zero_based_index else '__i - 1'}]"),
            "__in2": dace.Memlet(f"a[{'__i + 2' if zero_based_index else '__i + 1'}]"),
        },
        code="__out = __in2 - __in1",
        outputs={"__out": dace.Memlet(f"lap[{'__i' if zero_based_index else '__i - 1'}]")},
        output_nodes={lap},
        external_edges=True,
    )
    laplap_tlet, laplap_me, _ = state.add_mapped_tasklet(
        "laplap",
        map_ranges={"__i": "0:6" if zero_based_index else "2:8"},
        inputs={
            "__in1": dace.Memlet(f"lap[{'__i' if zero_based_index else '__i - 1 - 1'}]"),
            "__in2": dace.Memlet(f"lap[{'__i + 2' if zero_based_index else '__i + 1 - 1'}]"),
        },
        code="__out = __in2 - __in1",
        outputs={"__out": dace.Memlet(f"laplap[{'__i' if zero_based_index else '__i - 2'}]")},
        input_nodes={lap},
        external_edges=True,
    )
    sdfg.validate()

    edge_m1 = next(iter(oedge for oedge in state.out_edges(laplap_me) if oedge.dst_conn == "__in1"))
    edge_p1 = next(iter(oedge for oedge in state.out_edges(laplap_me) if oedge.dst_conn == "__in2"))
    return sdfg, state, lap, laplap_me, edge_m1, edge_p1, laplap_tlet


@pytest.mark.parametrize("zero_based_index", [True, False])
def test_laplap(zero_based_index: bool):
    sdfg, state, lap, laplap_me, edge_m1, edge_p1, laplap_tlet = _make_laplap_sdfg(zero_based_index)

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
    assert str(nsdfg_m1.symbol_mapping["__i"]).replace(" ", "") == (
        "__i" if zero_based_index else "__i-1"
    )

    assert state.out_degree(output_node_p1) == 1
    assert next(iter(state.in_edges_by_connector(laplap_tlet, "__in2"))).src is output_node_p1
    assert str(nsdfg_p1.symbol_mapping["__i"]).replace(" ", "") == (
        "__i+2" if zero_based_index else "__i+1"
    )

    util.compile_and_run_sdfg(sdfg, **res)
    assert util.compare_sdfg_res(ref=ref, res=res)


def _make_multiple_value_read_sdfg(
    transmit_all: bool,
) -> tuple[
    dace.SDFG,
    dace.SDFGState,
    dace_nodes.AccessNode,
    dace_nodes.AccessNode,
    dace_nodes.MapEntry,
    dace_graph.MultiConnectorEdge[dace.Memlet],
]:
    sdfg = dace.SDFG(util.unique_name(f"multiple_value_generator"))
    state = sdfg.add_state(is_start_block=True)

    sdfg.add_array(
        "a",
        shape=(10,),
        dtype=dace.float64,
        transient=False,
    )
    sdfg.add_array(
        "b",
        shape=(10, 3),
        dtype=dace.float64,
        transient=True,
    )
    sdfg.add_array(
        "c",
        shape=(10,),
        dtype=dace.float64,
        transient=False,
    )
    sdfg.add_array(
        "z",
        shape=(3,),
        dtype=dace.float64,
        transient=True,
    )

    z_writeback_lower_bound = 0 if transmit_all else 1

    a, b, z = (state.add_access(name) for name in "abz")
    me1, mx1 = state.add_map("first_map", ndrange={"__i": "0:10"})
    me1_inner, mx1_inner = state.add_map(
        "first_map_inner",
        ndrange={"__k": f"{z_writeback_lower_bound}:3"},
        schedule=dace.dtypes.ScheduleType.Sequential,
    )
    tlet1_inner = state.add_tasklet(
        "inner_tasklet1",
        inputs={"__in"},
        outputs={"__out"},
        code="__out = __in + __k * 1.5",
    )

    state.add_edge(a, None, me1, "IN_a", dace.Memlet("a[0:10]"))
    state.add_edge(me1, "OUT_a", me1_inner, "IN_a", dace.Memlet("a[__i]"))
    state.add_edge(me1_inner, "OUT_a", tlet1_inner, "__in", dace.Memlet("a[__i]"))
    state.add_edge(tlet1_inner, "__out", mx1_inner, "IN_z", dace.Memlet("z[__k]"))
    state.add_edge(mx1_inner, "OUT_z", z, None, dace.Memlet(f"z[{z_writeback_lower_bound}:3]"))
    state.add_edge(
        z,
        None,
        mx1,
        "IN_b",
        dace.Memlet(f"b[__i, {z_writeback_lower_bound}:3] -> [{z_writeback_lower_bound}:3]"),
    )
    state.add_edge(mx1, "OUT_b", b, None, dace.Memlet(f"b[0:10, {z_writeback_lower_bound}:3]"))
    me1.add_scope_connectors("a")
    me1_inner.add_scope_connectors("a")
    mx1_inner.add_scope_connectors("z")
    mx1.add_scope_connectors("b")

    _, me2, _ = state.add_mapped_tasklet(
        "map2",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("b[__i, 1]")},
        code="__out = __in + 0.7",
        outputs={"__out": dace.Memlet("c[__i]")},
        input_nodes={b},
        external_edges=True,
    )
    edge_to_replace = next(iter(state.out_edges(me2)))

    sdfg.validate()

    return sdfg, state, z, b, me2, edge_to_replace


def test_multiple_value_exchange():
    sdfg, state, z, b, second_map_entry, edge_to_replace = _make_multiple_value_read_sdfg(
        transmit_all=True
    )

    ref, res = util.make_sdfg_args(sdfg)
    util.compile_and_run_sdfg(sdfg, **ref)

    nsdfg_node, output_node = gtx_transformations.inline_dataflow_into_map(
        sdfg=sdfg,
        state=state,
        edge=edge_to_replace,
    )

    sdfg.validate()

    assert state.out_degree(b) == 0
    assert sdfg.arrays[output_node.data].shape == (1, 3)
    assert sdfg.arrays[output_node.data].transient

    nsdfg = nsdfg_node.sdfg
    nstate = nsdfg.start_block
    inner_z = next(iter(dnode for dnode in nstate.data_nodes() if dnode.data == "z"))
    inner_output_name = next(iter(nsdfg_node.out_connectors))
    inner_output_node = next(
        iter(dnode for dnode in nstate.data_nodes() if dnode.data == inner_output_name)
    )
    inner_output_edge = next(iter(nstate.in_edges(inner_output_node)))

    assert nstate.out_degree(inner_output_node) == 0
    assert nstate.in_degree(inner_output_node) == 1
    assert not inner_output_node.desc(nsdfg).transient
    assert inner_output_node.desc(nsdfg).shape == (1, 3)
    assert inner_output_edge.data.dst_subset == dace_sbs.Range.from_string("0, 0:3")

    outer_output_edge = next(iter(state.in_edges(output_node)))
    outer_read_edge = next(iter(state.out_edges(output_node)))

    assert state.in_degree(output_node) == 1
    assert state.in_degree(output_node) == 1
    assert outer_output_edge.data.dst_subset == dace_sbs.Range.from_string("0, 0:3")
    assert outer_read_edge.data.src_subset == dace_sbs.Range.from_string("0, 1")

    util.compile_and_run_sdfg(sdfg, **res)
    assert util.compare_sdfg_res(ref=ref, res=res)


def test_multiple_value_exchange_partial():
    """
    This test is similar to `test_multiple_value_exchange()` with the difference that
    only `z[1:3]` contains meaningful value. The transformation can not apply because
    of that. Note that it could apply, but we do not handle that case in the transformation.
    """
    sdfg, state, z, b, second_map_entry, edge_to_replace = _make_multiple_value_read_sdfg(
        transmit_all=False
    )

    case_not_supported_and_thus_none = gtx_transformations.inline_dataflow_into_map(
        sdfg=sdfg,
        state=state,
        edge=edge_to_replace,
    )
    assert case_not_supported_and_thus_none is None


def _make_sdfg_with_dref_tasklet():
    sdfg = dace.SDFG(util.unique_name(f"sdfg_with_dref_target"))
    state = sdfg.add_state(is_start_block=True)

    for name in "abc":
        sdfg.add_array(
            name,
            shape=(10,),
            dtype=dace.float64,
            transient=name == "b",
        )
    sdfg.add_array(
        "idx",
        shape=(10,),
        dtype=dace.int32,
        transient=False,
    )

    b = state.add_access("b")
    state.add_mapped_tasklet(
        "first_map",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("a[__i]")},
        outputs={"__out": dace.Memlet("b[__i]")},
        code="__out = math.sin(__in) + 1.0",
        output_nodes={b},
        external_edges=True,
    )
    _, me2, _ = state.add_mapped_tasklet(
        "dref",
        map_ranges={"__in": "0:10"},
        inputs={
            "__field": dace.Memlet("b[0:10]"),
            "__idx": dace.Memlet("idx[__i]"),
        },
        outputs={"__out": dace.Memlet("c[__i]")},
        code="__out = __field[__idx] + 1.0",
        input_nodes={b},
        external_edges=True,
    )
    sdfg.validate()

    edge_to_replace = next(
        iter(oedge for oedge in state.out_edges(me2) if oedge.dst_conn == "__field")
    )

    return sdfg, state, edge_to_replace


def test_deref_tasklet():
    sdfg, state, edge_to_replace = _make_sdfg_with_dref_tasklet()

    case_not_supported_and_thus_none = gtx_transformations.inline_fuser.find_nodes_to_inline(
        sdfg, state, edge_to_replace
    )
    assert case_not_supported_and_thus_none is None
