# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the scan-carry scalarization pass of the schedule-tree lowering."""

import dace

from gt4py.next.program_processors.runners.dace.lowering_stree import (
    scan_carry_scalarization as gtx_dace_scan_carry_scalarization,
)


def _make_scan_sdfg(carry_index: str) -> dace.SDFG:
    """Build the SDFG shape produced by ``translate_scan`` for an ND scan.

    A horizontal map contains a nested SDFG that holds the vertical scan
    ``LoopRegion``; the scan carry is a transient array internal to the nested
    SDFG, indexed by ``carry_index``.

    Args:
        carry_index: The subset expression used for every carry access.

    Returns:
        The top-level SDFG, with the horizontal map parameter named ``__i``,
        the field origin symbol named ``__origin`` and the scan column index
        named ``__k``.
    """
    nsdfg = dace.SDFG("scan_body")
    nsdfg.add_symbol("__i", dace.int32)
    nsdfg.add_symbol("__origin", dace.int32)
    nsdfg.add_array("__out", [10], dace.float64)
    nsdfg.add_transient("carry", [10], dace.float64)

    loop = dace.sdfg.state.LoopRegion(
        label="scan",
        loop_var="__k",
        condition_expr="__k < 10",
        initialize_expr="__k = 0",
        update_expr="__k = __k + 1",
    )
    nsdfg.add_node(loop, ensure_unique_name=True)
    body = loop.add_state("body", is_start_block=True)
    body.add_mapped_tasklet(
        "scan",
        map_ranges={"__unused": "0:1"},
        code="__carry_w = __carry + 1.0\n__o = __carry + 1.0",
        inputs={"__carry": dace.Memlet(data="carry", subset=carry_index)},
        outputs={
            "__carry_w": dace.Memlet(data="carry", subset=carry_index),
            "__o": dace.Memlet(data="__out", subset="__k"),
        },
        external_edges=True,
    )

    sdfg = dace.SDFG("scan_program")
    sdfg.add_symbol("__origin", dace.int32)
    sdfg.add_array("out", [10, 10], dace.float64)
    state = sdfg.add_state(is_start_block=True)
    map_entry, map_exit = state.add_map("fieldop", {"__i": "0:10"})
    nsdfg_node = state.add_nested_sdfg(
        nsdfg,
        inputs=set(),
        outputs={"__out"},
        symbol_mapping={"__i": "__i", "__origin": "__origin"},
    )
    state.add_edge(map_entry, None, nsdfg_node, None, dace.Memlet())
    state.add_edge(
        nsdfg_node, "__out", map_exit, "IN_out", dace.Memlet(data="out", subset="__i, 0:10")
    )
    map_exit.add_in_connector("IN_out")
    map_exit.add_out_connector("OUT_out")
    state.add_edge(
        map_exit,
        "OUT_out",
        state.add_access("out"),
        None,
        dace.Memlet(data="out", subset="__i, 0:10"),
    )
    return sdfg


def test_scalarize_carry_indexed_by_map_param_and_origin():
    """The carry index mixes the map parameter with the (symbolic) field origin.

    This is what the lowering actually emits: requiring the subset symbols to
    be a subset of the map parameters would reject it and leave a per-thread
    heap-allocated array inside the GPU kernel.
    """
    sdfg = _make_scan_sdfg("__i - __origin")

    scalarized = gtx_dace_scan_carry_scalarization.scalarize_scan_carries(sdfg)

    assert scalarized == {"carry"}
    nsdfg = next(
        node.sdfg
        for state in sdfg.all_states()
        for node in state.nodes()
        if isinstance(node, dace.sdfg.nodes.NestedSDFG)
    )
    assert isinstance(nsdfg.arrays["carry"], dace.data.Scalar)
    sdfg.validate()


def test_no_scalarization_when_carry_index_varies_inside_the_scan():
    """A carry indexed by the scan column index addresses a different element
    on every iteration and must be left alone."""
    sdfg = _make_scan_sdfg("__k")

    assert gtx_dace_scan_carry_scalarization.scalarize_scan_carries(sdfg) == set()


def test_no_scalarization_when_carry_index_is_map_independent():
    """A container that is not a per-map-point slot is not a scan carry."""
    sdfg = _make_scan_sdfg("__origin")

    assert gtx_dace_scan_carry_scalarization.scalarize_scan_carries(sdfg) == set()


def _make_2d_scan_sdfg() -> dace.SDFG:
    """Same as ``_make_scan_sdfg`` but with two horizontal map dimensions.

    A scan over an ``IJK`` field carries a 2D container, one slot per
    horizontal map point.
    """
    nsdfg = dace.SDFG("scan_body_2d")
    nsdfg.add_symbol("__i", dace.int32)
    nsdfg.add_symbol("__j", dace.int32)
    nsdfg.add_symbol("__origin", dace.int32)
    nsdfg.add_array("__out", [10], dace.float64)
    nsdfg.add_transient("carry", [10, 10], dace.float64)

    loop = dace.sdfg.state.LoopRegion(
        label="scan",
        loop_var="__k",
        condition_expr="__k < 10",
        initialize_expr="__k = 0",
        update_expr="__k = __k + 1",
    )
    nsdfg.add_node(loop, ensure_unique_name=True)
    body = loop.add_state("body", is_start_block=True)
    body.add_mapped_tasklet(
        "scan",
        map_ranges={"__unused": "0:1"},
        code="__carry_w = __carry + 1.0\n__o = __carry + 1.0",
        inputs={"__carry": dace.Memlet(data="carry", subset="__i - __origin, __j - __origin")},
        outputs={
            "__carry_w": dace.Memlet(data="carry", subset="__i - __origin, __j - __origin"),
            "__o": dace.Memlet(data="__out", subset="__k"),
        },
        external_edges=True,
    )

    sdfg = dace.SDFG("scan_program_2d")
    sdfg.add_symbol("__origin", dace.int32)
    sdfg.add_array("out", [10, 10, 10], dace.float64)
    state = sdfg.add_state(is_start_block=True)
    map_entry, map_exit = state.add_map("fieldop", {"__i": "0:10", "__j": "0:10"})
    nsdfg_node = state.add_nested_sdfg(
        nsdfg,
        inputs=set(),
        outputs={"__out"},
        symbol_mapping={"__i": "__i", "__j": "__j", "__origin": "__origin"},
    )
    state.add_edge(map_entry, None, nsdfg_node, None, dace.Memlet())
    state.add_edge(
        nsdfg_node, "__out", map_exit, "IN_out", dace.Memlet(data="out", subset="__i, __j, 0:10")
    )
    map_exit.add_in_connector("IN_out")
    map_exit.add_out_connector("OUT_out")
    state.add_edge(
        map_exit,
        "OUT_out",
        state.add_access("out"),
        None,
        dace.Memlet(data="out", subset="__i, __j, 0:10"),
    )
    return sdfg


def test_scalarize_carry_of_scan_over_two_horizontal_dimensions():
    """A 2D carry (``IJK`` scan) must be scalarized too.

    Skipping it leaves an ``n_i * n_j`` array heap-allocated inside every GPU
    thread, which does not link on ROCm (no device-side aligned ``new[]``).
    """
    sdfg = _make_2d_scan_sdfg()

    assert gtx_dace_scan_carry_scalarization.scalarize_scan_carries(sdfg) == {"carry"}

    nsdfg = next(
        node.sdfg
        for state in sdfg.all_states()
        for node in state.nodes()
        if isinstance(node, dace.sdfg.nodes.NestedSDFG)
    )
    assert isinstance(nsdfg.arrays["carry"], dace.data.Scalar)
    sdfg.validate()
