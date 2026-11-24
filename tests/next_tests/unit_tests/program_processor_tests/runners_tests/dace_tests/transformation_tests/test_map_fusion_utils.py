# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dace
import pytest

from dace import subsets as dace_subsets
from dace.sdfg import nodes as dace_nodes

from gt4py.next.program_processors.runners.dace.transformations import (
    map_fusion_utils as gtx_map_fusion_utils,
)

import numpy as np

from . import util


def test_copy_map_graph():
    N = dace.symbol("N", dace.int32)
    sdfg = dace.SDFG(util.unique_name("copy_map_graph"))
    A, A_desc = sdfg.add_array("A", [N], dtype=dace.float64)
    B = sdfg.add_datadesc("B", A_desc.clone())
    st = sdfg.add_state()
    A_node = st.add_access(A)
    B_node = st.add_access(B)
    _, first_map_entry, first_map_exit = st.add_mapped_tasklet(
        "plus1",
        {"i": "0:N"},
        code="_out = _inp + 1.0",
        inputs={"_inp": dace.Memlet(data=A, subset="i")},
        outputs={"_out": dace.Memlet(data=B, subset="i")},
        input_nodes={A_node},
        output_nodes={B_node},
        external_edges=True,
    )
    sdfg.validate()
    assert len(st.nodes()) == 5

    # We have created an SDFG with a mapped tasklet, that adds 1 to input array 'A'
    # and writes the result to output array 'B'. We store the SDFG hash.
    sdfg_hash = sdfg.hash_sdfg()

    # We verify that it computes the right thing.
    A_data = np.random.rand(10)
    B1_data = np.zeros_like(A_data)
    B2_data = np.zeros_like(A_data)
    sdfg(A=A_data, B=B1_data, N=10)
    assert np.allclose(B1_data, A_data + 1.0)

    # Now we copy the mapped tasklöet to a new map scope.
    #  Note that we use an empty suffix to have an execat replica, including names.
    second_map_extry, second_map_exit = gtx_map_fusion_utils.copy_map_graph(
        sdfg, st, first_map_entry, first_map_exit, suffix=None
    )
    assert second_map_extry.map is second_map_exit.map
    assert second_map_extry.map is not first_map_entry.map
    assert len(st.nodes()) == 8

    # We delete the original map and verify that the SDFG hash is the same.
    #  Note that this is expected because we use an empty suffix to copy the nodes.
    gtx_map_fusion_utils.delete_map(st, first_map_entry, first_map_exit)
    assert sdfg.hash_sdfg() == sdfg_hash

    # We should still get the same result.
    sdfg(A=A_data, B=B2_data, N=10)
    assert np.allclose(B2_data, A_data + 1.0)


@pytest.mark.parametrize(
    "map_ranges",
    [
        ({"i": "0:N"}, {"i": "0:N"}, {}, {}),  # same ndrange, no splitting
        ({"i": "0:N"}, {"i": "N:N+N"}, {}, {}),  # non-overlapping, no splitting
        ({"i": "0:N"}, {"i": "1:N"}, {"0", "1:N"}, {"1:N"}),  # partially overlapping
        ({"i": "0:M"}, {"i": "1:N"}, {}, {}),  # symbolic ndrange, no splitting
        (
            {"i": "0:M", "j": "0:N-1"},
            {"i": "1:M", "j": "0:N"},
            {"0,0:N-1", "1:M,0:N-1"},
            {"1:M,0:N-1", "1:M,N-1"},
        ),  # overlapping, same ndrange order
        (
            {"i": "0:M", "j": "0:N-1"},
            {"j": "0:N", "i": "1:M"},
            {"0,0:N-1", "1:M,0:N-1"},
            {"0:N-1,1:M", "N-1,1:M"},
        ),  # overlapping, different ndrange order
        (
            {"i": "0:M", "j": "0:N-1", "k": "3:7"},
            {"i": "1:M", "j": "0:N"},
            {},
            {},
        ),  # overlapping but one extra ndrange parameter on first map, no splitting
    ],
)
def test_split_overlapping_map_range(map_ranges):
    first_ndrange, second_ndrange = map_ranges[0:2]

    sdfg = dace.SDFG(util.unique_name("split_overlapping_map_range"))
    st = sdfg.add_state()
    first_map_entry, _ = st.add_map("first", first_ndrange)
    second_map_entry, _ = st.add_map("second", second_ndrange)

    ret = gtx_map_fusion_utils.split_overlapping_map_range(
        first_map_entry.map, second_map_entry.map
    )
    if len(map_ranges[2]) == 0 and len(map_ranges[3]) == 0:
        assert ret is None

    else:
        splitted_range_first_map, splitted_range_second_map = ret
        assert set(splitted_range_first_map) == {
            dace_subsets.Range.from_string(expected_range) for expected_range in map_ranges[2]
        }
        assert set(splitted_range_second_map) == {
            dace_subsets.Range.from_string(expected_range) for expected_range in map_ranges[3]
        }


def _make_sdfg_with_nested_sdfg_in_map() -> tuple[
    dace.SDFG, dace.SDFGState, dace_nodes.MapEntry, dace_nodes.MapExit
]:
    """Creates an SDFG with a Map whose scope contains a `NestedSDFG`.

    The Map reads from the array `A` and writes to the array `B`, while the
    nested SDFG, which computes `b = a + 1`, is inside the Map scope.
    """
    sdfg = dace.SDFG(util.unique_name("map_with_nested_sdfg"))
    state = sdfg.add_state(is_start_block=True)
    for name in "AB":
        sdfg.add_array(name, shape=(10,), dtype=dace.float64, transient=False)

    inner_sdfg = dace.SDFG("nested_computation")
    inner_state = inner_sdfg.add_state(is_start_block=True)
    for name in "ab":
        inner_sdfg.add_array(name, shape=(10,), dtype=dace.float64, transient=False)
    tasklet = inner_state.add_tasklet("compute", {"__in"}, {"__out"}, "__out = __in + 1.0")
    inner_state.add_edge(inner_state.add_access("a"), None, tasklet, "__in", dace.Memlet("a[0]"))
    inner_state.add_edge(tasklet, "__out", inner_state.add_access("b"), None, dace.Memlet("b[0]"))

    map_entry, map_exit = state.add_map("map", {"i": "0:10"})
    A, B = (state.add_access(name) for name in "AB")
    map_entry.add_scope_connectors("A")
    map_exit.add_scope_connectors("B")
    nsdfg = state.add_nested_sdfg(sdfg=inner_sdfg, inputs={"a"}, outputs={"b"}, symbol_mapping={})
    state.add_edge(A, None, map_entry, "IN_A", dace.Memlet("A[0:10]"))
    state.add_edge(map_entry, "OUT_A", nsdfg, "a", dace.Memlet("A[i]"))
    state.add_edge(nsdfg, "b", map_exit, "IN_B", dace.Memlet("B[i]"))
    state.add_edge(map_exit, "OUT_B", B, None, dace.Memlet("B[0:10]"))
    sdfg.validate()
    return sdfg, state, map_entry, map_exit


def test_copy_map_graph_does_not_resolve_symbols_for_tasklet_only_scope(monkeypatch):
    """`copy_map_graph()` must not resolve the symbols for a tasklet-only Map scope.

    The symbol resolution is an expensive full-SDFG scan, however, its result is
    only consumed by `add_nested_sdfg()`, which is never called for a scope
    that does not contain a `NestedSDFG`.
    """
    sdfg = dace.SDFG(util.unique_name("tasklet_only_map"))
    sdfg.add_array("A", shape=(10,), dtype=dace.float64, transient=False)
    sdfg.add_array("B", shape=(10,), dtype=dace.float64, transient=False)
    state = sdfg.add_state(is_start_block=True)
    state.add_mapped_tasklet(
        "computation",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("A[__i]")},
        code="__out = __in + 1.0",
        outputs={"__out": dace.Memlet("B[__i]")},
        external_edges=True,
    )
    sdfg.validate()
    map_entry = next(n for n in state.nodes() if isinstance(n, dace_nodes.MapEntry))
    map_exit = state.exit_node(map_entry)

    counter = util.DefinedSymbolsCounter(monkeypatch)
    new_map_entry, new_map_exit = gtx_map_fusion_utils.copy_map_graph(
        sdfg, state, map_entry, map_exit, suffix="copy"
    )
    assert counter.calls == 0
    assert new_map_entry is not map_entry
    assert new_map_exit is not map_exit


@pytest.mark.parametrize(
    "pass_defined_symbols",
    [True, False],
    ids=["with-defined-symbols", "without-defined-symbols"],
)
def test_copy_map_graph_symbol_resolution(monkeypatch, pass_defined_symbols):
    """`copy_map_graph()` must only resolve the symbols when no cache is given.

    When the caller manages the symbol cache itself, for example because it copies
    several map graphs of one state, the copy must reuse it without resolving the
    symbols again. If no cache is given, the symbols must be resolved when the
    copied scope contains a `NestedSDFG`, but only once for the whole scope even
    if it contains several nested SDFGs.
    """
    sdfg, state, map_entry, map_exit = _make_sdfg_with_nested_sdfg_in_map()

    # A caller that manages the cache resolves the symbols before the copy, such
    # that the resolutions performed by `copy_map_graph()` itself are counted.
    defined_symbols = state.defined_symbols() if pass_defined_symbols else None

    counter = util.DefinedSymbolsCounter(monkeypatch)
    new_map_entry, _ = gtx_map_fusion_utils.copy_map_graph(
        sdfg, state, map_entry, map_exit, suffix="copy", defined_symbols=defined_symbols
    )
    assert counter.calls == (0 if pass_defined_symbols else 1)
    copied_scope_nodes = state.scope_subgraph(
        new_map_entry, include_entry=True, include_exit=True
    ).nodes()
    assert any(isinstance(node, dace_nodes.NestedSDFG) for node in copied_scope_nodes)
