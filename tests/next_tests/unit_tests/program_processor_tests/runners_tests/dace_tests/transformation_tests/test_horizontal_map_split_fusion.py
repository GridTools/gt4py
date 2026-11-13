# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dace
import pytest
import copy

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
)
from gt4py.next.program_processors.runners.dace.transformations import (
    map_fusion_extended as gtx_map_fusion_extended,
)
from dace.sdfg import nodes as dace_nodes, propagation as dace_propagation
from sympy.core.numbers import Number

from . import util


def _make_sdfg_with_multiple_maps_that_share_inputs(
    N: str | int,
) -> tuple[dace.SDFG, dace.SDFGState]:
    """
    Create a SDFG with multiple maps that share inputs and outputs.
    The SDFG has originally 4 maps:
    - The first map has multiple tasklets and a nested map.
    - The second, third and fourth map have a single tasklet.
    The some of the maps share inputs with each other.
    The SDFG has the following inputs and outputs:
    - Inputs: a[i, j], b[i, j], c[i, j], d[i, j]
    - Outputs: out1[i, j], out2[i, j], out3[i, j], out4[i, j]
    """
    shape = (N, N)
    sdfg = dace.SDFG(util.unique_name("multiple_maps"))
    state = sdfg.add_state(is_start_block=True)

    for name in ["a", "b", "c", "d", "out1", "out2", "out3", "out4"]:
        sdfg.add_array(
            name=name,
            shape=shape,
            dtype=dace.float64,
            transient=False,
        )

    sdfg.add_scalar("tmp1", dtype=dace.float64, transient=True)
    sdfg.add_scalar("tmp2", dtype=dace.float64, transient=True)
    sdfg.add_scalar("tmp3", dtype=dace.float64, transient=True)
    sdfg.add_scalar("tmp4", dtype=dace.float64, transient=True)
    a, b, out1, tmp1, tmp2, tmp3, tmp4 = (
        state.add_access(name) for name in ["a", "b", "out1", "tmp1", "tmp2", "tmp3", "tmp4"]
    )
    sdfg.add_symbol("horizontal_start", dace.int32)
    sdfg.add_symbol("horizontal_end", dace.int32)

    # First independent Tasklet
    task1 = state.add_tasklet(
        "task1_indepenent",
        inputs={
            "__in0",  # <- `b[i, j]`
        },
        outputs={
            "__out0",  # <- `tmp1`
        },
        code="__out0 = __in0 + 3.0",
    )

    # This is the second independent Tasklet
    task2 = state.add_tasklet(
        "task2_indepenent",
        inputs={
            "__in0",  # <- `tmp1`
            "__in1",  # <- `b[i, j]`
        },
        outputs={
            "__out0",  # <- `tmp2`
        },
        code="__out0 = __in0 + __in1",
    )

    # This is the third Tasklet, which is dependent
    task3 = state.add_tasklet(
        "task3_dependent",
        inputs={
            "__in0",  # <- `tmp2`
            "__in1",  # <- `a[i, j]`
        },
        outputs={
            "__out0",  # <- `out1[i, j]`.
        },
        code="__out0 = __in0 + __in1",
    )

    # Now create the map using the above tasklets
    mentry, mexit = state.add_map(
        "complex_map",
        ndrange={"i": f"0:{N}", "j": "horizontal_start:horizontal_end"},
    )

    small_map_entry, small_map_exit = state.add_map(
        "small_map",
        ndrange={"i": "0:4"},
    )

    task4 = state.add_tasklet(
        "task4_indepenent",
        inputs={
            "__in0",  # <- `b[i, j]`
        },
        outputs={
            "__out0",  # <- `b[i, j]+12`
        },
        code="__out0 = __in0 + 3.0",
    )

    # Now assemble everything.
    state.add_edge(mentry, "OUT_b", task1, "__in0", dace.Memlet("b[i, j]"))
    state.add_edge(task1, "__out0", tmp1, None, dace.Memlet("tmp1[0]"))

    state.add_edge(tmp1, None, task2, "__in0", dace.Memlet("tmp1[0]"))
    state.add_edge(mentry, "OUT_b", task2, "__in1", dace.Memlet("b[i, j]"))
    state.add_edge(task2, "__out0", tmp2, None, dace.Memlet("tmp2[0]"))

    state.add_edge(tmp2, None, task3, "__in0", dace.Memlet("tmp2[0]"))
    state.add_edge(mentry, "OUT_a", task3, "__in1", dace.Memlet("a[i, j]"))
    state.add_edge(task3, "__out0", tmp3, None, dace.Memlet("tmp3[0]"))
    state.add_edge(tmp3, None, small_map_entry, "IN_tmp3", dace.Memlet("tmp3[0]"))
    state.add_edge(small_map_entry, "OUT_tmp3", task4, "__in0", dace.Memlet("tmp3[0]"))

    state.add_edge(task4, "__out0", small_map_exit, "IN_tmp4", dace.Memlet("tmp4[0]"))
    state.add_edge(small_map_exit, "OUT_tmp4", tmp4, None, dace.Memlet("tmp4[0]"))
    state.add_edge(tmp4, None, mexit, "IN_out1", dace.Memlet("tmp4[0] -> [i, j]"))

    state.add_edge(a, None, mentry, "IN_a", sdfg.make_array_memlet("a"))
    state.add_edge(b, None, mentry, "IN_b", sdfg.make_array_memlet("b"))
    state.add_edge(mexit, "OUT_out1", out1, None, sdfg.make_array_memlet("out1"))
    for name in ["a", "b"]:
        mentry.add_scope_connectors(name)
    mexit.add_scope_connectors("out1")
    small_map_entry.add_scope_connectors("tmp3")
    small_map_exit.add_scope_connectors("tmp4")

    state.add_mapped_tasklet(
        name="second_computation",
        map_ranges=[("i", f"0:{N // 2}"), ("j", "horizontal_start:horizontal_end")],
        inputs={"__in0": dace.Memlet("b[i, j]")},
        code="__out = __in0 + 1.0",
        outputs={"__out": dace.Memlet("out2[i, j]")},
        external_edges=True,
    )

    state.add_mapped_tasklet(
        name="third_computation",
        map_ranges=[("i", f"0:{N // 2}"), ("j", "horizontal_start:horizontal_end")],
        inputs={"__in0": dace.Memlet("c[i, j]"), "__in1": dace.Memlet("a[i, j]")},
        code="__out = __in0 + __in1 + 1.0",
        outputs={"__out": dace.Memlet("out3[i, j]")},
        external_edges=True,
    )

    state.add_mapped_tasklet(
        name="fourth_computation",
        map_ranges=[("i", f"{N // 4}:{N}"), ("j", "horizontal_start:horizontal_end")],
        inputs={"__in0": dace.Memlet("d[i, j]"), "__in1": dace.Memlet("a[i, j]")},
        code="__out = __in0 + __in1 + 2.0",
        outputs={"__out": dace.Memlet("out4[i, j]")},
        external_edges=True,
    )

    existing_access_nodes = {}

    # Remove duplicate access nodes which get generated by the above maps
    for access_node in state.nodes():
        if isinstance(access_node, dace_nodes.AccessNode):
            if access_node.label not in existing_access_nodes:
                existing_access_nodes[access_node.label] = access_node
            else:
                edges_for_removal = []
                for edge in state.in_edges(access_node):
                    edges_for_removal.append(edge)
                    state.add_edge(
                        edge.src,
                        edge.src_conn,
                        existing_access_nodes[access_node.label],
                        edge.dst_conn,
                        copy.deepcopy(edge.data),
                    )
                for edge in state.out_edges(access_node):
                    edges_for_removal.append(edge)
                    state.add_edge(
                        existing_access_nodes[access_node.label],
                        edge.src_conn,
                        edge.dst,
                        edge.dst_conn,
                        copy.deepcopy(edge.data),
                    )
                for edge in edges_for_removal:
                    state.remove_edge(edge)
                state.remove_node(access_node)

    dace_propagation.propagate_states(sdfg)
    sdfg.validate()

    return sdfg, state


@pytest.mark.parametrize("run_map_fusion", [True, False])
def test_horizontal_map_fusion(run_map_fusion: bool):
    N = 20
    sdfg, state = _make_sdfg_with_multiple_maps_that_share_inputs(N)

    nb_applications = gtx_transformations.gt_horizontal_map_split_fusion(
        sdfg=sdfg,
        run_simplify=False,
        fuse_map_fragments=True,
        run_map_fusion=run_map_fusion,
        consolidate_edges_only_if_not_extending=False,
        validate=True,
        validate_all=True,
    )
    scope_dict = state.scope_dict()

    expected_in_degree_of_outputs = {"out1": 3, "out2": 2, "out3": 2, "out4": 2}
    for ac in state.data_nodes():
        if ac.data in expected_in_degree_of_outputs:
            assert expected_in_degree_of_outputs[ac.data] == state.in_degree(ac)

    if run_map_fusion:
        # Automatic fusion was requested.
        assert nb_applications == 5

    else:
        # Automatic fusion was not enabled, thus only 3 applications and there are
        #  some overlapping maps. That we have to fuse later.
        assert nb_applications == 3
        assert (
            sum(
                1
                for map_entry in util.count_nodes(state, dace_nodes.MapEntry, True)
                if scope_dict[map_entry] is None
            )
            == 5
        )
        aux_fuses = sdfg.apply_transformations_repeated(
            gtx_transformations.MapFusionHorizontal(
                only_if_common_ancestor=True,
                consolidate_edges_only_if_not_extending=False,
            ),
            validate=True,
            validate_all=True,
        )
        assert aux_fuses == 2

    assert (
        sum(
            1
            for map_entry in util.count_nodes(state, dace_nodes.MapEntry, True)
            if scope_dict[map_entry] is None
        )
        == 3
    )

    # check that there is no overlap between the maps' ranges
    map_entries = util.count_nodes(sdfg, dace_nodes.MapEntry, return_nodes=True)
    # check that there are no maps with overlapping ranges if they share an input
    for i, map_entry_i in enumerate(map_entries):
        for map_entry_j in map_entries[i + 1 :]:
            # Check if the maps share an input
            shared_input = any(
                iedge.src == jedge.src and iedge.src_conn == jedge.src_conn
                for iedge in sdfg.state(0).in_edges(map_entry_i)
                for jedge in sdfg.state(0).in_edges(map_entry_j)
            )
            if not shared_input:
                continue

            # Check if the ranges overlap
            range_i = map_entry_i.map.range
            range_j = map_entry_j.map.range
            for dim in range(len(range_i)):
                if (
                    isinstance(range_i[dim][0], (Number, int))
                    and isinstance(range_i[dim][1], (Number, int))
                    and isinstance(range_j[dim][0], (Number, int))
                    and isinstance(range_j[dim][1], (Number, int))
                    and not (
                        range_i[dim][1] <= range_j[dim][0] or range_j[dim][1] <= range_i[dim][0]
                    )
                ):
                    raise AssertionError(
                        f"Found maps with overlapping ranges: {map_entry_i.label} and {map_entry_j.label} "
                        f"[{sdfg.state(0).in_edges(map_entry_i)[0].src.label}]"
                    )


def test_horizontal_map_fusion_rejects_contradictory_map_scopes():
    """`can_be_applied()` tests the scope itself, so it has to reject this like DaCe does."""
    sdfg, _ = _make_sdfg_with_multiple_maps_that_share_inputs(10)

    with pytest.raises(ValueError, match="only_inner_maps"):
        sdfg.apply_transformations_repeated(
            gtx_transformations.MapFusionHorizontal(
                only_inner_maps=True,
                only_toplevel_maps=True,
            ),
            validate=False,
        )


def _make_sdfg_with_splitable_maps_one_containing_a_nested_sdfg() -> tuple[
    dace.SDFG, dace.SDFGState, dace_nodes.MapEntry, dace_nodes.MapExit
]:
    """Creates an SDFG with two splitable sibling Maps, the first containing a `NestedSDFG`.

    The first Map (`first_map`, range `i` in `[0, 10)`) contains a nested SDFG,
    which computes `b = a + 1`, inside its scope, while the second Map
    (`second_map_map`, range `i` in `[5, 15)`) is a tasklet-only Map. The two
    ranges partially overlap, such that `SplitMapRange.split_maps()` splits
    both Maps into fragments.
    """
    sdfg = dace.SDFG(util.unique_name("split_maps_with_nested_sdfg"))
    state = sdfg.add_state(is_start_block=True)
    for name in "ABCD":
        sdfg.add_array(name, shape=(15,), dtype=dace.float64, transient=False)

    inner_sdfg = dace.SDFG("nested_computation")
    inner_state = inner_sdfg.add_state(is_start_block=True)
    for name in "ab":
        inner_sdfg.add_array(name, shape=(15,), dtype=dace.float64, transient=False)
    tasklet = inner_state.add_tasklet("compute", {"__in"}, {"__out"}, "__out = __in + 1.0")
    inner_state.add_edge(inner_state.add_access("a"), None, tasklet, "__in", dace.Memlet("a[0]"))
    inner_state.add_edge(tasklet, "__out", inner_state.add_access("b"), None, dace.Memlet("b[0]"))

    first_map_entry, first_map_exit = state.add_map("first_map", {"i": "0:10"})
    A, B = (state.add_access(name) for name in "AB")
    first_map_entry.add_scope_connectors("A")
    first_map_exit.add_scope_connectors("B")
    nsdfg = state.add_nested_sdfg(sdfg=inner_sdfg, inputs={"a"}, outputs={"b"}, symbol_mapping={})
    state.add_edge(A, None, first_map_entry, "IN_A", dace.Memlet("A[0:10]"))
    state.add_edge(first_map_entry, "OUT_A", nsdfg, "a", dace.Memlet("A[i]"))
    state.add_edge(nsdfg, "b", first_map_exit, "IN_B", dace.Memlet("B[i]"))
    state.add_edge(first_map_exit, "OUT_B", B, None, dace.Memlet("B[0:10]"))

    state.add_mapped_tasklet(
        "second_map",
        map_ranges={"i": "5:15"},
        inputs={"__in": dace.Memlet("C[i]")},
        code="__out = __in + 2.0",
        outputs={"__out": dace.Memlet("D[i]")},
        external_edges=True,
    )
    sdfg.validate()
    return sdfg, state, first_map_entry, first_map_exit


def test_split_maps_does_not_resolve_symbols_for_tasklet_only_maps(monkeypatch):
    """`split_maps()` must not resolve the symbols when both Maps are tasklet-only.

    The symbol resolution is an expensive full-SDFG scan, however, its result is
    only consumed by `add_nested_sdfg()`, which is never called when none of the
    copied scopes contains a `NestedSDFG`.
    """
    sdfg, state = _make_sdfg_with_multiple_maps_that_share_inputs(20)
    second_map_entry = next(
        n
        for n in state.nodes()
        if isinstance(n, dace_nodes.MapEntry) and n.label == "second_computation_map"
    )
    fourth_map_entry = next(
        n
        for n in state.nodes()
        if isinstance(n, dace_nodes.MapEntry) and n.label == "fourth_computation_map"
    )

    counter = util.DefinedSymbolsCounter(monkeypatch)
    gtx_map_fusion_extended.HorizontalSplitMapRange(fuse_map_fragments=False).split_maps(
        state,
        sdfg,
        second_map_entry,
        state.exit_node(second_map_entry),
        fourth_map_entry,
        state.exit_node(fourth_map_entry),
    )
    assert counter.calls == 0


def test_split_maps_resolves_symbols_once_when_a_scope_contains_a_nested_sdfg(monkeypatch):
    """`split_maps()` must resolve the symbols once when a copied scope has a `NestedSDFG`.

    The symbols must be resolved when one of the copied scopes contains a
    `NestedSDFG`, but only once for all Maps split by the transformation, i.e.
    the result must be shared by all fragments.
    """
    sdfg, state, first_map_entry, first_map_exit = (
        _make_sdfg_with_splitable_maps_one_containing_a_nested_sdfg()
    )
    second_map_entry = next(
        n
        for n in state.nodes()
        if isinstance(n, dace_nodes.MapEntry) and n.label.startswith("second_map")
    )

    counter = util.DefinedSymbolsCounter(monkeypatch)
    gtx_map_fusion_extended.HorizontalSplitMapRange(fuse_map_fragments=False).split_maps(
        state,
        sdfg,
        first_map_entry,
        first_map_exit,
        second_map_entry,
        state.exit_node(second_map_entry),
    )
    assert counter.calls == 1
    # The fragments of the first Map must still contain the nested SDFG.
    assert sum(1 for node in state.nodes() if isinstance(node, dace_nodes.NestedSDFG)) == 2
