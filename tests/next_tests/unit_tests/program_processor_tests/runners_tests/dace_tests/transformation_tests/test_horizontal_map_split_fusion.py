# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import copy

dace = pytest.importorskip("dace")

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
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
    state.add_edge(tmp4, None, mexit, "IN_out1", dace.Memlet("tmp4[0]"))

    state.add_edge(a, None, mentry, "IN_a", sdfg.make_array_memlet("a"))
    state.add_edge(b, None, mentry, "IN_b", sdfg.make_array_memlet("b"))
    state.add_edge(mexit, "OUT_out1", out1, None, sdfg.make_array_memlet("out1"))
    for name in ["a", "b"]:
        mentry.add_scope_connectors(name)
    mexit.add_in_connector("IN_out1")
    mexit.add_out_connector("OUT_out1")
    small_map_entry.add_in_connector("IN_tmp3")
    small_map_entry.add_out_connector("OUT_tmp3")
    small_map_exit.add_in_connector("IN_tmp4")
    small_map_exit.add_out_connector("OUT_tmp4")

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
