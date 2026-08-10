# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

dace = pytest.importorskip("dace")
from dace.sdfg import nodes as dace_nodes

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
)

from . import util


import dace


def _make_sdfg_simple() -> tuple[dace.SDFG, dace.SDFGState]:
    sdfg = dace.SDFG(util.unique_name("simple_sdfg"))
    state = sdfg.add_state(is_start_block=True)

    for name in "abc":
        sdfg.add_array(
            name,
            shape=((10,) if name == "a" else (8,)),
            dtype=dace.float64,
        )
    sdfg.arrays["b"].transient = True

    a, b, c = (state.add_access(name) for name in "abc")

    state.add_mapped_tasklet(
        "computation",
        map_ranges={"__i": "2:6"},
        inputs={"__in": dace.Memlet("a[__i]")},
        code="__out = __in + 1.34",
        outputs={"__out": dace.Memlet("b[__i - 2]")},
        output_nodes={b},
        input_nodes={a},
        external_edges=True,
    )

    state.add_nedge(b, c, dace.Memlet("b[1:4] -> [4:7]"))

    sdfg.validate()
    return sdfg, state


@pytest.mark.parametrize("remove_dead_dataflow", [True, False])
def test_map_spliter_simple(remove_dead_dataflow: bool) -> None:
    sdfg, state = _make_sdfg_simple()

    initial_ac = util.count_nodes(sdfg, dace_nodes.AccessNode, True)
    assert len(initial_ac) == 3
    assert set("abc") == {ac.data for ac in initial_ac}
    assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 1

    res, ref = util.make_sdfg_args(sdfg)
    util.compile_and_run_sdfg(sdfg, **ref)

    ret = sdfg.apply_transformations_repeated(
        gtx_transformations.MapSplitter(remove_dead_dataflow=remove_dead_dataflow),
        validate=True,
        validate_all=True,
    )
    assert ret == 1

    if remove_dead_dataflow:
        expected_maps_after = 1
        expected_ac_after = 3
    else:
        expected_maps_after = 2
        expected_ac_after = 4

    assert util.count_nodes(sdfg, dace_nodes.MapEntry) == expected_maps_after

    access_nodes = util.count_nodes(sdfg, dace_nodes.AccessNode, True)
    data_containers = {ac.data for ac in access_nodes}

    assert len(access_nodes) == expected_ac_after
    assert len(data_containers) == expected_ac_after
    assert set("ac").issubset(data_containers)

    split_ac = {ac for ac in access_nodes if ac.data not in "abc"}
    assert split_ac.isdisjoint(initial_ac)
    dead_dataflow_map = None
    reduced_map = None

    for ac in split_ac:
        assert state.in_degree(ac) == 1
        source_edge = next(iter(state.in_edges(ac)))
        source_node = source_edge.src
        write_subset = source_edge.data.dst_subset
        assert isinstance(source_node, dace_nodes.MapExit)

        if state.out_degree(ac) == 0:
            assert dead_dataflow_map is None
            assert all(start == 2 and start == stop for start, stop, _ in source_node.map.range)
            assert all(start == 0 and start == stop for start, stop, _ in write_subset)
            dead_dataflow_map = source_node

        else:
            assert reduced_map is None
            assert state.out_degree(ac) == 1
            consumer_edge = next(iter(state.out_edges(ac)))
            consumer_node = consumer_edge.dst
            assert isinstance(consumer_node, dace_nodes.AccessNode)
            assert consumer_node.data == "c"
            assert all(start == 3 and stop == start + 2 for start, stop, _ in source_node.map.range)
            assert all(start == 0 and stop == start + 2 for start, stop, _ in write_subset)
            assert all(
                start == 0 and stop == start + 2 for start, stop, _ in consumer_edge.data.src_subset
            )
            assert all(
                start == 4 and stop == start + 2 for start, stop, _ in consumer_edge.data.dst_subset
            )
            reduced_map = source_node

    if remove_dead_dataflow:
        assert dead_dataflow_map is None
        assert isinstance(reduced_map, dace_nodes.MapExit)

    else:
        assert all(isinstance(mx, dace_nodes.MapExit) for mx in [reduced_map, dead_dataflow_map])
        assert reduced_map != dead_dataflow_map

    util.compile_and_run_sdfg(sdfg, **res)
    assert util.compare_sdfg_res(ref=ref, res=res)
