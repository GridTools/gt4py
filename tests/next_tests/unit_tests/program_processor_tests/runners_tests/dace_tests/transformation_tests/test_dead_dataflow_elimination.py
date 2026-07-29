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


def _make_empty_memlets_sdfg() -> tuple[dace.SDFG, dace.SDFGState]:
    sdfg = dace.SDFG(util.unique_name("empty_memlets"))
    state = sdfg.add_state(is_start_block=True)

    anames = ["a", "b", "c"]
    for name in anames:
        sdfg.add_array(
            name,
            shape=(10,),
            dtype=dace.float64,
            transient=False,
        )
    sdfg.arrays["b"].transient = True

    a, b, c = (state.add_access(name) for name in anames)

    state.add_nedge(a, b, dace.Memlet("a[1:1] -> [1:1]"))
    state.add_nedge(a, b, dace.Memlet("a[2:2] -> [2:2]"))
    state.add_nedge(b, c, dace.Memlet("b[4:4] -> [4:4]"))
    state.add_nedge(b, c, dace.Memlet("b[5:5] -> [5:5]"))
    state.add_nedge(a, c, dace.Memlet("a[8:8] -> [8:8]"))
    state.add_nedge(a, c, dace.Memlet("a[9] -> [9]"))

    sdfg.validate()
    return sdfg, state


def _make_zero_iter_step_map() -> tuple[dace.SDFG, dace.SDFGState]:
    sdfg = dace.SDFG(util.unique_name("empty_map"))
    state = sdfg.add_state(is_start_block=True)

    anames = ["a", "b", "c"]
    for name in anames:
        sdfg.add_array(
            name,
            shape=(10,),
            dtype=dace.float64,
            transient=False,
        )

    a, b, c = (state.add_access(name) for name in anames)

    state.add_mapped_tasklet(
        "no_ops",
        map_ranges={"__i": "3:3"},
        inputs={
            "__in1": dace.Memlet("a[__i]"),
            "__in2": dace.Memlet("b[__i]"),
        },
        code="__out = __in1 + __in2",
        outputs={"__out": dace.Memlet("c[__i]")},
        input_nodes={a, b},
        output_nodes={c},
        external_edges=True,
    )
    state.add_mapped_tasklet(
        "ops",
        map_ranges={"__i": "0:6"},
        inputs={"__in": dace.Memlet("a[__i]")},
        code="__out = __in + 2.0",
        outputs={"__out": dace.Memlet("c[__i]")},
        input_nodes={a},
        output_nodes={c},
        external_edges=True,
    )
    sdfg.validate()

    return sdfg, state


def test_remove_empty_dataflow():
    sdfg, state = _make_empty_memlets_sdfg()
    assert state.number_of_edges() == 6
    assert util.count_nodes(sdfg, dace_nodes.AccessNode) == 3

    # Out of the 6 edges only the one between `a` and `c` is valid.
    nb_removed = gtx_transformations.gt_eliminate_dead_dataflow(
        sdfg=sdfg,
        run_simplify=False,
        validate=True,
        validate_all=True,
    )

    ac_nodes = util.count_nodes(sdfg, dace_nodes.AccessNode, return_nodes=True)
    assert nb_removed == 5
    assert len(ac_nodes) == 2
    assert {"a", "c"} == {ac.data for ac in ac_nodes}
    assert {"a", "c"} == sdfg.arrays.keys()
    assert state.number_of_edges() == 1

    a_c_edge = list(state.edges())[0]
    assert a_c_edge.data.subset.min_element()[0] == 9
    assert a_c_edge.data.other_subset.min_element()[0] == 9


def test_remove_empty_map():
    sdfg, state = _make_zero_iter_step_map()
    assert sdfg.number_of_nodes() == 1
    assert util.count_nodes(sdfg, dace_nodes.AccessNode) == 3
    assert util.count_nodes(sdfg, dace_nodes.Tasklet) == 2
    assert (
        sum(
            me.map.label.startswith("no_ops_")
            for me in util.count_nodes(sdfg, dace_nodes.MapEntry, return_nodes=True)
        )
        == 1
    )

    # The function will remove the "no_ops" maps and it will apply at one Map.
    nb_removed = gtx_transformations.gt_eliminate_dead_dataflow(
        sdfg=sdfg,
        run_simplify=False,
        validate_all=True,
    )

    ac_nodes = util.count_nodes(sdfg, dace_nodes.AccessNode, return_nodes=True)
    me_nodes = util.count_nodes(sdfg, dace_nodes.MapEntry, return_nodes=True)
    assert nb_removed == 1
    assert len(ac_nodes) == 2
    assert {"a", "c"} == {ac.data for ac in ac_nodes}
    assert len(me_nodes) == 1
    assert me_nodes[0].map.label.startswith("ops_")
