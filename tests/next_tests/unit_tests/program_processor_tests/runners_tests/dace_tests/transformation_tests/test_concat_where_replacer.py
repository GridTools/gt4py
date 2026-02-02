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
import dace
from dace.sdfg import nodes as dace_nodes, graph as dace_graph
from dace import data as dace_data, subsets as dace_sbs
from dace.transformation import dataflow as dace_dftrafo

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
)

from . import util


def _make_concat_where_different_inputs() -> tuple[
    dace.SDFG, dace.SDFGState, dace_nodes.AccessNode, dace_nodes.MapEntry
]:
    """ """
    sdfg = dace.SDFG(util.unique_name("concat_where_replacer_different_inputs"))
    state = sdfg.add_state()
    for name in "abcde":
        sdfg.add_array(
            name=name,
            shape=(10,),
            dtype=dace.float64,
            transient=(name == "c"),
        )
    a, b, c, d, e = (state.add_access(name) for name in "abcde")

    state.add_nedge(a, c, dace.Memlet("a[1:6] -> [0:5]"))
    state.add_nedge(b, c, dace.Memlet("b[3:8] -> [5:10]"))

    _, me, _ = state.add_mapped_tasklet(
        "computation",
        map_ranges={"__i": "0:10"},
        inputs={
            "__in0": dace.Memlet("c[__i]"),
            "__in1": dace.Memlet("e[__i]"),
        },
        code="__out2 = __in1 + __in0",
        outputs={"__out2": dace.Memlet("d[__i]")},
        input_nodes={c, e},
        output_nodes={d},
        external_edges=True,
        schedule=dace.dtypes.ScheduleType.Sequential,
    )
    sdfg.validate()

    return sdfg, state, c, me


def test_concat_where_different_inputs():
    sdfg, state, concat_node, me = _make_concat_where_different_inputs()

    access_nodes_before = util.count_nodes(state, dace_nodes.AccessNode, return_nodes=True)
    assert len(access_nodes_before) == 5
    assert concat_node.data == "c"
    assert [ac for ac in access_nodes_before if ac.desc(sdfg).transient] == [concat_node]
    assert util.count_nodes(state, dace_nodes.Tasklet) == 1
    assert state.in_degree(me) == 2

    ref, res = util.make_sdfg_args(sdfg)
    util.compile_and_run_sdfg(sdfg, **ref)

    gtx_transformations.gt_replace_concat_where_node(
        state=state,
        sdfg=sdfg,
        concat_node=concat_node,
        map_entry=me,
    )
    sdfg.validate()

    access_nodes_after = util.count_nodes(state, dace_nodes.AccessNode, return_nodes=True)
    assert len(access_nodes_after) == 5
    assert "c" not in access_nodes_after
    assert all(
        ac.data.startswith("__gt4py_concat_where_mapper_temp_for_c")
        for ac in access_nodes_after
        if ac.desc(sdfg).transient
    )
    assert util.count_nodes(state, dace_nodes.Tasklet) == 2
    assert state.in_degree(me) == 3

    csdfg = util.compile_and_run_sdfg(sdfg, **res)
    assert util.compare_sdfg_res(ref=ref, res=res)
