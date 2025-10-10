# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest
import copy
import numpy as np

dace = pytest.importorskip("dace")

from dace import libraries as dace_libnode
from dace.sdfg import nodes as dace_nodes

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
)

from . import util

import dace


def _make_tasklet_in_map() -> tuple[
    dace.SDFG,
    dace.SDFGState,
    dace_nodes.AccessNode,
    dace_nodes.AccessNode,
    dace_nodes.AccessNode,
    dace_nodes.AccessNode,
    dace_nodes.MapEntry,
]:
    sdfg = dace.SDFG(util.unique_name("tasklet_in_map"))
    state = sdfg.add_state(is_start_block=True)

    sdfg.add_array(
        "a",
        shape=(20,),
        dtype=dace.float64,
        transient=False,
    )
    sdfg.add_array(
        "b",
        shape=(10,),
        dtype=dace.float64,
        transient=True,
    )
    sdfg.add_array(
        "c",
        shape=(10,),
        dtype=dace.float64,
        transient=True,
    )
    sdfg.add_array(
        "d",
        shape=(20,),
        dtype=dace.float64,
        transient=False,
    )

    a, b, c, d = (state.add_access(name) for name in "abcd")

    _, me, _ = state.add_mapped_tasklet(
        "computation",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("b[__i]")},
        code="__out = __in + 3.0",
        outputs={"__out": dace.Memlet("c[__i]")},
        input_nodes={b},
        output_nodes={c},
        external_edges=True,
    )

    state.add_nedge(a, b, dace.Memlet("a[3:13] -> [0:10]"))
    state.add_nedge(c, d, dace.Memlet("c[0:10] -> [2:12]"))
    sdfg.validate()

    return sdfg, state, a, b, c, d, me


def test_reconfigure_tasklet_in_map():
    sdfg, state, a, b, c, d, me = _make_tasklet_in_map()

    assert state.out_degree(b) == 1
    old_edge = next(iter(state.out_edges(b)))

    new_edge = gtx_transformations.utils.reroute_edge(
        is_producer_edge=False,
        current_edge=old_edge,
        ss_offset=[3],
        state=state,
        sdfg=sdfg,
        old_node=b,
        new_node=a,
    )

    # Currently the SDFG is invalid, because the old edge has not been removed.
    #  We thus check if it is invalid in the correct way.
    assert state.out_degree(a) == 2
    assert state.in_degree(me) == 2
    assert {b, me} == {e.dst for e in state.out_edges(a)}
    assert len({e.dst_conn for e in state.in_edges(me)}) == 1
    assert new_edge.data.dst_subset == dace.subsets.Range.from_string("3:13")

    # The edge on on the inside has not been updated yet.
    me_oedge = next(iter(state.out_edges(me)))
    assert me_oedge.data.src_subset == dace.subsets.Range.from_string("__i")
    assert me_oedge.data.data == "b"

    # Now let's propagate the change into the Map.
    gtx_transformations.utils.reconfigure_dataflow_after_rerouting(
        is_producer_edge=False,
        new_edge=new_edge,
        ss_offset=[3],
        state=state,
        sdfg=sdfg,
        old_node=b,
        new_node=a,
    )

    # Now the edge inside the Map has changed.
    assert me_oedge.data.src_subset == dace.subsets.Range.from_string("__i")
    assert me_oedge.data.data == "b"


def test_reconfigure_reduction_in_map():
    pass


def test_reconfigure_tasklet():
    pass


def test_reconfigure_reduction():
    pass


def test_reconfigure_access_node():
    pass


def test_reconfigure_access_node_in_map():
    pass
