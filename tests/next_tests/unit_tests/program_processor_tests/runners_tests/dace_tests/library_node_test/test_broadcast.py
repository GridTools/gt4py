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

from dace import data as dace_data, subsets as dace_sbs
from dace.sdfg import nodes as dace_nodes

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
    library_nodes as gtx_lib_nodes,
)

from ..transformation_tests import util


import dace


def _make_2d_broadcast() -> tuple[dace.SDFG, dace.SDFGState]:
    sdfg = dace.SDFG(gtx_transformations.utils.unique_name("broadcast_2d"))
    state = sdfg.add_state(is_start_block=True)

    sdfg.add_scalar(
        "bcast_value",
        dtype=dace.float64,
        transient=False,
    )
    sdfg.add_array(
        "bcast_result",
        shape=(10, 10),
        dtype=dace.float64,
        transient=False,
    )

    bcast_value = state.add_access("bcast_value")
    bcast_result = state.add_access("bcast_result")

    bcast_node1 = gtx_lib_nodes.Broadcast(name="bcast_node1", broadcast_in_dims=[])
    bcast_node2 = gtx_lib_nodes.Broadcast(name="bcast_node2", broadcast_in_dims=[])

    state.add_edge(
        bcast_value, None, bcast_node1, "_inp", dace.Memlet(data=bcast_value.data, subset="0")
    )
    state.add_edge(
        bcast_node1,
        "_outp",
        bcast_result,
        None,
        dace.Memlet(data=bcast_result.data, subset="1, 3:7"),
    )

    state.add_edge(
        bcast_value, None, bcast_node2, "_inp", dace.Memlet(data=bcast_value.data, subset="0")
    )
    state.add_edge(
        bcast_node2,
        "_outp",
        bcast_result,
        None,
        dace.Memlet(data=bcast_result.data, subset="5:8, 4:6"),
    )

    sdfg.validate()

    return sdfg, state


@pytest.mark.parametrize("use_inplace_expansion", [True, False])
def test_broadcast_expansion_inplace(use_inplace_expansion: bool):
    sdfg, state = _make_2d_broadcast()

    assert state.number_of_nodes() == 4
    assert util.count_nodes(sdfg, gtx_lib_nodes.Broadcast) == 2
    assert util.count_nodes(sdfg, dace_nodes.AccessNode) == 2

    ref, res = util.make_sdfg_args(sdfg)

    ref["bcast_result"][1, 3:7] = ref["bcast_value"]
    ref["bcast_result"][5:8, 4:6] = ref["bcast_value"]

    for node in list(state.nodes()):
        if isinstance(node, gtx_lib_nodes.Broadcast):
            if use_inplace_expansion:
                gtx_lib_nodes.inplace_broadcast_expander(node, state, sdfg)
            else:
                node.expand(state)

    assert util.count_nodes(sdfg, gtx_lib_nodes.Broadcast) == 0
    assert util.count_nodes(sdfg, dace_nodes.AccessNode) == 2

    if use_inplace_expansion:
        assert state.number_of_nodes() == 8
        assert util.count_nodes(sdfg, dace_nodes.Tasklet) == 2
        assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 2
    else:
        assert state.number_of_nodes() == 4
        assert util.count_nodes(sdfg, dace_nodes.NestedSDFG) == 2

    util.compile_and_run_sdfg(sdfg, **res)
    assert util.compare_sdfg_res(ref=ref, res=res)
