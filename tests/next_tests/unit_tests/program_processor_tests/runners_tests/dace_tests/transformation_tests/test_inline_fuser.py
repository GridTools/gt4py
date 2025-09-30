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
from dace import data as dace_data
from dace.transformation import dataflow as dace_dftrafo

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
)

from . import util

import dace


def _create_simple_fusable_sdfg() -> tuple[
    dace.SDFG, dace.SDFGState, dace_graph.MultiConnectorEdge[dace.Memlet]
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
    _, me, _ = state.add_mapped_tasklet(
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

    return sdfg, state, next(iter(state.out_edges(me)))


def test_inline_fuse_simple_case():
    sdfg, state, edge_to_replace = _create_simple_fusable_sdfg()

    nsdfg, intermediate_node = gtx_transformations.inline_dataflow_into_map(
        sdfg=sdfg,
        state=state,
        edge=edge_to_replace,
    )
