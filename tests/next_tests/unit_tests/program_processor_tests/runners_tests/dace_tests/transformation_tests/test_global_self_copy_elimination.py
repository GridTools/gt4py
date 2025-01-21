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

from gt4py.next.program_processors.runners.dace_fieldview import (
    transformations as gtx_transformations,
)

from . import util

import dace


def _make_self_copy_sdfg() -> tuple[dace.SDFG, dace.SDFGState]:
    """Generates an SDFG that contains the self copying pattern."""
    sdfg = dace.SDFG(util.unique_name("self_copy_sdfg"))
    state = sdfg.add_state(is_start_block=True)

    for name in "GT":
        sdfg.add_array(
            name,
            shape=(10, 10),
            dtype=dace.float64,
            transient=True,
        )
    sdfg.arrays["G"].transient = False
    g_read, tmp_node, g_write = (state.add_access(name) for name in "GTG")

    state.add_nedge(g_read, tmp_node, dace.Memlet("G[0:10, 0:10]"))
    state.add_nedge(tmp_node, g_write, dace.Memlet("G[0:10, 0:10]"))
    sdfg.validate()

    return sdfg, state


def test_global_self_copy_elimination_only_pattern():
    """Contains only the pattern -> Total elimination."""
    sdfg, state = _make_self_copy_sdfg()
    assert sdfg.number_of_nodes() == 1
    assert state.number_of_nodes() == 3
    assert util.count_nodes(state, dace_nodes.AccessNode) == 3
    assert state.number_of_edges() == 2

    count = sdfg.apply_transformations_repeated(
        gtx_transformations.GT4PyGlobalSelfCopyElimination, validate=True, validate_all=True
    )
    assert count != 0

    assert sdfg.number_of_nodes() == 1
    assert (
        state.number_of_nodes() == 0
    ), f"Expected that 0 access nodes remained, but {state.number_of_nodes()} were there."


def test_global_self_copy_elimination_g_downstream():
    """`G` is read downstream.

    Since we ignore reads to `G` downstream, this will not influence the
    transformation.
    """
    sdfg, state1 = _make_self_copy_sdfg()

    # Add a read to `G` downstream.
    state2 = sdfg.add_state_after(state1)
    sdfg.add_array(
        "output",
        shape=(10, 10),
        dtype=dace.float64,
        transient=False,
    )

    state2.add_mapped_tasklet(
        "downstream_computation",
        map_ranges={"__i0": "0:10", "__i1": "0:10"},
        inputs={"__in": dace.Memlet("G[__i0, __i1]")},
        code="__out = __in + 10.0",
        outputs={"__out": dace.Memlet("output[__i0, __i1]")},
        external_edges=True,
    )
    sdfg.validate()
    assert state2.number_of_nodes() == 5

    count = sdfg.apply_transformations_repeated(
        gtx_transformations.GT4PyGlobalSelfCopyElimination, validate=True, validate_all=True
    )
    assert count != 0

    assert sdfg.number_of_nodes() == 2
    assert (
        state1.number_of_nodes() == 0
    ), f"Expected that 0 access nodes remained, but {state.number_of_nodes()} were there."
    assert state2.number_of_nodes() == 5
    assert util.count_nodes(state2, dace_nodes.AccessNode) == 2
    assert util.count_nodes(state2, dace_nodes.MapEntry) == 1


def test_global_self_copy_elimination_tmp_downstream():
    """`T` is read downstream.

    Because `T` is read downstream, the read to `G` will be retained, but the write
    will be removed.
    """
    sdfg, state1 = _make_self_copy_sdfg()

    # Add a read to `G` downstream.
    state2 = sdfg.add_state_after(state1)
    sdfg.add_array(
        "output",
        shape=(10, 10),
        dtype=dace.float64,
        transient=False,
    )

    state2.add_mapped_tasklet(
        "downstream_computation",
        map_ranges={"__i0": "0:10", "__i1": "0:10"},
        inputs={"__in": dace.Memlet("T[__i0, __i1]")},
        code="__out = __in + 10.0",
        outputs={"__out": dace.Memlet("output[__i0, __i1]")},
        external_edges=True,
    )
    sdfg.validate()
    assert state2.number_of_nodes() == 5

    count = sdfg.apply_transformations_repeated(
        gtx_transformations.GT4PyGlobalSelfCopyElimination, validate=True, validate_all=True
    )
    assert count != 0

    assert sdfg.number_of_nodes() == 2
    assert state1.number_of_nodes() == 2
    assert util.count_nodes(state1, dace_nodes.AccessNode) == 2
    assert all(state1.degree(node) == 1 for node in state1.nodes())
    assert next(iter(state1.source_nodes())).data == "G"
    assert next(iter(state1.sink_nodes())).data == "T"

    assert state2.number_of_nodes() == 5
    assert util.count_nodes(state2, dace_nodes.AccessNode) == 2
    assert util.count_nodes(state2, dace_nodes.MapEntry) == 1
