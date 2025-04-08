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

import numpy as np

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
)

from . import util


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


def _make_direct_self_copy_elimination_used_sdfg() -> dace.SDFG:
    sdfg = dace.SDFG(util.unique_name("direct_self_copy_elimination_used"))
    state = sdfg.add_state(is_start_block=True)

    for name in "ABCG":
        sdfg.add_array(
            name,
            shape=(20,),
            dtype=dace.float64,
            transient=False,
        )

    g_read = state.add_access("G")
    g_write = state.add_access("G")

    state.add_nedge(state.add_access("A"), g_read, dace.Memlet("A[0:5] -> [0:5]"))
    state.add_nedge(g_read, state.add_access("B"), dace.Memlet("G[5:10] -> [5:10]"))
    state.add_nedge(g_write, state.add_access("C"), dace.Memlet("G[10:15] -> [10:15]"))

    # It might look a bit strange (and might not be correct in general), but in
    #  GT4Py there is no requirement that the whole array is self copied.
    state.add_nedge(g_read, g_write, dace.Memlet("G[1:19] -> [1:19]"))
    sdfg.validate()

    return sdfg


def _make_self_copy_sdfg_with_multiple_paths() -> (
    tuple[
        dace.SDFG,
        dace.SDFGState,
        dace_nodes.AccessNode,
        dace_nodes.AccessNode,
        dace_nodes.AccessNode,
    ]
):
    """There are multiple paths between the two global nodes.

    There are to global nodes and two paths between them. The first one is direct,
    i.e. `(G) -> (G)`. The second one involves an intermediate buffer, i.e. the
    pattern `(G) -> (T) -> (G)`.
    The merge mode, which is supposed to be the normal mode, of the
    `SingleStateGlobalDirectSelfCopyElimination` transformation can not handle
    this case, but its split node can handle it.
    """
    sdfg = dace.SDFG(util.unique_name("self_copy_sdfg_with_multiple_paths"))
    state = sdfg.add_state(is_start_block=True)

    for name in "GT":
        sdfg.add_array(
            name,
            shape=(20,),
            dtype=dace.float64,
            transient=False,
        )
    sdfg.arrays["T"].transient = True

    g_read = state.add_access("G")
    g_write = state.add_access("G")
    tmp_node = state.add_access("T")

    state.add_nedge(g_read, g_write, dace.Memlet("G[0:20] -> [0:20]"))
    state.add_nedge(g_read, tmp_node, dace.Memlet("G[5:15] -> [5:15]"))
    state.add_nedge(tmp_node, g_write, dace.Memlet("T[5:15] -> [5:15]"))
    sdfg.validate()

    return sdfg, state, g_read, tmp_node, g_write


def test_global_self_copy_elimination_only_pattern():
    """Contains only the pattern -> Total elimination."""
    sdfg, state = _make_self_copy_sdfg()
    assert sdfg.number_of_nodes() == 1
    assert state.number_of_nodes() == 3
    assert util.count_nodes(state, dace_nodes.AccessNode) == 3
    assert state.number_of_edges() == 2

    count = sdfg.apply_transformations_repeated(
        gtx_transformations.SingleStateGlobalSelfCopyElimination, validate=True, validate_all=True
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
        gtx_transformations.SingleStateGlobalSelfCopyElimination, validate=True, validate_all=True
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
        gtx_transformations.SingleStateGlobalSelfCopyElimination, validate=True, validate_all=True
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


def test_direct_global_self_copy_simple():
    sdfg = dace.SDFG(util.unique_name("simple_direct_self_copy"))
    state = sdfg.add_state(is_start_block=True)

    sdfg.add_array(
        name="G",
        shape=(20, 20),
        dtype=dace.float64,
        transient=False,
    )

    state.add_nedge(
        state.add_access("G"),
        state.add_access("G"),
        dace.Memlet("G[0:20, 0:20] -> [0:20, 0:20]"),
    )

    sdfg.validate()
    assert util.count_nodes(sdfg, dace_nodes.AccessNode) == 2

    count = sdfg.apply_transformations_repeated(
        gtx_transformations.SingleStateGlobalDirectSelfCopyElimination,
        validate=True,
        validate_all=True,
    )

    assert count == 1
    assert util.count_nodes(sdfg, dace_nodes.AccessNode) == 0


def test_direct_global_self_copy_used():
    """The SDFG has a direct self copy pattern, but there are other involved nodes."""
    sdfg = _make_direct_self_copy_elimination_used_sdfg()
    assert util.count_nodes(sdfg, dace_nodes.AccessNode) == 5

    ref = {
        "A": np.array(np.random.rand(20), dtype=np.float64, copy=True),
        "B": np.array(np.random.rand(20), dtype=np.float64, copy=True),
        "C": np.array(np.random.rand(20), dtype=np.float64, copy=True),
        "G": np.array(np.random.rand(20), dtype=np.float64, copy=True),
    }
    res = {k: np.copy(v, order="K") for k, v in ref.items()}

    csdfg_ref = sdfg.compile()
    csdfg_ref(**ref)

    count = sdfg.apply_transformations_repeated(
        gtx_transformations.SingleStateGlobalDirectSelfCopyElimination,
        validate=True,
        validate_all=True,
    )

    ac_nodes_after = util.count_nodes(sdfg, dace_nodes.AccessNode, return_nodes=True)
    assert count == 1
    assert len(ac_nodes_after) == 4
    assert len(set(ac.data for ac in ac_nodes_after)) == 4

    csdfg_res = sdfg.compile()
    csdfg_res(**res)

    assert all(np.all(ref[k] == res[k]) for k in ref.keys())


def test_direct_self_copy_elimination_split_mode():
    sdfg, state, node_read_g, node_tmp, node_write_g = _make_self_copy_sdfg_with_multiple_paths()
    assert state.number_of_nodes() == 3
    assert state.number_of_edges() == 3
    assert state.out_degree(node_read_g) == 2
    assert state.in_degree(node_write_g) == 2
    assert state.degree(node_tmp) == 2

    # The `SingleStateGlobalSelfCopyElimination` transformation will use its "split"
    #  mode, thus only removing the direct connection between the two g nodes.
    count = sdfg.apply_transformations_repeated(
        gtx_transformations.SingleStateGlobalDirectSelfCopyElimination,
        validate=True,
        validate_all=True,
    )

    assert count == 1
    assert state.number_of_nodes() == 3
    assert state.number_of_edges() == 2
    assert state.out_degree(node_read_g) == 1
    assert state.in_degree(node_write_g) == 1
    assert state.degree(node_tmp) == 2


def test_global_self_copy_elimination_multi_path():
    sdfg, _, node_read_g, node_tmp, node_write_g = _make_self_copy_sdfg_with_multiple_paths()
    assert util.count_nodes(sdfg, dace_nodes.AccessNode) == 3

    # For some reason calling `sdfg.apply_transformations_repeated()` does not work.
    #  Probably a problem in the matcher, for that reason we will call it directly.
    gtx_transformations.SingleStateGlobalSelfCopyElimination.apply_to(
        sdfg=sdfg,
        verify=True,
        node_read_g=node_read_g,
        node_tmp=node_tmp,
        node_write_g=node_write_g,
    )

    sdfg.validate()
    assert util.count_nodes(sdfg, dace_nodes.AccessNode) == 0
