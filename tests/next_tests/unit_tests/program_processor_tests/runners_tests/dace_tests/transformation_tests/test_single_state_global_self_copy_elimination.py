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

    state.add_nedge(g_read, tmp_node, dace.Memlet("G[0:10, 0:10] -> [0:10, 0:10]"))
    state.add_nedge(tmp_node, g_write, dace.Memlet("G[0:10, 0:10] -> [0:10, 0:10]"))
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


def _make_concat_where_like(
    handle_last_level: bool,
    whole_write_back: bool,
) -> tuple[
    dace.SDFG,
    dace.SDFGState,
    dace_nodes.AccessNode,
    dace_nodes.AccessNode,
    dace_nodes.AccessNode,
    dace_nodes.AccessNode,
]:
    if handle_last_level:
        j_desc = "last"
        j_idx = 9
        j_range = "0:9"
    else:
        j_desc = "first"
        j_idx = 0
        j_range = "1:10"

    sdfg = dace.SDFG(util.unique_name(f"self_copy_sdfg_concat_where_like_{j_desc}_level"))
    state = sdfg.add_state(is_start_block=True)

    sdfg.add_array(
        "g",
        shape=(10, 10),
        dtype=dace.float64,
        transient=False,
    )
    sdfg.add_array(
        "o",
        shape=(10, 12),
        dtype=dace.float64,
        transient=False,
    )
    sdfg.add_array(
        "t",
        shape=(8, 10),
        dtype=dace.float64,
        transient=True,
    )

    g1, t, g2, o = (state.add_access(name) for name in "gtgo")

    state.add_nedge(g1, t, dace.Memlet(f"g[2:10, {j_idx}] -> [0:8, {j_idx}]"))

    if whole_write_back:
        state.add_nedge(t, g2, dace.Memlet(f"t[0:4, 0:10] -> [2:6, 0:10]"))
        state.add_nedge(t, g2, dace.Memlet(f"t[4:8, 0:10] -> [6:10, 0:10]"))
    else:
        state.add_nedge(t, g2, dace.Memlet(f"t[0:4, {j_range}] -> [2:6, {j_range}]"))
        state.add_nedge(t, g2, dace.Memlet(f"t[4:8, {j_range}] -> [6:10, {j_range}]"))

    state.add_mapped_tasklet(
        f"{j_desc}_level",
        map_ranges={"__i": "2:10"},
        inputs={},
        code="__out = __i / 3.0",
        outputs={"__out": dace.Memlet(f"g[__i, {j_idx}]")},
        output_nodes={g1},
        external_edges=True,
    )
    state.add_mapped_tasklet(
        "bulk_value",
        map_ranges={
            "__i": "0:8",
            "__j": j_range,
        },
        inputs={},
        code="__out = sin(__i + 0.5)**2 + cos(__j + 0.1) ** 2",
        outputs={"__out": dace.Memlet("t[__i, __j]")},
        output_nodes={t},
        external_edges=True,
    )
    state.add_mapped_tasklet(
        "consumer",
        map_ranges={
            "__i": "0:8",
            "__j": "0:10",
        },
        inputs={"__in": dace.Memlet("t[__i, __j]")},
        code="__out = __in + 1.0",
        outputs={"__out": dace.Memlet("o[__i + 1, __j + 1]")},
        input_nodes={t},
        output_nodes={o},
        external_edges=True,
    )
    sdfg.validate()

    return sdfg, state, g1, t, g2, o


def _make_concat_where_like_with_silent_write_to_g1(
    whole_write_back: bool,
    bigger_silent_write: bool,
) -> tuple[
    dace.SDFG,
    dace.SDFGState,
    dace_nodes.AccessNode,
    dace_nodes.AccessNode,
    dace_nodes.AccessNode,
    dace_nodes.AccessNode,
]:
    sdfg = dace.SDFG(util.unique_name(f"self_copy_sdfg_concat_where_like_multiple_writes_to_g1"))
    state = sdfg.add_state(is_start_block=True)

    sdfg.add_array(
        "g",
        shape=(10, 10),
        dtype=dace.float64,
        transient=False,
    )
    sdfg.add_array(
        "o",
        shape=(10, 12),
        dtype=dace.float64,
        transient=False,
    )
    sdfg.add_array(
        "t",
        shape=(8, 9),
        dtype=dace.float64,
        transient=True,
    )

    g1, t, g2, o = (state.add_access(name) for name in "gtgo")

    state.add_nedge(g1, t, dace.Memlet("g[2:10, 0] -> [0:8, 0]"))

    if whole_write_back:
        state.add_nedge(t, g2, dace.Memlet("t[0:4, 0:9] -> [2:6, 0:9]"))
        state.add_nedge(t, g2, dace.Memlet("t[4:8, 0:9] -> [6:10, 0:9]"))
    else:
        state.add_nedge(t, g2, dace.Memlet("t[0:4, 1:9] -> [2:6, 1:9]"))
        state.add_nedge(t, g2, dace.Memlet("t[4:8, 1:9] -> [6:10, 1:9]"))

    state.add_mapped_tasklet(
        "silent_write_to_g1",
        map_ranges={"__i": "0:10" if bigger_silent_write else "2:10"},
        inputs={},
        code="__out = -(__i / 4.0)",
        outputs={"__out": dace.Memlet("g[__i, 9]")},
        output_nodes={g1},
        external_edges=True,
    )
    state.add_mapped_tasklet(
        "first_level",
        map_ranges={"__i": "2:10"},
        inputs={},
        code="__out = __i / 3.0",
        outputs={"__out": dace.Memlet("g[__i, 0]")},
        output_nodes={g1},
        external_edges=True,
    )
    state.add_mapped_tasklet(
        "bulk_value",
        map_ranges={
            "__i": "0:8",
            "__j": "0:9",
        },
        inputs={},
        code="__out = sin(__i + 0.5)**2 + cos(__j + 0.1) ** 2",
        outputs={"__out": dace.Memlet("t[__i, __j]")},
        output_nodes={t},
        external_edges=True,
    )
    state.add_mapped_tasklet(
        "consumer",
        map_ranges={
            "__i": "0:8",
            "__j": "0:9",
        },
        inputs={"__in": dace.Memlet("t[__i, __j]")},
        code="__out = __in + 1.0",
        outputs={"__out": dace.Memlet("o[__i + 1, __j + 1]")},
        input_nodes={t},
        output_nodes={o},
        external_edges=True,
    )
    sdfg.validate()

    return sdfg, state, g1, t, g2, o


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

    Note:
        This case is currently not implemented, it was handled by the previous
        version, but it was kind of a special case that was not important.
        Thus we keep the test but change it such that something else is tested.
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
    assert count == 0


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

    util.compile_and_run_sdfg(sdfg, **ref)

    count = sdfg.apply_transformations_repeated(
        gtx_transformations.SingleStateGlobalDirectSelfCopyElimination,
        validate=True,
        validate_all=True,
    )

    ac_nodes_after = util.count_nodes(sdfg, dace_nodes.AccessNode, return_nodes=True)
    assert count == 1
    assert len(ac_nodes_after) == 4
    assert len(set(ac.data for ac in ac_nodes_after)) == 4

    util.compile_and_run_sdfg(sdfg, **res)
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
        node_g1=node_read_g,
        node_tmp=node_tmp,
        node_g2=node_write_g,
    )

    sdfg.validate()
    assert util.count_nodes(sdfg, dace_nodes.AccessNode) == 0


@pytest.mark.parametrize("handle_last_level", [True, False])
def test_global_self_copy_elimination_concat_where_like(
    handle_last_level: bool,
) -> None:
    sdfg, state, g1, t, g2, o = _make_concat_where_like(
        handle_last_level=handle_last_level,
        whole_write_back=False,
    )
    initial_ac_nodes = util.count_nodes(state, dace_nodes.AccessNode, True)
    assert len(initial_ac_nodes) == 4

    res, ref = util.make_sdfg_args(sdfg)
    util.compile_and_run_sdfg(sdfg, **ref)

    count = sdfg.apply_transformations_repeated(
        gtx_transformations.SingleStateGlobalSelfCopyElimination,
        validate=True,
        validate_all=True,
    )

    ac_nodes = util.count_nodes(state, dace_nodes.AccessNode, True)
    assert count == 1
    assert {g2, o} == set(ac_nodes)
    assert util.count_nodes(state, dace_nodes.MapExit) == 3

    util.compile_and_run_sdfg(sdfg, **res)
    assert util.compare_sdfg_res(ref, res)


def test_global_self_copy_elimination_concat_where_whole_write_back() -> None:
    sdfg, state, g1, t, g2, o = _make_concat_where_like(
        handle_last_level=True,
        whole_write_back=True,
    )
    initial_ac_nodes = util.count_nodes(state, dace_nodes.AccessNode, True)
    assert len(initial_ac_nodes) == 4

    res, ref = util.make_sdfg_args(sdfg)
    util.compile_and_run_sdfg(sdfg, **ref)

    count = sdfg.apply_transformations_repeated(
        gtx_transformations.SingleStateGlobalSelfCopyElimination,
        validate=True,
        validate_all=True,
    )

    ac_nodes = util.count_nodes(state, dace_nodes.AccessNode, True)
    assert count == 1
    assert {g2, o} == set(ac_nodes)
    assert util.count_nodes(state, dace_nodes.MapExit) == 3

    util.compile_and_run_sdfg(sdfg, **res)
    assert util.compare_sdfg_res(ref, res)


def test_global_self_copy_elimination_concat_where_like_silent_write_g1():
    sdfg, state, g1, t, g2, o = _make_concat_where_like_with_silent_write_to_g1(
        whole_write_back=False,
        bigger_silent_write=False,
    )
    initial_ac_nodes = util.count_nodes(state, dace_nodes.AccessNode, True)
    assert len(initial_ac_nodes) == 4

    res, ref = util.make_sdfg_args(sdfg)
    util.compile_and_run_sdfg(sdfg, **ref)

    count = sdfg.apply_transformations_repeated(
        gtx_transformations.SingleStateGlobalSelfCopyElimination,
        validate=True,
        validate_all=True,
    )

    ac_nodes = util.count_nodes(state, dace_nodes.AccessNode, True)
    assert count == 1

    util.compile_and_run_sdfg(sdfg, **res)
    assert util.compare_sdfg_res(ref, res)


def test_global_self_copy_elimination_concat_where_like_silent_write_g1_whole_write_back():
    assert False
