# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dace
import pytest
import numpy as np
import copy

from dace.sdfg import nodes as dace_nodes
from dace import data as dace_data
from dace.transformation import dataflow as dace_dftrafo

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
)

from . import util

import dace


def _create_sdfg_double_read_part_1(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    me: dace.nodes.MapEntry,
    mx: dace.nodes.MapExit,
    A_in: dace.nodes.AccessNode,
    nb: int,
) -> dace.nodes.Tasklet:
    tskl = state.add_tasklet(
        name=f"tasklet_1", inputs={"__in1"}, outputs={"__out"}, code="__out = __in1 + 1.0"
    )

    state.add_edge(A_in, None, me, f"IN_{nb}", dace.Memlet("A[0:10]"))
    state.add_edge(me, f"OUT_{nb}", tskl, "__in1", dace.Memlet("A[__i0]"))
    me.add_scope_connectors(str(nb))

    state.add_edge(tskl, "__out", mx, f"IN_{nb}", dace.Memlet("A[__i0]"))
    state.add_edge(mx, f"OUT_{nb}", state.add_access("A"), None, dace.Memlet("A[0:10]"))
    mx.add_scope_connectors(str(nb))


def _create_sdfg_double_read_part_2(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    me: dace.nodes.MapEntry,
    mx: dace.nodes.MapExit,
    A_in: dace.nodes.AccessNode,
    nb: int,
) -> dace.nodes.Tasklet:
    tskl = state.add_tasklet(
        name=f"tasklet_2", inputs={"__in1"}, outputs={"__out"}, code="__out = __in1 + 3.0"
    )

    state.add_edge(A_in, None, me, f"IN_{nb}", dace.Memlet("A[0:10]"))
    state.add_edge(me, f"OUT_{nb}", tskl, "__in1", dace.Memlet("A[__i0]"))
    me.add_scope_connectors(str(nb))

    state.add_edge(tskl, "__out", mx, f"IN_{nb}", dace.Memlet("B[__i0]"))
    state.add_edge(mx, f"OUT_{nb}", state.add_access("B"), None, dace.Memlet("B[0:10]"))
    mx.add_scope_connectors(str(nb))


def _create_sdfg_double_read(
    version: int,
) -> dace.SDFG:
    sdfg = dace.SDFG(util.unique_name(f"double_read_version_{version}"))
    state = sdfg.add_state(is_start_block=True)
    for name in "AB":
        sdfg.add_array(
            name,
            shape=(10,),
            dtype=dace.float64,
            transient=False,
        )
    A_in = state.add_access("A")
    me, mx = state.add_map("map", ndrange={"__i0": "0:10"})

    if version == 0:
        _create_sdfg_double_read_part_1(sdfg, state, me, mx, A_in, 0)
        _create_sdfg_double_read_part_2(sdfg, state, me, mx, A_in, 1)
    elif version == 1:
        _create_sdfg_double_read_part_1(sdfg, state, me, mx, A_in, 1)
        _create_sdfg_double_read_part_2(sdfg, state, me, mx, A_in, 0)
    else:
        raise ValueError(f"Does not know version {version}")
    sdfg.validate()
    return sdfg


def _create_non_scalar_read() -> dace.SDFG:
    sdfg = dace.SDFG(util.unique_name(f"non_scalar_read_sdfg"))
    state = sdfg.add_state(is_start_block=True)

    sdfg.add_array(
        name="A",
        shape=(10, 10),
        dtype=dace.float64,
        transient=False,
    )

    state.add_mapped_tasklet(
        "comp",
        map_ranges={
            "__i": "0:10",
            "__j": "0:10",
        },
        inputs={"__in": dace.Memlet("A[__i, __j]")},
        code="__out = __in + 10.0",
        outputs={"__out": dace.Memlet("A[__i, __j]")},
        external_edges=True,
    )
    sdfg.apply_transformations(dace_dftrafo.MapExpansion)
    sdfg.validate()

    return sdfg


def test_local_double_buffering_double_read_sdfg():
    sdfg0 = _create_sdfg_double_read(0)
    sdfg1 = _create_sdfg_double_read(1)
    args0 = {name: np.array(np.random.rand(10), dtype=np.float64, copy=True) for name in "AB"}
    args1 = copy.deepcopy(args0)

    count0 = gtx_transformations.gt_create_local_double_buffering(sdfg0)
    assert count0 == 1

    count1 = gtx_transformations.gt_create_local_double_buffering(sdfg1)
    assert count1 == 1

    sdfg0(**args0)
    sdfg1(**args1)
    for name in args0:
        assert np.allclose(args0[name], args1[name]), f"Failed verification in '{name}'."


def test_local_double_buffering_no_connection():
    """There is no direct connection between read and write."""
    sdfg = dace.SDFG(util.unique_name("local_double_buffering_no_connection"))
    state = sdfg.add_state(is_start_block=True)
    for name in "AB":
        sdfg.add_array(
            name,
            shape=(10,),
            dtype=dace.float64,
            transient=False,
        )
    A_in, B, A_out = (state.add_access(name) for name in "ABA")

    comp_tskl, me, mx = state.add_mapped_tasklet(
        "computation",
        map_ranges={"__i0": "0:10"},
        inputs={"__in1": dace.Memlet("A[__i0]")},
        code="__out = __in1 + 10.0",
        outputs={"__out": dace.Memlet("B[__i0]")},
        input_nodes={A_in},
        output_nodes={B},
        external_edges=True,
    )

    fill_tasklet = state.add_tasklet(
        name="fill_tasklet",
        inputs=set(),
        code="__out = 2.",
        outputs={"__out"},
    )
    state.add_nedge(me, fill_tasklet, dace.Memlet())
    state.add_edge(fill_tasklet, "__out", mx, "IN_1", dace.Memlet("A[__i0]"))
    state.add_edge(mx, "OUT_1", A_out, None, dace.Memlet("A[0:10]"))
    mx.add_scope_connectors("1")
    sdfg.validate()

    count = gtx_transformations.gt_create_local_double_buffering(sdfg)
    assert count == 1

    # Ensure that a second application of the transformation does not run again.
    count_again = gtx_transformations.gt_create_local_double_buffering(sdfg)
    assert count_again == 0

    # Find the newly created access node.
    comp_tasklet_producers = [in_edge.src for in_edge in state.in_edges(comp_tskl)]
    assert len(comp_tasklet_producers) == 1
    new_double_buffer = comp_tasklet_producers[0]
    assert isinstance(new_double_buffer, dace_nodes.AccessNode)
    assert not any(new_double_buffer.data == name for name in "AB")
    assert isinstance(new_double_buffer.desc(sdfg), dace_data.Scalar)
    assert new_double_buffer.desc(sdfg).transient

    # The newly created access node, must have an empty Memlet to the fill tasklet.
    read_dependencies = [
        out_edge.dst for out_edge in state.out_edges(new_double_buffer) if out_edge.data.is_empty()
    ]
    assert len(read_dependencies) == 1
    assert read_dependencies[0] is fill_tasklet

    res = {name: np.array(np.random.rand(10), dtype=np.float64, copy=True) for name in "AB"}
    ref = {"A": np.full_like(res["A"], 2.0), "B": res["A"] + 10.0}
    sdfg(**res)
    for name in res:
        assert np.allclose(res[name], ref[name]), f"Failed verification in '{name}'."


def test_local_double_buffering_no_apply():
    """Here it does not apply, because are all distinct."""
    sdfg = dace.SDFG(util.unique_name("local_double_buffering_no_apply"))
    state = sdfg.add_state(is_start_block=True)
    for name in "AB":
        sdfg.add_array(
            name,
            shape=(10,),
            dtype=dace.float64,
            transient=False,
        )
    state.add_mapped_tasklet(
        "computation",
        map_ranges={"__i0": "0:10"},
        inputs={"__in1": dace.Memlet("A[__i0]")},
        code="__out = __in1 + 10.0",
        outputs={"__out": dace.Memlet("B[__i0]")},
        external_edges=True,
    )
    sdfg.validate()

    count = gtx_transformations.gt_create_local_double_buffering(sdfg)
    assert count == 0


def test_local_double_buffering_already_buffered():
    """It is already buffered."""
    sdfg = dace.SDFG(util.unique_name("local_double_buffering_no_apply"))
    state = sdfg.add_state(is_start_block=True)
    sdfg.add_array(
        "A",
        shape=(10,),
        dtype=dace.float64,
        transient=False,
    )

    tsklt, me, mx = state.add_mapped_tasklet(
        "computation",
        map_ranges={"__i0": "0:10"},
        inputs={"__in1": dace.Memlet("A[__i0]")},
        code="__out = __in1 + 10.0",
        outputs={"__out": dace.Memlet("A[__i0]")},
        external_edges=True,
    )

    sdfg.add_scalar("tmp", dtype=dace.float64, transient=True)
    tmp = state.add_access("tmp")
    me_to_tskl_edge = next(iter(state.out_edges(me)))

    state.add_edge(me, me_to_tskl_edge.src_conn, tmp, None, dace.Memlet("A[__i0]"))
    state.add_edge(tmp, None, tsklt, "__in1", dace.Memlet("tmp[0]"))
    state.remove_edge(me_to_tskl_edge)
    sdfg.validate()

    count = gtx_transformations.gt_create_local_double_buffering(sdfg)
    assert count == 0


def test_non_scalar_read():
    sdfg = _create_non_scalar_read()

    # Because of the nested Maps, the Memlet that connects the outer with the
    #  inner Map carries more than a scalar and thus the transformation
    #  does not apply.
    count = gtx_transformations.gt_create_local_double_buffering(sdfg)
    assert count == 0


def _make_war_hazard_sdfg() -> dace.SDFG:
    """Builds an SDFG with a write-after-read hazard pattern on `G`.

    A single Map contains two independent branches:
    - The writer branch produces the new value of `G` into the buffer `tmp`,
        which is written back to `G` after the Map: `O(i) = G(i) + 1.0` and
        `G(i) = 2.0 * A(i)`.
    - The reader branch reads the (old) value of `G` and forwards it,
        incremented, to the output `O`.

    This is the pattern generated by stencils that compute an auxiliary field
    from a field they also update in place, e.g.
    `extrapolate_temporally_exner_pressure`. Inlining the write to `G` into the
    Map is invalid here, because the reader branch is not dataflow-ordered
    before the writer branch, so it may observe the newly written value
    instead of the old one (WAR hazard inside the Map body).
    """
    sdfg = dace.SDFG(util.unique_name("map_buffer_war"))
    state = sdfg.add_state(is_start_block=True)

    for name in ("G", "A", "O"):
        sdfg.add_array(name, shape=(10,), dtype=dace.float64, transient=False)
    sdfg.add_array("tmp", shape=(10,), dtype=dace.float64, transient=True)

    G_read = state.add_access("G")
    A_read = state.add_access("A")
    O_write = state.add_access("O")
    tmp_write = state.add_access("tmp")
    G_write = state.add_access("G")

    map_entry, map_exit = state.add_map("map", ndrange={"__i0": "0:10"})

    t_write = state.add_tasklet("t_write", {"__in": None}, {"__out": None}, "__out = 2.0 * __in")
    state.add_memlet_path(
        A_read, map_entry, t_write, dst_conn="__in", memlet=dace.Memlet("A[__i0]")
    )
    state.add_memlet_path(
        t_write, map_exit, tmp_write, src_conn="__out", memlet=dace.Memlet("tmp[__i0]")
    )

    t_read = state.add_tasklet("t_read", {"__in": None}, {"__out": None}, "__out = __in + 1.0")
    state.add_memlet_path(G_read, map_entry, t_read, dst_conn="__in", memlet=dace.Memlet("G[__i0]"))
    state.add_memlet_path(
        t_read, map_exit, O_write, src_conn="__out", memlet=dace.Memlet("O[__i0]")
    )

    state.add_nedge(tmp_write, G_write, dace.Memlet("tmp[0:10] -> [0:10]"))
    sdfg.validate()
    return sdfg


def test_local_double_buffering_war_hazard():
    """Double buffering must fire after buffer elimination inlines a write
    that races with an independent read inside the same Map.

    The SDFG computes `O(i) = G(i) + 1.0` and `G(i) = 2.0 * A(i)`. After
    `GT4PyMapBufferElimination` inlines the write-back to `G` into the Map,
    `G` becomes both read and written inside the Map body. Without double
    buffering the reader may observe the newly written value instead of the
    old one (WAR hazard). `gt_create_local_double_buffering` must detect this
    and insert a local double buffer so the reader sees the original value.
    """
    sdfg = _make_war_hazard_sdfg()

    ref = {
        name: np.array(np.random.rand(10), dtype=np.float64, copy=True) for name in ("G", "A", "O")
    }
    expected = {
        "G": 2.0 * copy.deepcopy(ref["A"]),
        "O": copy.deepcopy(ref["G"]) + 1.0,
        "A": copy.deepcopy(ref["A"]),
    }

    count_elim = sdfg.apply_transformations_repeated(
        gtx_transformations.GT4PyMapBufferElimination(assume_pointwise=True),
        validate=True,
        validate_all=True,
    )
    assert count_elim >= 1, "Buffer elimination must fire to inline the write."

    count_db = gtx_transformations.gt_create_local_double_buffering(sdfg)
    assert count_db >= 1, "Double buffering must fire to prevent the WAR hazard."

    args = copy.deepcopy(ref)
    sdfg(**args)
    for name in args:
        assert np.allclose(args[name], expected[name]), f"Failed verification in '{name}'."
