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
from dace.sdfg import nodes as dace_nodes
from dace import data as dace_data

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
    me.add_in_connector(f"IN_{nb}")
    me.add_out_connector(f"OUT_{nb}")

    state.add_edge(tskl, "__out", mx, f"IN_{nb}", dace.Memlet("A[__i0]"))
    state.add_edge(mx, f"OUT_{nb}", state.add_access("A"), None, dace.Memlet("A[0:10]"))
    mx.add_in_connector(f"IN_{nb}")
    mx.add_out_connector(f"OUT_{nb}")


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
    me.add_in_connector(f"IN_{nb}")
    me.add_out_connector(f"OUT_{nb}")

    state.add_edge(tskl, "__out", mx, f"IN_{nb}", dace.Memlet("B[__i0]"))
    state.add_edge(mx, f"OUT_{nb}", state.add_access("B"), None, dace.Memlet("B[0:10]"))
    mx.add_in_connector(f"IN_{nb}")
    mx.add_out_connector(f"OUT_{nb}")


def _create_sdfg_double_read(
    version: int,
) -> tuple[dace.SDFG]:
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
    mx.add_in_connector("IN_1")
    mx.add_out_connector("OUT_1")
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
