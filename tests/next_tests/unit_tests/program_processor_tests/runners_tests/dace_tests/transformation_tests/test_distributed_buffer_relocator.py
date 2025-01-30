# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import numpy as np

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
)

from . import util


# dace = pytest.importorskip("dace")
from dace.sdfg import nodes as dace_nodes
import dace


def _mk_distributed_buffer_sdfg() -> tuple[dace.SDFG, dace.SDFGState, dace.SDFGState]:
    sdfg = dace.SDFG(util.unique_name("distributed_buffer_sdfg"))

    for name in ["a", "b", "tmp"]:
        sdfg.add_array(name, shape=(10, 10), dtype=dace.float64, transient=False)
    sdfg.arrays["tmp"].transient = True
    sdfg.arrays["b"].shape = (100, 100)

    state1: dace.SDFGState = sdfg.add_state(is_start_block=True)
    state1.add_mapped_tasklet(
        "computation",
        map_ranges={"__i1": "0:10", "__i2": "0:10"},
        inputs={"__in": dace.Memlet("a[__i1, __i2]")},
        code="__out = __in + 10.0",
        outputs={"__out": dace.Memlet("tmp[__i1, __i2]")},
        external_edges=True,
    )

    state2 = sdfg.add_state_after(state1)
    state2_tskl = state2.add_tasklet(
        name="empty_blocker_tasklet",
        inputs={},
        code="pass",
        outputs={"__out"},
        side_effects=True,
    )
    state2.add_edge(
        state2_tskl,
        "__out",
        state2.add_access("a"),
        None,
        dace.Memlet("a[0, 0]"),
    )

    state3 = sdfg.add_state_after(state2)
    state3.add_edge(
        state3.add_access("tmp"),
        None,
        state3.add_access("b"),
        None,
        dace.Memlet("tmp[0:10, 0:10] -> [11:21, 22:32]"),
    )
    sdfg.validate()
    assert sdfg.number_of_nodes() == 3

    return sdfg, state1, state3


def test_distributed_buffer_remover():
    sdfg, state1, state3 = _mk_distributed_buffer_sdfg()
    assert state1.number_of_nodes() == 5
    assert not any(dnode.data == "b" for dnode in state1.data_nodes())

    res = gtx_transformations.gt_reduce_distributed_buffering(sdfg)
    assert res[sdfg]["DistributedBufferRelocator"][state3] == {"tmp"}

    # Because the final state has now become empty
    assert sdfg.number_of_nodes() == 3
    assert state1.number_of_nodes() == 6
    assert any(dnode.data == "b" for dnode in state1.data_nodes())
    assert any(dnode.data == "tmp" for dnode in state1.data_nodes())


def _make_distributed_buffer_global_memory_data_race_sdfg() -> tuple[dace.SDFG, dace.SDFGState]:
    sdfg = dace.SDFG(util.unique_name("distributed_buffer_global_memory_data_race"))
    arr_names = ["a", "b", "t"]
    for name in arr_names:
        sdfg.add_array(
            name=name,
            shape=(10, 10),
            dtype=dace.float64,
            transient=False,
        )
    sdfg.arrays["t"].transient = True

    state1 = sdfg.add_state(is_start_block=True)
    state2 = sdfg.add_state_after(state1)

    a_state1 = state1.add_access("a")
    state1.add_mapped_tasklet(
        "computation",
        map_ranges={"__i0": "0:10", "__i1": "0:10"},
        inputs={"__in": dace.Memlet("a[__i0, __i1]")},
        code="__out = __in + 10",
        outputs={"__out": dace.Memlet("t[__i0, __i1]")},
        input_nodes={a_state1},
        external_edges=True,
    )
    state1.add_nedge(a_state1, state1.add_access("b"), dace.Memlet("a[0:10, 0:10]"))

    state2.add_nedge(state2.add_access("t"), state2.add_access("a"), dace.Memlet("t[0:10, 0:10]"))
    sdfg.validate()

    return sdfg, state2


def test_distributed_buffer_global_memory_data_race():
    """Tests if the transformation realized that it would create a data race.

    If the transformation would apply, then `a` is read twice, once from two
    different branches, whose order of execution is indeterminate.
    """
    sdfg, state2 = _make_distributed_buffer_global_memory_data_race_sdfg()
    assert state2.number_of_nodes() == 2

    sdfg.simplify()
    assert sdfg.number_of_nodes() == 2

    res = gtx_transformations.gt_reduce_distributed_buffering(sdfg)
    assert "DistributedBufferRelocator" not in res[sdfg]
    assert state2.number_of_nodes() == 2


def _make_distributed_buffer_global_memory_data_race_sdfg2() -> (
    tuple[dace.SDFG, dace.SDFGState, dace.SDFGState]
):
    sdfg = dace.SDFG(util.unique_name("distributed_buffer_global_memory_data_race2_sdfg"))
    arr_names = ["a", "b", "t"]
    for name in arr_names:
        sdfg.add_array(
            name=name,
            shape=(10, 10),
            dtype=dace.float64,
            transient=False,
        )
    sdfg.arrays["t"].transient = True

    state1 = sdfg.add_state(is_start_block=True)
    state2 = sdfg.add_state_after(state1)

    state1.add_mapped_tasklet(
        "computation1",
        map_ranges={"__i0": "0:10", "__i1": "0:10"},
        inputs={"__in": dace.Memlet("a[__i0, __i1]")},
        code="__out = __in + 10",
        outputs={"__out": dace.Memlet("t[__i0, __i1]")},
        external_edges=True,
    )
    state1.add_mapped_tasklet(
        "computation1",
        map_ranges={"__i0": "0:10", "__i1": "0:10"},
        inputs={"__in": dace.Memlet("a[__i0, __i1]")},
        code="__out = __in - 10",
        outputs={"__out": dace.Memlet("b[__i0, __i1]")},
        external_edges=True,
    )
    state2.add_nedge(state2.add_access("t"), state2.add_access("a"), dace.Memlet("t[0:10, 0:10]"))
    sdfg.validate()

    return sdfg, state1, state2


def test_distributed_buffer_global_memory_data_race2():
    """Tests if the transformation realized that it would create a data race.

    Similar situation but now there are two different subgraphs. This is needed
    because it is another branch that checks it.
    """
    sdfg, state1, state2 = _make_distributed_buffer_global_memory_data_race_sdfg2()
    assert state1.number_of_nodes() == 10
    assert state2.number_of_nodes() == 2

    res = gtx_transformations.gt_reduce_distributed_buffering(sdfg)
    assert "DistributedBufferRelocator" not in res[sdfg]
    assert state1.number_of_nodes() == 10
    assert state2.number_of_nodes() == 2


def _make_distributed_buffer_global_memory_data_no_rance() -> tuple[dace.SDFG, dace.SDFGState]:
    sdfg = dace.SDFG(util.unique_name("distributed_buffer_global_memory_data_no_rance_sdfg"))
    arr_names = ["a", "t"]
    for name in arr_names:
        sdfg.add_array(
            name=name,
            shape=(10, 10),
            dtype=dace.float64,
            transient=False,
        )
    sdfg.arrays["t"].transient = True

    state1 = sdfg.add_state(is_start_block=True)
    state2 = sdfg.add_state_after(state1)

    a_state1 = state1.add_access("a")
    state1.add_mapped_tasklet(
        "computation",
        map_ranges={"__i0": "0:10", "__i1": "0:10"},
        inputs={"__in": dace.Memlet("a[__i0, __i1]")},
        code="__out = __in + 10",
        outputs={"__out": dace.Memlet("t[__i0, __i1]")},
        input_nodes={a_state1},
        external_edges=True,
    )

    state2.add_nedge(state2.add_access("t"), state2.add_access("a"), dace.Memlet("t[0:10, 0:10]"))
    sdfg.validate()

    return sdfg, state2


def test_distributed_buffer_global_memory_data_no_rance():
    """Transformation applies if there is no data race.

    According to ADR18, pointwise dependencies are fine. This tests checks if the
    checks for the read-write conflicts are not too strong.
    """
    sdfg, state2 = _make_distributed_buffer_global_memory_data_no_rance()
    assert state2.number_of_nodes() == 2

    res = gtx_transformations.gt_reduce_distributed_buffering(sdfg)
    assert res[sdfg]["DistributedBufferRelocator"][state2] == {"t"}
    assert state2.number_of_nodes() == 0


def _make_distributed_buffer_global_memory_data_no_rance2() -> tuple[dace.SDFG, dace.SDFGState]:
    sdfg = dace.SDFG(util.unique_name("distributed_buffer_global_memory_data_no_rance2_sdfg"))
    arr_names = ["a", "t"]
    for name in arr_names:
        sdfg.add_array(
            name=name,
            shape=(10, 10),
            dtype=dace.float64,
            transient=False,
        )
    sdfg.arrays["t"].transient = True

    state1 = sdfg.add_state(is_start_block=True)
    state2 = sdfg.add_state_after(state1)

    a_state1 = state1.add_access("a")
    state1.add_mapped_tasklet(
        "computation1",
        map_ranges={"__i0": "0:10", "__i1": "0:10"},
        inputs={"__in": dace.Memlet("a[__i0, __i1]")},
        code="__out = __in + 10",
        outputs={"__out": dace.Memlet("a[__i0, __i1]")},
        output_nodes={a_state1},
        external_edges=True,
    )
    state1.add_mapped_tasklet(
        "computation2",
        map_ranges={"__i0": "0:10", "__i1": "0:10"},
        inputs={"__in": dace.Memlet("a[__i0, __i1]")},
        code="__out = __in + 10",
        outputs={"__out": dace.Memlet("t[__i0, __i1]")},
        input_nodes={a_state1},
        external_edges=True,
    )

    state2.add_nedge(state2.add_access("t"), state2.add_access("a"), dace.Memlet("t[0:10, 0:10]"))
    sdfg.validate()

    return sdfg, state2


def test_distributed_buffer_global_memory_data_no_rance2():
    """Transformation applies if there is no data race.

    These dependency is fine, because the access nodes are in a clear serial order.
    """
    sdfg, state2 = _make_distributed_buffer_global_memory_data_no_rance2()
    assert state2.number_of_nodes() == 2

    res = gtx_transformations.gt_reduce_distributed_buffering(sdfg)
    assert res[sdfg]["DistributedBufferRelocator"][state2] == {"t"}
    assert state2.number_of_nodes() == 0


def _make_distributed_buffer_non_sink_temporary_sdfg() -> (
    tuple[dace.SDFG, dace.SDFGState, dace.SDFGState]
):
    sdfg = dace.SDFG(util.unique_name("distributed_buffer_non_sink_temporary_sdfg"))
    state = sdfg.add_state(is_start_block=True)
    wb_state = sdfg.add_state_after(state)

    names = ["a", "b", "c", "t1", "t2"]
    for name in names:
        sdfg.add_array(
            name,
            shape=(10,),
            dtype=dace.float64,
            transient=False,
        )
    sdfg.arrays["t1"].transient = True
    sdfg.arrays["t2"].transient = True
    t1 = state.add_access("t1")

    state.add_mapped_tasklet(
        "comp1",
        map_ranges={"__i": "0:10"},
        inputs={"__in1": dace.Memlet("a[__i]")},
        code="__out = __in1 + 10.0",
        outputs={"__out": dace.Memlet("t1[__i]")},
        output_nodes={t1},
        external_edges=True,
    )
    state.add_mapped_tasklet(
        "comp2",
        map_ranges={"__i": "0:10"},
        inputs={"__in1": dace.Memlet("t1[__i]")},
        code="__out = __in1 / 2.0",
        outputs={"__out": dace.Memlet("t2[__i]")},
        input_nodes={t1},
        external_edges=True,
    )

    wb_state.add_nedge(wb_state.add_access("t1"), wb_state.add_access("b"), dace.Memlet("t1[0:10]"))
    wb_state.add_nedge(wb_state.add_access("t2"), wb_state.add_access("b"), dace.Memlet("t2[0:10]"))

    sdfg.validate()
    return sdfg, state, wb_state


def test_distributed_buffer_non_sink_temporary():
    """Tests the transformation if one of the temporaries is not a sink node.

    Note that the SDFG has two temporaries, `t1` is not a sink node and `t2` is
    a sink node.
    """
    sdfg, state, wb_state = _make_distributed_buffer_non_sink_temporary_sdfg()
    assert wb_state.number_of_nodes() == 4

    res = gtx_transformations.gt_reduce_distributed_buffering(sdfg)
    sdfg.view()
    assert res[sdfg]["DistributedBufferRelocator"][wb_state] == {"t1", "t2"}
    assert wb_state.number_of_nodes() == 0
