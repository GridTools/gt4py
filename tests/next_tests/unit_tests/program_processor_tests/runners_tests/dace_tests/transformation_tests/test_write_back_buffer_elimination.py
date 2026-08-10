# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy

import numpy as np
import pytest

dace = pytest.importorskip("dace")

from dace.transformation import pass_pipeline as dace_ppl

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
)

from . import util


def _mk_write_back_buffer_sdfg(
    write_back_subset: str,
) -> tuple[dace.SDFG, dace.SDFGState, dace.SDFGState]:
    """`tmp` is written back into `b` and additionally read by a second consumer."""
    sdfg = dace.SDFG(util.unique_name("write_back_buffer_sdfg"))

    for name in ["a", "c"]:
        sdfg.add_array(name, shape=(10, 10), dtype=dace.float64, transient=False)
    sdfg.add_array("b", shape=(100, 100), dtype=dace.float64, transient=False)
    sdfg.add_array("tmp", shape=(10, 10), dtype=dace.float64, transient=True)

    state1: dace.SDFGState = sdfg.add_state(is_start_block=True)
    state1.add_mapped_tasklet(
        "producer",
        map_ranges={"__i1": "0:10", "__i2": "0:10"},
        inputs={"__in": dace.Memlet("a[__i1, __i2]")},
        code="__out = __in + 10.0",
        outputs={"__out": dace.Memlet("tmp[__i1, __i2]")},
        external_edges=True,
    )

    state2 = sdfg.add_state_after(state1)
    state2.add_edge(
        state2.add_access("tmp"),
        None,
        state2.add_access("b"),
        None,
        dace.Memlet(write_back_subset),
    )
    # This second consumer is what makes `DistributedBufferRelocator` and
    #  `GT4PyMapBufferElimination` bail out.
    state2.add_mapped_tasklet(
        "consumer",
        map_ranges={"__i1": "0:10", "__i2": "0:10"},
        inputs={"__in": dace.Memlet("tmp[__i1, __i2]")},
        code="__out = __in * 2.0",
        outputs={"__out": dace.Memlet("c[__i1, __i2]")},
        external_edges=True,
    )
    sdfg.validate()

    return sdfg, state1, state2


def _apply(sdfg: dace.SDFG) -> bool:
    res = dace_ppl.Pipeline(
        [gtx_transformations.GT4PyWriteBackBufferElimination(assume_pointwise=True)]
    ).apply_pass(sdfg, {})
    return bool(res and res.get("GT4PyWriteBackBufferElimination"))


def test_write_back_buffer_elimination():
    sdfg, state1, state2 = _mk_write_back_buffer_sdfg("tmp[0:10, 0:10] -> [11:21, 22:32]")
    ref, res = util.make_sdfg_args(sdfg)
    util.compile_and_run_sdfg(sdfg, **ref)

    assert _apply(sdfg)

    assert "tmp" not in sdfg.arrays
    assert not any(dnode.data == "tmp" for dnode in state1.data_nodes())
    assert not any(dnode.data == "tmp" for dnode in state2.data_nodes())
    # The producer now writes `b` directly, at the offset of the former write back.
    (producer_out,) = [
        edge for edge in state1.edges() if isinstance(edge.dst, dace.sdfg.nodes.AccessNode)
    ]
    assert producer_out.dst.data == "b"
    assert str(producer_out.data.subset) == "11:21, 22:32"

    util.compile_and_run_sdfg(sdfg, **res)
    assert util.compare_sdfg_res(ref, res)


def test_write_back_buffer_elimination_partial_write_back():
    """A partially copied buffer must be kept.

    Writing `b` directly would touch `b[16:21, 22:32]`, which is outside the range
    that the write back covers.
    """
    sdfg, _, _ = _mk_write_back_buffer_sdfg("tmp[0:5, 0:10] -> [11:16, 22:32]")
    assert not _apply(sdfg)
    assert "tmp" in sdfg.arrays


def test_write_back_buffer_elimination_nested_sdfg_strides():
    """The strides of a NestedSDFG reading `tmp` must follow `tmp` becoming `b`."""
    sdfg = dace.SDFG(util.unique_name("write_back_nested_strides"))

    sdfg.add_array("a", shape=(10, 10), dtype=dace.float64, transient=False)
    sdfg.add_array("c", shape=(10, 10), dtype=dace.float64, transient=False)
    # `b` is strided differently than the contiguous `tmp` it replaces.
    sdfg.add_array("b", shape=(100, 100), dtype=dace.float64, transient=False)
    sdfg.add_array("tmp", shape=(10, 10), dtype=dace.float64, transient=True)

    state: dace.SDFGState = sdfg.add_state(is_start_block=True)
    tmp = state.add_access("tmp")
    state.add_mapped_tasklet(
        "producer",
        map_ranges={"__i1": "0:10", "__i2": "0:10"},
        inputs={"__in": dace.Memlet("a[__i1, __i2]")},
        code="__out = __in + 10.0",
        outputs={"__out": dace.Memlet("tmp[__i1, __i2]")},
        output_nodes={tmp},
        external_edges=True,
    )

    # A NestedSDFG consumes `tmp` as a whole; its inner descriptor inherits the
    #  strides of `tmp` and has to be updated when the accesses are moved to `b`.
    inner = dace.SDFG(util.unique_name("inner"))
    inner.add_array("inp", shape=(10, 10), dtype=dace.float64, transient=False)
    inner.add_array("outp", shape=(10, 10), dtype=dace.float64, transient=False)
    inner_state = inner.add_state(is_start_block=True)
    inner_state.add_mapped_tasklet(
        "inner_comp",
        map_ranges={"__i1": "0:10", "__i2": "0:10"},
        inputs={"__in": dace.Memlet("inp[__i1, __i2]")},
        code="__out = __in * 2.0",
        outputs={"__out": dace.Memlet("outp[__i1, __i2]")},
        external_edges=True,
    )
    nsdfg = state.add_nested_sdfg(inner, {"inp"}, {"outp"})
    state.add_edge(tmp, None, nsdfg, "inp", dace.Memlet("tmp[0:10, 0:10]"))
    state.add_edge(nsdfg, "outp", state.add_access("c"), None, dace.Memlet("c[0:10, 0:10]"))

    state2: dace.SDFGState = sdfg.add_state_after(state)
    state2.add_nedge(
        state2.add_access("tmp"),
        state2.add_access("b"),
        dace.Memlet("tmp[0:10, 0:10] -> [0:10, 0:10]"),
    )
    sdfg.validate()

    ref = {
        name: np.array(np.random.rand(*desc.shape), copy=True, dtype=desc.dtype.as_numpy_dtype())
        for name, desc in sdfg.arrays.items()
        if not desc.transient
    }
    res = copy.deepcopy(ref)
    util.compile_and_run_sdfg(sdfg, **ref)

    res_pipeline = dace_ppl.Pipeline(
        [gtx_transformations.GT4PyWriteBackBufferElimination()]
    ).apply_pass(sdfg, {})
    assert res_pipeline is not None
    assert "tmp" not in sdfg.arrays

    util.compile_and_run_sdfg(sdfg, **res)
    assert all(np.allclose(ref[name], res[name]) for name in ref.keys())


def test_write_back_buffer_elimination_unrelated_reader():
    """`assume_pointwise` must not waive a read of `b` by an unrelated Map."""
    sdfg = dace.SDFG(util.unique_name("write_back_unrelated_reader"))

    for name in ["a", "d"]:
        sdfg.add_array(name, shape=(20,), dtype=dace.float64, transient=False)
    sdfg.add_array("b", shape=(20,), dtype=dace.float64, transient=False)
    sdfg.add_array("tmp", shape=(20,), dtype=dace.float64, transient=True)

    state: dace.SDFGState = sdfg.add_state(is_start_block=True)
    state.add_mapped_tasklet(
        "producer",
        map_ranges={"__i": "0:20"},
        inputs={"__in": dace.Memlet("a[__i]")},
        code="__out = __in + 10.0",
        outputs={"__out": dace.Memlet("tmp[__i]")},
        external_edges=True,
    )
    # An independent Map reads `b` in the same state; it is unordered with respect
    #  to the producer, so writing `b` directly would race with it.
    state.add_mapped_tasklet(
        "unrelated_reader",
        map_ranges={"__i": "0:20"},
        inputs={"__in": dace.Memlet("b[19 - __i]")},
        code="__out = __in * 5.0",
        outputs={"__out": dace.Memlet("d[__i]")},
        external_edges=True,
    )

    state2: dace.SDFGState = sdfg.add_state_after(state)
    state2.add_nedge(
        state2.add_access("tmp"), state2.add_access("b"), dace.Memlet("tmp[0:20] -> [0:20]")
    )
    sdfg.validate()

    dace_ppl.Pipeline(
        [gtx_transformations.GT4PyWriteBackBufferElimination(assume_pointwise=True)]
    ).apply_pass(sdfg, {})
    assert "tmp" in sdfg.arrays, "the pass must not fire when an unrelated Map reads 'b'"


def _mk_map_consumer_sdfg(inner_memlet_uses_tmp: bool) -> dace.SDFG:
    """`tmp` is written back into `b` and additionally consumed inside nested Maps."""
    sdfg = dace.SDFG(util.unique_name("write_back_map_consumer"))

    sdfg.add_array("a", shape=(10, 10), dtype=dace.float64, transient=False)
    sdfg.add_array("c", shape=(10, 10), dtype=dace.float64, transient=False)
    sdfg.add_array("b", shape=(100, 100), dtype=dace.float64, transient=False)
    sdfg.add_array("tmp", shape=(10, 10), dtype=dace.float64, transient=True)
    sdfg.add_array(
        "local",
        shape=(10,),
        dtype=dace.float64,
        transient=True,
        storage=dace.dtypes.StorageType.Register,
        lifetime=dace.dtypes.AllocationLifetime.Scope,
    )

    state1: dace.SDFGState = sdfg.add_state(is_start_block=True)
    state1.add_mapped_tasklet(
        "producer",
        map_ranges={"__i1": "0:10", "__i2": "0:10"},
        inputs={"__in": dace.Memlet("a[__i1, __i2]")},
        code="__out = __in + 10.0",
        outputs={"__out": dace.Memlet("tmp[__i1, __i2]")},
        external_edges=True,
    )

    state2 = sdfg.add_state_after(state1)
    state2.add_nedge(
        state2.add_access("tmp"),
        state2.add_access("b"),
        dace.Memlet("tmp[0:10, 0:10] -> [11:21, 22:32]"),
    )

    tmp_node = state2.add_access("tmp")
    outer_entry, outer_exit = state2.add_map("outer", ndrange={"__i1": "0:10"})
    inner_entry, inner_exit = state2.add_map("inner", ndrange={"__i2": "0:10"})
    tasklet = state2.add_tasklet(
        "comp", inputs={"__in"}, outputs={"__out"}, code="__out = __in * 2"
    )
    local_node = state2.add_access("local")
    c_node = state2.add_access("c")

    state2.add_edge(tmp_node, None, outer_entry, "IN_tmp", dace.Memlet("tmp[__i1, 0:10]"))
    outer_entry.add_in_connector("IN_tmp")
    outer_entry.add_out_connector("OUT_tmp")
    if inner_memlet_uses_tmp:
        inner_copy_memlet = dace.Memlet("tmp[__i1, 0:10] -> [0:10]")
    else:
        # The very same copy, but expressed relative to `local`; the subset that
        #  refers to `tmp` is now the memlet's `other_subset`.
        inner_copy_memlet = dace.Memlet(data="local", subset="0:10", other_subset="__i1, 0:10")
    state2.add_edge(outer_entry, "OUT_tmp", local_node, None, inner_copy_memlet)

    state2.add_edge(local_node, None, inner_entry, "IN_local", dace.Memlet("local[0:10]"))
    inner_entry.add_in_connector("IN_local")
    inner_entry.add_out_connector("OUT_local")
    state2.add_edge(inner_entry, "OUT_local", tasklet, "__in", dace.Memlet("local[__i2]"))
    state2.add_edge(tasklet, "__out", inner_exit, "IN_c", dace.Memlet("c[__i1, __i2]"))
    inner_exit.add_in_connector("IN_c")
    inner_exit.add_out_connector("OUT_c")
    state2.add_edge(inner_exit, "OUT_c", outer_exit, "IN_c", dace.Memlet("c[__i1, 0:10]"))
    outer_exit.add_in_connector("IN_c")
    outer_exit.add_out_connector("OUT_c")
    state2.add_edge(outer_exit, "OUT_c", c_node, None, dace.Memlet("c[0:10, 0:10]"))

    sdfg.validate()
    return sdfg


@pytest.mark.parametrize("inner_memlet_uses_tmp", [True, False])
def test_write_back_buffer_elimination_map_consumer(inner_memlet_uses_tmp):
    sdfg = _mk_map_consumer_sdfg(inner_memlet_uses_tmp)
    ref, res = util.make_sdfg_args(sdfg)
    util.compile_and_run_sdfg(sdfg, **ref)

    assert _apply(sdfg)
    assert "tmp" not in sdfg.arrays
    sdfg.validate()

    util.compile_and_run_sdfg(sdfg, **res)
    assert util.compare_sdfg_res(ref, res)


def test_write_back_buffer_elimination_pointwise_reader():
    """`b` is read elementwise by the producer of `tmp`, i.e. ADR-18 rule 3."""
    sdfg = dace.SDFG(util.unique_name("write_back_pointwise_reader"))

    sdfg.add_array("a", shape=(20,), dtype=dace.float64, transient=False)
    sdfg.add_array("c", shape=(20,), dtype=dace.float64, transient=False)
    sdfg.add_array("b", shape=(20,), dtype=dace.float64, transient=False)
    sdfg.add_array("tmp", shape=(20,), dtype=dace.float64, transient=True)

    state1: dace.SDFGState = sdfg.add_state(is_start_block=True)
    state1.add_mapped_tasklet(
        "producer",
        map_ranges={"__i": "0:20"},
        inputs={"__in_a": dace.Memlet("a[__i]"), "__in_b": dace.Memlet("b[__i]")},
        code="__out = __in_a + __in_b",
        outputs={"__out": dace.Memlet("tmp[__i]")},
        external_edges=True,
    )

    state2: dace.SDFGState = sdfg.add_state_after(state1)
    state2.add_nedge(
        state2.add_access("tmp"), state2.add_access("b"), dace.Memlet("tmp[0:20] -> [0:20]")
    )
    state2.add_mapped_tasklet(
        "consumer",
        map_ranges={"__i": "0:20"},
        inputs={"__in": dace.Memlet("tmp[__i]")},
        code="__out = __in * 2.0",
        outputs={"__out": dace.Memlet("c[__i]")},
        external_edges=True,
    )
    sdfg.validate()

    ref, res = util.make_sdfg_args(sdfg)
    util.compile_and_run_sdfg(sdfg, **ref)

    assert _apply(sdfg)
    assert "tmp" not in sdfg.arrays
    # The read of `b` and the write to `b` must use two distinct AccessNodes, a
    #  single one would make the state cyclic.
    assert len([dnode for dnode in state1.data_nodes() if dnode.data == "b"]) == 2
    sdfg.validate()

    util.compile_and_run_sdfg(sdfg, **res)
    assert util.compare_sdfg_res(ref, res)
