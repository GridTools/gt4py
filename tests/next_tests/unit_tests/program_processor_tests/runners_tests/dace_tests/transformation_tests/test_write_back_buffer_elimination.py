# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dace
import numpy as np
import pytest

from dace.sdfg import nodes as dace_nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation import pass_pipeline as dace_ppl

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
)

from . import util


def _apply(sdfg: dace.SDFG, assume_pointwise: bool = True) -> bool:
    res = dace_ppl.Pipeline(
        [gtx_transformations.GT4PyWriteBackBufferElimination(assume_pointwise=assume_pointwise)]
    ).apply_pass(sdfg, {})
    return bool(res and res.get("GT4PyWriteBackBufferElimination"))


def _verify(sdfg: dace.SDFG, symbols: dict[str, int] = {}) -> None:
    """Applies the transformation and compares against the unmodified SDFG."""
    ref, res = util.make_sdfg_args(sdfg, symbols)
    util.compile_and_run_sdfg(sdfg, **ref)

    assert _apply(sdfg)
    assert "tmp" not in sdfg.arrays
    sdfg.validate()

    util.compile_and_run_sdfg(sdfg, **res)
    assert util.compare_sdfg_res(ref, res)


def _tmp_producer(state: dace.SDFGState, shape: str = "0:10, 0:10") -> None:
    """A Map that computes `tmp` from `a`."""
    indices = ", ".join(f"__i{dim}" for dim in range(shape.count(":")))
    state.add_mapped_tasklet(
        "producer",
        map_ranges=dict(zip(indices.split(", "), shape.split(", "))),
        inputs={"__in": dace.Memlet(f"a[{indices}]")},
        code="__out = __in + 10.0",
        outputs={"__out": dace.Memlet(f"tmp[{indices}]")},
        external_edges=True,
    )


# `T` is copied into `G` and additionally consumed. How the consumer refers to `T`
#  and how far the copy shifts it are independent, and both influence what has to be
#  rewritten, so they are crossed.
#  - `direct_copy`: the consumer is reached by an edge between two AccessNodes.
#  - `map_scope`: the consumer is inside a Map, so the edge that carries the subset
#    of `T` runs from a MapEntry to an AccessNode and is part of a Memlet tree.
#  - `nested_sdfg`: the consumer is a NestedSDFG, so besides the subset the strides
#    of the descriptor it received from `T` have to follow.
#  The subset of `T` can be the Memlet's own subset or its `other_subset`, depending
#  on which of the two containers the Memlet is associated to. An edge into a
#  NestedSDFG connector always names the outer data, so that axis has one value there.
_CONSUMER_MATRIX = [
    pytest.param(consumer, memlet_on, offset, id=f"{consumer}-memlet_on_{memlet_on}-{offset}")
    for consumer in ("direct_copy", "map_scope", "nested_sdfg")
    for memlet_on in (("tmp",) if consumer == "nested_sdfg" else ("tmp", "other"))
    for offset in ((0, 0), (11, 22))
]


def _mk_consumer_sdfg(consumer: str, memlet_on: str, offset: tuple[int, int]) -> dace.SDFG:
    """`tmp` is written back into `b` at `offset` and additionally consumed."""
    sdfg = dace.SDFG(util.unique_name(f"write_back_{consumer}"))

    for name in ["a", "c"]:
        sdfg.add_array(name, shape=(10, 10), dtype=dace.float64, transient=False)
    sdfg.add_array("b", shape=(100, 100), dtype=dace.float64, transient=False)
    sdfg.add_array("tmp", shape=(10, 10), dtype=dace.float64, transient=True)

    state1: dace.SDFGState = sdfg.add_state(is_start_block=True)
    _tmp_producer(state1)

    state2 = sdfg.add_state_after(state1)
    tmp_node = state2.add_access("tmp")
    off0, off1 = offset
    state2.add_nedge(
        tmp_node,
        state2.add_access("b"),
        dace.Memlet(f"tmp[0:10, 0:10] -> [{off0}:{off0 + 10}, {off1}:{off1 + 10}]"),
    )

    if consumer == "direct_copy":
        sdfg.add_array("d", shape=(10, 10), dtype=dace.float64, transient=True)
        d_node = state2.add_access("d")
        state2.add_nedge(
            tmp_node,
            d_node,
            dace.Memlet("tmp[0:10, 0:10] -> [0:10, 0:10]")
            if memlet_on == "tmp"
            else dace.Memlet(data="d", subset="0:10, 0:10", other_subset="0:10, 0:10"),
        )
        state2.add_mapped_tasklet(
            "consumer",
            map_ranges={"__i1": "0:10", "__i2": "0:10"},
            inputs={"__in": dace.Memlet("d[__i1, __i2]")},
            code="__out = __in * 2.0",
            outputs={"__out": dace.Memlet("c[__i1, __i2]")},
            input_nodes={d_node},
            external_edges=True,
        )

    elif consumer == "map_scope":
        sdfg.add_array(
            "local",
            shape=(10,),
            dtype=dace.float64,
            transient=True,
            storage=dace.dtypes.StorageType.Register,
            lifetime=dace.dtypes.AllocationLifetime.Scope,
        )
        outer_entry, outer_exit = state2.add_map("outer", ndrange={"__i1": "0:10"})
        inner_entry, inner_exit = state2.add_map("inner", ndrange={"__i2": "0:10"})
        tasklet = state2.add_tasklet(
            "comp", inputs={"__in"}, outputs={"__out"}, code="__out = __in * 2.0"
        )
        local_node = state2.add_access("local")

        outer_entry.add_in_connector("IN_tmp")
        outer_entry.add_out_connector("OUT_tmp")
        state2.add_edge(tmp_node, None, outer_entry, "IN_tmp", dace.Memlet("tmp[__i1, 0:10]"))
        state2.add_edge(
            outer_entry,
            "OUT_tmp",
            local_node,
            None,
            dace.Memlet("tmp[__i1, 0:10] -> [0:10]")
            if memlet_on == "tmp"
            else dace.Memlet(data="local", subset="0:10", other_subset="__i1, 0:10"),
        )
        inner_entry.add_in_connector("IN_local")
        inner_entry.add_out_connector("OUT_local")
        state2.add_edge(local_node, None, inner_entry, "IN_local", dace.Memlet("local[0:10]"))
        state2.add_edge(inner_entry, "OUT_local", tasklet, "__in", dace.Memlet("local[__i2]"))
        inner_exit.add_in_connector("IN_c")
        inner_exit.add_out_connector("OUT_c")
        state2.add_edge(tasklet, "__out", inner_exit, "IN_c", dace.Memlet("c[__i1, __i2]"))
        outer_exit.add_in_connector("IN_c")
        outer_exit.add_out_connector("OUT_c")
        state2.add_edge(inner_exit, "OUT_c", outer_exit, "IN_c", dace.Memlet("c[__i1, 0:10]"))
        state2.add_edge(
            outer_exit, "OUT_c", state2.add_access("c"), None, dace.Memlet("c[0:10, 0:10]")
        )

    else:
        assert consumer == "nested_sdfg"
        inner = dace.SDFG(util.unique_name("inner"))
        for name in ["inp", "outp"]:
            inner.add_array(name, shape=(10, 10), dtype=dace.float64, transient=False)
        inner_state = inner.add_state(is_start_block=True)
        inner_state.add_mapped_tasklet(
            "inner_comp",
            map_ranges={"__i1": "0:10", "__i2": "0:10"},
            inputs={"__in": dace.Memlet("inp[__i1, __i2]")},
            code="__out = __in * 2.0",
            outputs={"__out": dace.Memlet("outp[__i1, __i2]")},
            external_edges=True,
        )
        nsdfg = state2.add_nested_sdfg(inner, {"inp"}, {"outp"})
        state2.add_edge(tmp_node, None, nsdfg, "inp", dace.Memlet("tmp[0:10, 0:10]"))
        state2.add_edge(nsdfg, "outp", state2.add_access("c"), None, dace.Memlet("c[0:10, 0:10]"))

    sdfg.validate()
    return sdfg


@pytest.mark.parametrize("consumer, memlet_on, offset", _CONSUMER_MATRIX)
def test_write_back_buffer_elimination_consumer(consumer, memlet_on, offset):
    sdfg = _mk_consumer_sdfg(consumer, memlet_on, offset)
    state1 = sdfg.start_block
    _verify(sdfg)

    # The producer writes `b` where the write back used to put `tmp`.
    (producer_out,) = [
        edge for edge in state1.edges() if isinstance(edge.dst, dace_nodes.AccessNode)
    ]
    assert producer_out.dst.data == "b"
    assert (
        str(producer_out.data.subset)
        == f"{offset[0]}:{offset[0] + 10}, {offset[1]}:{offset[1] + 10}"
    )


def _mk_write_back_buffer_sdfg(
    write_back_subset: str,
) -> tuple[dace.SDFG, dace.SDFGState, dace.SDFGState]:
    """`tmp` is written back into `b` and additionally read by a Map."""
    sdfg = dace.SDFG(util.unique_name("write_back_buffer_sdfg"))

    for name in ["a", "c"]:
        sdfg.add_array(name, shape=(10, 10), dtype=dace.float64, transient=False)
    sdfg.add_array("b", shape=(100, 100), dtype=dace.float64, transient=False)
    sdfg.add_array("tmp", shape=(10, 10), dtype=dace.float64, transient=True)

    state1: dace.SDFGState = sdfg.add_state(is_start_block=True)
    _tmp_producer(state1)

    state2 = sdfg.add_state_after(state1)
    tmp_node = state2.add_access("tmp")
    state2.add_nedge(tmp_node, state2.add_access("b"), dace.Memlet(write_back_subset))
    state2.add_mapped_tasklet(
        "consumer",
        map_ranges={"__i1": "0:10", "__i2": "0:10"},
        inputs={"__in": dace.Memlet("tmp[__i1, __i2]")},
        code="__out = __in * 2.0",
        outputs={"__out": dace.Memlet("c[__i1, __i2]")},
        input_nodes={tmp_node},
        external_edges=True,
    )
    sdfg.validate()

    return sdfg, state1, state2


def test_write_back_buffer_elimination_partial_write_back():
    """A partially copied buffer must be kept.

    Writing `b` directly would touch `b[16:21, 22:32]`, which is outside the range
    that the write back covers.
    """
    sdfg, _, _ = _mk_write_back_buffer_sdfg("tmp[0:5, 0:10] -> [11:16, 22:32]")
    assert not _apply(sdfg)
    assert "tmp" in sdfg.arrays


def test_write_back_buffer_elimination_strided_write_back():
    """A strided write back must be kept.

    `Range.size()` ignores the step, so the destination has the same size as `tmp`,
    but the rewrite can only shift the accesses, not scatter them.
    """
    sdfg, _, _ = _mk_write_back_buffer_sdfg("tmp[0:10, 0:10] -> [0:20:2, 22:32]")
    assert not _apply(sdfg)
    assert "tmp" in sdfg.arrays


def test_write_back_buffer_elimination_wcr_write_back():
    """A write back with conflict resolution must be kept."""
    sdfg, _, state2 = _mk_write_back_buffer_sdfg("tmp[0:10, 0:10] -> [11:21, 22:32]")
    (write_back_edge,) = [
        edge
        for edge in state2.edges()
        if isinstance(edge.src, dace_nodes.AccessNode) and edge.src.data == "tmp"
        if isinstance(edge.dst, dace_nodes.AccessNode)
    ]
    write_back_edge.data.wcr = "lambda a, b: a + b"

    assert not _apply(sdfg)
    assert "tmp" in sdfg.arrays


def test_write_back_buffer_elimination_tmp_used_after_write_back():
    """A consumer of `tmp` downstream of the write back needs no extra condition.

    From the write back onwards `tmp` and `b[11:21, 22:32]` hold the same value, and
    ADR-18 rule 6 forbids a second AccessNode writing `b` downstream of the copy. The
    consumer also reads `b` elsewhere, so the rewritten access has to join that
    AccessNode instead of adding a second one.
    """
    sdfg, _, state2 = _mk_write_back_buffer_sdfg("tmp[0:10, 0:10] -> [11:21, 22:32]")
    sdfg.add_array("e", shape=(10, 10), dtype=dace.float64, transient=False)

    state3 = sdfg.add_state_after(state2)
    state3.add_mapped_tasklet(
        "late_consumer",
        map_ranges={"__i1": "0:10", "__i2": "0:10"},
        inputs={
            "__in": dace.Memlet("tmp[__i1, __i2]"),
            "__in_b": dace.Memlet("b[50 + __i1, __i2]"),
        },
        code="__out = __in * 4.0 + __in_b",
        outputs={"__out": dace.Memlet("e[__i1, __i2]")},
        external_edges=True,
    )
    sdfg.validate()

    _verify(sdfg)

    # ADR-18 rule 8: `b` is only read in `state3`, so a single AccessNode serves both.
    assert len([node for node in state3.data_nodes() if node.data == "b"]) == 1


def test_write_back_buffer_elimination_global_read_after_write_back():
    """`b` may be read once the write back has happened.

    Both in the write back state itself, through the AccessNode that the copy writes,
    and in a state downstream of it, the two versions of the SDFG agree on `b`.
    """
    sdfg, _, state2 = _mk_write_back_buffer_sdfg("tmp[0:10, 0:10] -> [11:21, 22:32]")
    for name in ["e", "f"]:
        sdfg.add_array(name, shape=(10, 10), dtype=dace.float64, transient=False)

    (glob_node,) = [node for node in state2.data_nodes() if node.data == "b"]
    state2.add_mapped_tasklet(
        "read_after_copy",
        map_ranges={"__i1": "0:10", "__i2": "0:10"},
        inputs={"__in": dace.Memlet("b[11 + __i1, 22 + __i2]")},
        code="__out = __in * 5.0",
        outputs={"__out": dace.Memlet("e[__i1, __i2]")},
        input_nodes={glob_node},
        external_edges=True,
    )

    state3 = sdfg.add_state_after(state2)
    state3.add_mapped_tasklet(
        "later_reader",
        map_ranges={"__i1": "0:10", "__i2": "0:10"},
        inputs={"__in": dace.Memlet("b[11 + __i1, 22 + __i2]")},
        code="__out = __in * 7.0",
        outputs={"__out": dace.Memlet("f[__i1, __i2]")},
        external_edges=True,
    )
    sdfg.validate()

    _verify(sdfg)


def test_write_back_buffer_elimination_global_read_before_definition():
    """`b` may be read upstream of the definition of `tmp`, nothing changed there."""
    sdfg, state1, _ = _mk_write_back_buffer_sdfg("tmp[0:10, 0:10] -> [11:21, 22:32]")
    sdfg.add_array("e", shape=(10, 10), dtype=dace.float64, transient=False)

    state0 = sdfg.add_state_before(state1, "before", is_start_block=True)
    state0.add_mapped_tasklet(
        "early_reader",
        map_ranges={"__i1": "0:10", "__i2": "0:10"},
        inputs={"__in": dace.Memlet("b[11 + __i1, 22 + __i2]")},
        code="__out = __in * 5.0",
        outputs={"__out": dace.Memlet("e[__i1, __i2]")},
        external_edges=True,
    )
    sdfg.validate()

    _verify(sdfg)


@pytest.mark.parametrize("disjoint", [True, False])
def test_write_back_buffer_elimination_second_global_write(disjoint):
    """A second write to `b` only matters if it touches the copied range."""
    sdfg, _, state2 = _mk_write_back_buffer_sdfg("tmp[0:10, 0:10] -> [11:21, 22:32]")
    sdfg.add_array("e", shape=(10, 10), dtype=dace.float64, transient=False)

    state3 = sdfg.add_state_after(state2)
    written = "50:60, 0:10" if disjoint else "11:21, 22:32"
    state3.add_mapped_tasklet(
        "second_writer",
        map_ranges={"__i1": "0:10", "__i2": "0:10"},
        inputs={"__in": dace.Memlet("e[__i1, __i2]")},
        code="__out = __in * 5.0",
        outputs={
            "__out": dace.Memlet(
                f"b[{written.split(', ')[0].split(':')[0]} + __i1,"
                f" {written.split(', ')[1].split(':')[0]} + __i2]"
            )
        },
        external_edges=True,
    )
    # The result of the write back is consumed after the second write, so `tmp` may
    #  only become `b` if the second write leaves the copied range alone.
    state4 = sdfg.add_state_after(state3)
    sdfg.add_array("f", shape=(10, 10), dtype=dace.float64, transient=False)
    state4.add_mapped_tasklet(
        "late_consumer",
        map_ranges={"__i1": "0:10", "__i2": "0:10"},
        inputs={"__in": dace.Memlet("tmp[__i1, __i2]")},
        code="__out = __in * 3.0",
        outputs={"__out": dace.Memlet("f[__i1, __i2]")},
        external_edges=True,
    )
    sdfg.validate()

    if disjoint:
        _verify(sdfg)
    else:
        assert not _apply(sdfg)
        assert "tmp" in sdfg.arrays


def _mk_conditional_definition_sdfg(write_back_in_branch: bool) -> dace.SDFG:
    """`tmp` is defined in both branches of a `ConditionalBlock`.

    ADR-18 rule 6 allows that, the two definitions are on different paths. With
    `write_back_in_branch` the write back itself sits in a branch too, and the sibling
    branch writes `b` from another transient.
    """
    sdfg = dace.SDFG(util.unique_name("write_back_conditional"))
    sdfg.add_symbol("flag", dace.int32)

    for name in ["a", "c"]:
        sdfg.add_array(name, shape=(10,), dtype=dace.float64, transient=False)
    sdfg.add_array("b", shape=(10,), dtype=dace.float64, transient=False)
    sdfg.add_array("tmp", shape=(10,), dtype=dace.float64, transient=True)
    sdfg.add_array("other", shape=(10,), dtype=dace.float64, transient=True)

    definitions = ConditionalBlock("definitions")
    sdfg.add_node(definitions, is_start_block=True)
    for index, (condition, factor) in enumerate([("flag == 1", 2.0), (None, 3.0)]):
        body = ControlFlowRegion(f"define_body{index}", sdfg=sdfg)
        definitions.add_branch(condition, body)
        state = body.add_state(f"define{index}", is_start_block=True)
        state.add_mapped_tasklet(
            "producer",
            map_ranges={"__i": "0:10"},
            inputs={"__in": dace.Memlet("a[__i]")},
            code=f"__out = __in * {factor}",
            outputs={"__out": dace.Memlet("tmp[__i]")},
            external_edges=True,
        )

    if write_back_in_branch:
        write_backs = ConditionalBlock("write_backs")
        sdfg.add_node(write_backs)
        sdfg.add_edge(definitions, write_backs, dace.InterstateEdge())

        taken = ControlFlowRegion("taken", sdfg=sdfg)
        write_backs.add_branch("flag == 1", taken)
        wb_state = taken.add_state("write_back", is_start_block=True)

        not_taken = ControlFlowRegion("not_taken", sdfg=sdfg)
        write_backs.add_branch(None, not_taken)
        other_state = not_taken.add_state("other_write_back", is_start_block=True)
        other_state.add_mapped_tasklet(
            "other_producer",
            map_ranges={"__i": "0:10"},
            inputs={"__in": dace.Memlet("a[__i]")},
            code="__out = __in * 9.0",
            outputs={"__out": dace.Memlet("other[__i]")},
            external_edges=True,
        )
        (other_node,) = [node for node in other_state.data_nodes() if node.data == "other"]
        other_state.add_nedge(
            other_node, other_state.add_access("b"), dace.Memlet("other[0:10] -> [0:10]")
        )
        consumer_state = sdfg.add_state("consume")
        sdfg.add_edge(write_backs, consumer_state, dace.InterstateEdge())
    else:
        wb_state = sdfg.add_state("write_back")
        sdfg.add_edge(definitions, wb_state, dace.InterstateEdge())
        consumer_state = wb_state

    wb_state.add_nedge(
        wb_state.add_access("tmp"), wb_state.add_access("b"), dace.Memlet("tmp[0:10] -> [0:10]")
    )
    consumer_state.add_mapped_tasklet(
        "consumer",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("tmp[__i]")},
        code="__out = __in * 5.0",
        outputs={"__out": dace.Memlet("c[__i]")},
        external_edges=True,
    )
    sdfg.validate()
    return sdfg


@pytest.mark.parametrize("flag", [0, 1])
def test_write_back_buffer_elimination_conditional_definition(flag):
    """Two AccessNodes writing `tmp` on different paths are allowed."""
    sdfg = _mk_conditional_definition_sdfg(write_back_in_branch=False)
    _verify(sdfg, symbols={"flag": flag})


def test_write_back_buffer_elimination_conditional_write_back():
    """A sibling branch that writes `b` itself must keep the write back.

    Preponing the write into `b` would put a write of `b` on the path that writes it
    from `other` and clobber that.
    """
    sdfg = _mk_conditional_definition_sdfg(write_back_in_branch=True)
    assert not _apply(sdfg)
    assert "tmp" in sdfg.arrays


def test_write_back_buffer_elimination_loop():
    """A write back inside a `LoopRegion` must be kept.

    ADR-18 rule 5 forbids cycles in the state graph but mandates `LoopRegion` for
    loops, so a state of the loop body reaches itself. The body defines `tmp`, then
    reads `b` and only then writes `tmp` back. The read is reachable from the write
    back, through the next iteration, but it must still see what the previous
    iteration wrote and not what the producer of `tmp` just computed.
    """
    sdfg = dace.SDFG(util.unique_name("write_back_loop"))

    for name in ["a", "b", "d"]:
        sdfg.add_array(name, shape=(20,), dtype=dace.float64, transient=False)
    sdfg.add_array("tmp", shape=(20,), dtype=dace.float64, transient=True)

    loop = LoopRegion(
        label="loop",
        condition_expr="it < 2",
        loop_var="it",
        initialize_expr="it = 0",
        update_expr="it = it + 1",
    )
    sdfg.add_node(loop, is_start_block=True)

    define_state = loop.add_state("define", is_start_block=True)
    define_state.add_mapped_tasklet(
        "producer",
        map_ranges={"__i": "0:20"},
        inputs={"__in": dace.Memlet("a[__i]")},
        code="__out = __in + it",
        outputs={"__out": dace.Memlet("tmp[__i]")},
        external_edges=True,
    )
    read_state = loop.add_state_after(define_state, "read")
    read_state.add_mapped_tasklet(
        "reader",
        map_ranges={"__i": "0:20"},
        inputs={"__in": dace.Memlet("b[__i]")},
        code="__out = __in * 2.0",
        outputs={"__out": dace.Memlet("d[__i]")},
        external_edges=True,
    )
    write_back_state = loop.add_state_after(read_state, "write_back")
    write_back_state.add_nedge(
        write_back_state.add_access("tmp"),
        write_back_state.add_access("b"),
        dace.Memlet("tmp[0:20] -> [0:20]"),
    )
    sdfg.validate()

    assert not _apply(sdfg)
    assert "tmp" in sdfg.arrays


def test_write_back_buffer_elimination_empty_memlet():
    """An empty Memlet next to `tmp` must keep imposing the same order.

    It has to stay empty, turning it into a copy of the full array would both move
    data that nobody asked for and, because the edge has no connectors, make the SDFG
    invalid. And it has to end up on the AccessNode of `b` that the producer writes,
    on a second one it would order the consumer after nothing.
    """
    sdfg = dace.SDFG(util.unique_name("write_back_empty_memlet"))

    for name in ["a", "c"]:
        sdfg.add_array(name, shape=(10,), dtype=dace.float64, transient=False)
    sdfg.add_array("b", shape=(20,), dtype=dace.float64, transient=False)
    sdfg.add_array("tmp", shape=(10,), dtype=dace.float64, transient=True)

    state1: dace.SDFGState = sdfg.add_state(is_start_block=True)
    tmp_node = state1.add_access("tmp")
    state1.add_mapped_tasklet(
        "producer",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("a[__i]")},
        code="__out = __in + 10.0",
        outputs={"__out": dace.Memlet("tmp[__i]")},
        output_nodes={tmp_node},
        external_edges=True,
    )
    # A Map that does not consume `tmp`, but that must run after it was computed.
    map_entry, map_exit = state1.add_map("unrelated", ndrange={"__i": "0:10"})
    tasklet = state1.add_tasklet(
        "comp", inputs={"__in"}, outputs={"__out"}, code="__out = __in * 3.0"
    )
    map_entry.add_in_connector("IN_a")
    map_entry.add_out_connector("OUT_a")
    state1.add_edge(state1.add_access("a"), None, map_entry, "IN_a", dace.Memlet("a[0:10]"))
    state1.add_edge(map_entry, "OUT_a", tasklet, "__in", dace.Memlet("a[__i]"))
    map_exit.add_in_connector("IN_c")
    map_exit.add_out_connector("OUT_c")
    state1.add_edge(tasklet, "__out", map_exit, "IN_c", dace.Memlet("c[__i]"))
    state1.add_edge(map_exit, "OUT_c", state1.add_access("c"), None, dace.Memlet("c[0:10]"))
    state1.add_nedge(tmp_node, map_entry, dace.Memlet())

    state2 = sdfg.add_state_after(state1)
    state2.add_nedge(
        state2.add_access("tmp"), state2.add_access("b"), dace.Memlet("tmp[0:10] -> [10:20]")
    )
    sdfg.validate()

    _verify(sdfg)

    (sequencing_edge,) = [edge for edge in state1.in_edges(map_entry) if edge.dst_conn is None]
    assert sequencing_edge.data.is_empty()
    # The order "after the producer" is only imposed if the empty Memlet starts at the
    #  AccessNode that the producer writes.
    (producer_out,) = [
        edge
        for edge in state1.edges()
        if isinstance(edge.src, dace_nodes.MapExit)
        if isinstance(edge.dst, dace_nodes.AccessNode) and edge.dst.data == "b"
    ]
    assert sequencing_edge.src is producer_out.dst


@pytest.mark.parametrize("read_reaches_producer", ["directly", "through_map"])
@pytest.mark.parametrize("assume_pointwise", [True, False])
def test_write_back_buffer_elimination_pointwise_reader(read_reaches_producer, assume_pointwise):
    """`b` is read elementwise by the producer of `tmp`, i.e. ADR-18 rule 3.

    The read either enters the Map that writes `tmp` or a Map upstream of it; before
    Map fusion the lowering emits the latter.
    """
    sdfg = dace.SDFG(util.unique_name("write_back_pointwise_reader"))

    for name in ["a", "c", "b"]:
        sdfg.add_array(name, shape=(20,), dtype=dace.float64, transient=False)
    sdfg.add_array("tmp", shape=(20,), dtype=dace.float64, transient=True)

    state1: dace.SDFGState = sdfg.add_state(is_start_block=True)
    if read_reaches_producer == "directly":
        state1.add_mapped_tasklet(
            "producer",
            map_ranges={"__i": "0:20"},
            inputs={"__in_a": dace.Memlet("a[__i]"), "__in_b": dace.Memlet("b[__i]")},
            code="__out = __in_a + __in_b",
            outputs={"__out": dace.Memlet("tmp[__i]")},
            external_edges=True,
        )
    else:
        sdfg.add_array("tt", shape=(20,), dtype=dace.float64, transient=True)
        tt_node = state1.add_access("tt")
        state1.add_mapped_tasklet(
            "pre_producer",
            map_ranges={"__i": "0:20"},
            inputs={"__in": dace.Memlet("b[__i]")},
            code="__out = __in * 0.5",
            outputs={"__out": dace.Memlet("tt[__i]")},
            output_nodes={tt_node},
            external_edges=True,
        )
        state1.add_mapped_tasklet(
            "producer",
            map_ranges={"__i": "0:20"},
            inputs={"__in_a": dace.Memlet("a[__i]"), "__in_tt": dace.Memlet("tt[__i]")},
            code="__out = __in_a + __in_tt",
            outputs={"__out": dace.Memlet("tmp[__i]")},
            input_nodes={tt_node},
            external_edges=True,
        )

    state2: dace.SDFGState = sdfg.add_state_after(state1)
    tmp_node = state2.add_access("tmp")
    state2.add_nedge(tmp_node, state2.add_access("b"), dace.Memlet("tmp[0:20] -> [0:20]"))
    state2.add_mapped_tasklet(
        "consumer",
        map_ranges={"__i": "0:20"},
        inputs={"__in": dace.Memlet("tmp[__i]")},
        code="__out = __in * 2.0",
        outputs={"__out": dace.Memlet("c[__i]")},
        input_nodes={tmp_node},
        external_edges=True,
    )
    sdfg.validate()

    if not assume_pointwise:
        assert not _apply(sdfg, assume_pointwise=False)
        assert "tmp" in sdfg.arrays
        return

    _verify(sdfg)
    # ADR-18 rule 8 allows a single AccessNode per data and state, rule 3 makes the
    #  exception for a global that is input and output; reusing the one that is read
    #  would make the state cyclic.
    assert len([node for node in state1.data_nodes() if node.data == "b"]) == 2


def test_write_back_buffer_elimination_pointwise_reader_shifted():
    """A shifted write back must be kept when the producer reads `b`.

    ADR-18 rule 3 is about using the very same memory as input and output. The
    rewritten producer would read `b[__i]` and write `b[5 + __i]`, so the iterations
    would clobber each other's input.
    """
    sdfg = dace.SDFG(util.unique_name("write_back_pointwise_shifted"))

    for name in ["a", "c"]:
        sdfg.add_array(name, shape=(20,), dtype=dace.float64, transient=False)
    sdfg.add_array("b", shape=(25,), dtype=dace.float64, transient=False)
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
    tmp_node = state2.add_access("tmp")
    state2.add_nedge(tmp_node, state2.add_access("b"), dace.Memlet("tmp[0:20] -> [5:25]"))
    state2.add_mapped_tasklet(
        "consumer",
        map_ranges={"__i": "0:20"},
        inputs={"__in": dace.Memlet("tmp[__i]")},
        code="__out = __in * 2.0",
        outputs={"__out": dace.Memlet("c[__i]")},
        input_nodes={tmp_node},
        external_edges=True,
    )
    sdfg.validate()

    assert not _apply(sdfg)
    assert "tmp" in sdfg.arrays


@pytest.mark.parametrize("shares_access_node", [True, False])
def test_write_back_buffer_elimination_unrelated_reader(shares_access_node):
    """`assume_pointwise` must not waive a read of `b` by an unrelated Map.

    The unrelated Map is unordered with respect to the producer of `tmp`, so writing
    `b` directly would race with it. That holds whether it reads `b` through its own
    AccessNode or through the one that also feeds the producer.
    """
    sdfg = dace.SDFG(util.unique_name("write_back_unrelated_reader"))

    for name in ["a", "d", "b"]:
        sdfg.add_array(name, shape=(20,), dtype=dace.float64, transient=False)
    sdfg.add_array("tmp", shape=(20,), dtype=dace.float64, transient=True)

    state: dace.SDFGState = sdfg.add_state(is_start_block=True)
    producer_inputs = {"__in": dace.Memlet("a[__i]")}
    code = "__out = __in + 10.0"
    if shares_access_node:
        producer_inputs["__in_b"] = dace.Memlet("b[__i]")
        code = "__out = __in + __in_b"
    state.add_mapped_tasklet(
        "producer",
        map_ranges={"__i": "0:20"},
        inputs=producer_inputs,
        code=code,
        outputs={"__out": dace.Memlet("tmp[__i]")},
        external_edges=True,
    )
    glob_nodes = {node for node in state.data_nodes() if node.data == "b"}
    state.add_mapped_tasklet(
        "unrelated_reader",
        map_ranges={"__i": "0:20"},
        inputs={"__in": dace.Memlet("b[19 - __i]")},
        code="__out = __in * 5.0",
        outputs={"__out": dace.Memlet("d[__i]")},
        input_nodes=glob_nodes if shares_access_node else set(),
        external_edges=True,
    )

    state2: dace.SDFGState = sdfg.add_state_after(state)
    state2.add_nedge(
        state2.add_access("tmp"), state2.add_access("b"), dace.Memlet("tmp[0:20] -> [0:20]")
    )
    sdfg.validate()

    assert not _apply(sdfg)
    assert "tmp" in sdfg.arrays


@pytest.mark.parametrize("viewed_data", ["tmp", "unrelated", "in_map"])
def test_write_back_buffer_elimination_view(viewed_data):
    """Any View makes the transformation give up.

    A View has its own descriptor, whose strides are the ones of the contiguous `tmp`
    and not the ones of `b`, and nothing updates them. GT4Py does not generate Views,
    so no attempt is made to tell a harmless one from a harmful one.
    """
    sdfg = dace.SDFG(util.unique_name("write_back_view"))

    for name in ["a", "c"]:
        sdfg.add_array(name, shape=(10, 10), dtype=dace.float64, transient=False)
    sdfg.add_array("b", shape=(100, 100), dtype=dace.float64, transient=False)
    sdfg.add_array("tmp", shape=(10, 10), dtype=dace.float64, transient=True)

    state1: dace.SDFGState = sdfg.add_state(is_start_block=True)
    _tmp_producer(state1)

    state2 = sdfg.add_state_after(state1)
    state2.add_nedge(
        state2.add_access("tmp"),
        state2.add_access("b"),
        dace.Memlet("tmp[0:10, 0:10] -> [11:21, 22:32]"),
    )

    if viewed_data == "in_map":
        # The View refers to `tmp` through the MapEntry, i.e. there is no AccessNode
        #  of `tmp` next to it.
        sdfg.add_view(
            "v", shape=(10,), dtype=dace.float64, strides=(sdfg.arrays["tmp"].strides[0],)
        )
        tmp_node = state2.add_access("tmp")
        map_entry, map_exit = state2.add_map("consumer", ndrange={"__i2": "0:10"})
        view_node = state2.add_access("v")
        map_entry.add_in_connector("IN_tmp")
        map_entry.add_out_connector("OUT_tmp")
        state2.add_edge(tmp_node, None, map_entry, "IN_tmp", dace.Memlet("tmp[0:10, __i2]"))
        state2.add_edge(map_entry, "OUT_tmp", view_node, "views", dace.Memlet("tmp[0:10, __i2]"))
        tasklet = state2.add_tasklet(
            "reduce",
            inputs={"__in"},
            outputs={"__out"},
            code="__out = 0.0\nfor __k in range(10):\n    __out = __out + __in[__k]",
            language=dace.dtypes.Language.Python,
        )
        state2.add_edge(view_node, None, tasklet, "__in", dace.Memlet("v[0:10]"))
        map_exit.add_in_connector("IN_c")
        map_exit.add_out_connector("OUT_c")
        state2.add_edge(tasklet, "__out", map_exit, "IN_c", dace.Memlet("c[0, __i2]"))
        state2.add_edge(map_exit, "OUT_c", state2.add_access("c"), None, dace.Memlet("c[0, 0:10]"))
    else:
        source = "tmp" if viewed_data == "tmp" else "a"
        sdfg.add_view("v", shape=(10, 10), dtype=dace.float64)
        view_node = state2.add_access("v")
        state2.add_edge(
            state2.add_access(source),
            None,
            view_node,
            "views",
            dace.Memlet(f"{source}[0:10, 0:10]"),
        )
        state2.add_mapped_tasklet(
            "consumer",
            map_ranges={"__i1": "0:10", "__i2": "0:10"},
            inputs={"__in": dace.Memlet("v[__i1, __i2]")},
            code="__out = __in * 2.0",
            outputs={"__out": dace.Memlet("c[__i1, __i2]")},
            input_nodes={view_node},
            external_edges=True,
        )
    sdfg.validate()

    assert not _apply(sdfg)
    assert "tmp" in sdfg.arrays
