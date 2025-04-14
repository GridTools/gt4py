# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import copy
from enum import Enum
from typing import Callable

import numpy as np
import pytest


dace = pytest.importorskip("dace")
from dace.sdfg import nodes as dace_nodes, propagation as dace_propagation

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
)


from . import util


def _get_simple_sdfg() -> tuple[dace.SDFG, Callable[[np.ndarray, np.ndarray], np.ndarray]]:
    """Creates a simple SDFG.

    The k blocking transformation can be applied to the SDFG, however no node
    can be taken out. This is because how it is constructed. However, applying
    some simplistic transformations will enable the transformation.
    """
    sdfg = dace.SDFG(util.unique_name("simple_block_sdfg"))
    state = sdfg.add_state("state", is_start_block=True)
    sdfg.add_symbol("N", dace.int32)
    sdfg.add_symbol("M", dace.int32)
    _, a = sdfg.add_array("a", ("N", "M"), dace.float64, transient=False)
    _, b = sdfg.add_array("b", ("N",), dace.float64, transient=False)
    _, c = sdfg.add_array("c", ("N", "M"), dace.float64, transient=False)
    state.add_mapped_tasklet(
        name="comp",
        map_ranges=dict(i="0:N", j="0:M"),
        inputs=dict(__in0=dace.Memlet("a[i, j]"), __in1=dace.Memlet("b[i]")),
        outputs=dict(__out=dace.Memlet("c[i, j]")),
        code="__out = __in0 + __in1",
        external_edges=True,
    )
    return sdfg, lambda a, b: a + b.reshape((-1, 1))


def _get_chained_sdfg() -> tuple[dace.SDFG, Callable[[np.ndarray, np.ndarray], np.ndarray]]:
    """Generates an SDFG that has chained Tasklets that are independent.

    The bottom Tasklet is the only dependent Tasklet.
    """
    sdfg = dace.SDFG(util.unique_name("chained_block_sdfg"))
    state = sdfg.add_state("state", is_start_block=True)
    sdfg.add_symbol("N", dace.int32)
    sdfg.add_symbol("M", dace.int32)
    sdfg.add_array("a", ("N", "M"), dace.float64, transient=False)
    sdfg.add_array("b", ("N",), dace.float64, transient=False)
    sdfg.add_array("c", ("N", "M"), dace.float64, transient=False)
    sdfg.add_scalar("tmp1", dtype=dace.float64, transient=True)
    sdfg.add_scalar("tmp2", dtype=dace.float64, transient=True)
    a, b, c, tmp1, tmp2 = (state.add_access(name) for name in ["a", "b", "c", "tmp1", "tmp2"])

    # First independent Tasklet.
    task1 = state.add_tasklet(
        "task1_indepenent",
        inputs={
            "__in0",  # <- `b[i]`
        },
        outputs={
            "__out0",  # <- `tmp1`
        },
        code="__out0 = __in0 + 3.0",
    )

    # This is the second independent Tasklet.
    task2 = state.add_tasklet(
        "task2_indepenent",
        inputs={
            "__in0",  # <- `tmp1`
            "__in1",  # <- `b[i]`
        },
        outputs={
            "__out0",  # <- `tmp2`
        },
        code="__out0 = __in0 + __in1",
    )

    # This is the third Tasklet, which is dependent.
    task3 = state.add_tasklet(
        "task3_dependent",
        inputs={
            "__in0",  # <- `tmp2`
            "__in1",  # <- `a[i, j]`
        },
        outputs={
            "__out0",  # <- `c[i, j]`.
        },
        code="__out0 = __in0 + __in1",
    )

    # Now create the map
    mentry, mexit = state.add_map(
        "map",
        ndrange={"i": "0:N", "j": "0:M"},
    )

    # Now assemble everything.
    state.add_edge(mentry, "OUT_b", task1, "__in0", dace.Memlet("b[i]"))
    state.add_edge(task1, "__out0", tmp1, None, dace.Memlet("tmp1[0]"))

    state.add_edge(tmp1, None, task2, "__in0", dace.Memlet("tmp1[0]"))
    state.add_edge(mentry, "OUT_b", task2, "__in1", dace.Memlet("b[i]"))
    state.add_edge(task2, "__out0", tmp2, None, dace.Memlet("tmp2[0]"))

    state.add_edge(tmp2, None, task3, "__in0", dace.Memlet("tmp2[0]"))
    state.add_edge(mentry, "OUT_a", task3, "__in1", dace.Memlet("a[i, j]"))
    state.add_edge(task3, "__out0", mexit, "IN_c", dace.Memlet("c[i, j]"))

    state.add_edge(a, None, mentry, "IN_a", sdfg.make_array_memlet("a"))
    state.add_edge(b, None, mentry, "IN_b", sdfg.make_array_memlet("b"))
    state.add_edge(mexit, "OUT_c", c, None, sdfg.make_array_memlet("c"))
    for name in ["a", "b"]:
        mentry.add_in_connector("IN_" + name)
        mentry.add_out_connector("OUT_" + name)
    mexit.add_in_connector("IN_c")
    mexit.add_out_connector("OUT_c")

    dace_propagation.propagate_states(sdfg)
    sdfg.validate()

    return sdfg, lambda a, b: (a + (2 * b.reshape((-1, 1)) + 3))


def _get_sdfg_with_empty_memlet(
    first_tasklet_independent: bool,
    only_empty_memlets: bool,
) -> tuple[
    dace.SDFG, dace_nodes.MapEntry, dace_nodes.Tasklet, dace_nodes.AccessNode, dace_nodes.Tasklet
]:
    """Generates an SDFG with an empty tasklet.

    The map contains two (serial) tasklets, connected through an access node.
    The first tasklet has an empty memlet that connects it to the map entry.
    Depending on `first_tasklet_independent` the tasklet is either independent
    or not. The second tasklet has an additional in connector that accesses an array.

    If `only_empty_memlets` is given then the second memlet will only depend
    on the input of the first tasklet. However, since it is connected to the
    map exit, it will be classified as dependent.

    Returns:
        The function returns the SDFG, the map entry and the first tasklet (that
        is either dependent or independent), the access node between the tasklets
        and the second tasklet that is always dependent.
    """
    sdfg = dace.SDFG(util.unique_name("empty_memlet_sdfg"))
    state = sdfg.add_state("state", is_start_block=True)
    sdfg.add_symbol("N", dace.int32)
    sdfg.add_symbol("M", dace.int32)
    sdfg.add_array("b", ("N", "M"), dace.float64, transient=False)
    b = state.add_access("b")
    sdfg.add_scalar("tmp", dtype=dace.float64, transient=True)
    tmp = state.add_access("tmp")

    if not only_empty_memlets:
        sdfg.add_array("a", ("N", "M"), dace.float64, transient=False)
        a = state.add_access("a")

    # This is the first tasklet.
    task1 = state.add_tasklet(
        "task1",
        inputs={},
        outputs={"__out0"},
        code="__out0 = 1.0" if first_tasklet_independent else "__out0 = j",
    )

    if only_empty_memlets:
        task2 = state.add_tasklet(
            "task2", inputs={"__in0"}, outputs={"__out0"}, code="__out0 = __in0 + 1.0"
        )
    else:
        task2 = state.add_tasklet(
            "task2", inputs={"__in0", "__in1"}, outputs={"__out0"}, code="__out0 = __in0 + __in1"
        )

    # Now create the map
    mentry, mexit = state.add_map("map", ndrange={"i": "0:N", "j": "0:M"})

    if not only_empty_memlets:
        state.add_edge(a, None, mentry, "IN_a", dace.Memlet("a[0:N, 0:M]"))
        state.add_edge(mentry, "OUT_a", task2, "__in1", dace.Memlet("a[i, j]"))

    state.add_edge(task2, "__out0", mexit, "IN_b", dace.Memlet("b[i, j]"))
    state.add_edge(mexit, "OUT_b", b, None, dace.Memlet("b[0:N, 0:M]"))

    state.add_edge(mentry, None, task1, None, dace.Memlet())
    state.add_edge(task1, "__out0", tmp, None, dace.Memlet("tmp[0]"))
    state.add_edge(tmp, None, task2, "__in0", dace.Memlet("tmp[0]"))

    if not only_empty_memlets:
        mentry.add_in_connector("IN_a")
        mentry.add_out_connector("OUT_a")
    mexit.add_in_connector("IN_b")
    mexit.add_out_connector("OUT_b")

    sdfg.validate()

    return sdfg, mentry, task1, tmp, task2


def test_only_dependent():
    """Just applying the transformation to the SDFG.

    Because all of nodes (which is only a Tasklet) inside the map scope are
    "dependent", see the transformation for explanation of terminology, the
    transformation will only add an inner map.
    """
    sdfg, reff = _get_simple_sdfg()

    N, M = 100, 10
    a = np.random.rand(N, M)
    b = np.random.rand(N)
    c = np.zeros_like(a)
    ref = reff(a, b)

    # Apply the transformation
    count = sdfg.apply_transformations_repeated(
        gtx_transformations.LoopBlocking(blocking_size=10, blocking_parameter="j"),
        validate=True,
        validate_all=True,
    )
    assert count == 1

    assert len(sdfg.states()) == 1
    state = sdfg.states()[0]
    source_nodes = state.source_nodes()
    assert len(source_nodes) == 2
    assert all(isinstance(x, dace_nodes.AccessNode) for x in source_nodes)
    source_node = source_nodes[0]  # Unspecific which one it is, but it does not matter.
    assert state.out_degree(source_node) == 1
    outer_map: dace_nodes.MapEntry = next(iter(state.out_edges(source_node))).dst
    assert isinstance(outer_map, dace_nodes.MapEntry)
    assert state.in_degree(outer_map) == 2
    assert state.out_degree(outer_map) == 2
    assert len(outer_map.map.params) == 2
    assert "j" not in outer_map.map.params
    assert all(isinstance(x.dst, dace_nodes.MapEntry) for x in state.out_edges(outer_map))
    inner_map: dace_nodes.MapEntry = next(iter(state.out_edges(outer_map))).dst
    assert len(inner_map.map.params) == 1
    assert inner_map.map.params[0] == "j"
    assert inner_map.map.schedule == dace.dtypes.ScheduleType.Sequential

    sdfg(a=a, b=b, c=c, N=N, M=M)
    assert np.allclose(ref, c)


def test_intermediate_access_node():
    """Test the lifting out, version "AccessNode".

    The Tasklet of the SDFG generated by `_get_simple_sdfg()` has to be inside the
    inner most loop because one of its input Memlet depends on `j`. However,
    one of its input, `b[i]` does not. Instead of connecting `b` directly with the
    Tasklet, this test will store `b[i]` inside a temporary inside the Map.
    This access node is independent of `j` and can thus be moved out of the inner
    most scope.
    """
    sdfg, reff = _get_simple_sdfg()

    N, M = 100, 10
    a = np.random.rand(N, M)
    b = np.random.rand(N)
    c = np.zeros_like(a)
    ref = reff(a, b)

    # Now make a small modification is such that the transformation does something.
    state = sdfg.states()[0]
    sdfg.add_scalar("tmp", dace.float64, transient=True)

    tmp = state.add_access("tmp")
    edge = next(
        e for e in state.edges() if isinstance(e.src, dace_nodes.MapEntry) and e.data.data == "b"
    )
    state.add_edge(edge.src, edge.src_conn, tmp, None, copy.deepcopy(edge.data))
    state.add_edge(tmp, None, edge.dst, edge.dst_conn, dace.Memlet("tmp[0]"))
    state.remove_edge(edge)

    # Test if after the modification the SDFG still works
    sdfg(a=a, b=b, c=c, N=N, M=M)
    assert np.allclose(ref, c)

    # Apply the transformation.
    count = sdfg.apply_transformations_repeated(
        gtx_transformations.LoopBlocking(blocking_size=10, blocking_parameter="j"),
        validate=True,
        validate_all=True,
    )
    assert count == 1

    # Inspect if the SDFG was modified correctly.
    #  We only inspect `tmp` which now has to be between the two maps.
    assert state.in_degree(tmp) == 1
    assert state.out_degree(tmp) == 1
    top_node = next(iter(state.in_edges(tmp))).src
    bottom_node = next(iter(state.out_edges(tmp))).dst
    assert isinstance(top_node, dace_nodes.MapEntry)
    assert isinstance(bottom_node, dace_nodes.MapEntry)
    assert bottom_node is not top_node

    c[:] = 0
    sdfg(a=a, b=b, c=c, N=N, M=M)
    assert np.allclose(ref, c)


def test_chained_access() -> None:
    """Test if chained access works."""
    sdfg, reff = _get_chained_sdfg()
    state: dace.SDFGState = next(iter(sdfg.states()))

    N, M = 100, 10
    a = np.random.rand(N, M)
    b = np.random.rand(N)
    c = np.zeros_like(a)
    ref = reff(a, b)

    # Before the optimization
    sdfg(a=a, b=b, c=c, M=M, N=N)
    assert np.allclose(c, ref)
    c[:] = 0

    # Apply the transformation.
    count = sdfg.apply_transformations_repeated(
        gtx_transformations.LoopBlocking(blocking_size=10, blocking_parameter="j"),
        validate=True,
        validate_all=True,
    )
    assert count == 1

    # Now run the SDFG to see if it is still the same
    sdfg(a=a, b=b, c=c, M=M, N=N)
    assert np.allclose(c, ref)

    # Now look for the outer map.
    outer_map = None
    for node in state.nodes():
        if not isinstance(node, dace_nodes.MapEntry):
            continue
        if state.scope_dict()[node] is None:
            assert (
                outer_map is None
            ), f"Found multiple outer maps, first '{outer_map}', second '{node}'."
            outer_map = node
    assert outer_map is not None, "Could not found the outer map."
    assert len(outer_map.map.params) == 2

    # Now inspect the SDFG if the transformation was applied correctly.
    first_level_tasklets: list[dace_nodes.Tasklet] = []
    inner_map: list[dace_nodes.MapEntry] = []

    for edge in state.out_edges(outer_map):
        node: dace_nodes.Node = edge.dst
        if isinstance(node, dace_nodes.Tasklet):
            first_level_tasklets.append(node)
        elif isinstance(node, dace_nodes.MapEntry):
            inner_map.append(node)
        else:
            assert False, f"Found unexpected node '{type(node).__name__}'."

    # Test what we found
    assert len(first_level_tasklets) == 2
    assert len(set(first_level_tasklets)) == 2
    assert len(inner_map) == 1
    assert state.scope_dict()[inner_map[0]] is outer_map

    # Now we look inside the inner map
    #  There we expect to find one Tasklet.
    inner_scope = state.scope_subgraph(next(iter(inner_map)), False, False)
    assert inner_scope.number_of_nodes() == 1
    inner_tasklet = next(iter(inner_scope.nodes()))

    assert isinstance(inner_tasklet, dace_nodes.Tasklet)
    assert inner_tasklet not in first_level_tasklets


def test_direct_map_exit_connection() -> dace.SDFG:
    """Generates a SDFG with a mapped independent tasklet connected to the map exit.

    Because the tasklet is connected to the map exit it can not be independent.
    """
    sdfg = dace.SDFG(util.unique_name("mapped_tasklet_sdfg"))
    state = sdfg.add_state("state", is_start_block=True)
    sdfg.add_array("a", (10,), dace.float64, transient=False)
    sdfg.add_array("b", (10, 30), dace.float64, transient=False)
    tsklt, me, mx = state.add_mapped_tasklet(
        name="comp",
        map_ranges=dict(i=f"0:10", j=f"0:30"),
        inputs=dict(__in0=dace.Memlet("a[i]")),
        outputs=dict(__out=dace.Memlet("b[i, j]")),
        code="__out = __in0 + 1",
        external_edges=True,
    )

    assert all(out_edge.dst is tsklt for out_edge in state.out_edges(me))
    assert all(in_edge.src is tsklt for in_edge in state.in_edges(mx))

    count = sdfg.apply_transformations_repeated(
        gtx_transformations.LoopBlocking(blocking_size=5, blocking_parameter="j"),
        validate=True,
        validate_all=True,
    )
    assert count == 1

    assert all(isinstance(out_edge.dst, dace_nodes.MapEntry) for out_edge in state.out_edges(me))
    assert all(isinstance(in_edge.src, dace_nodes.MapExit) for in_edge in state.in_edges(mx))


def test_empty_memlet_1():
    sdfg, mentry, itask, tmp, task2 = _get_sdfg_with_empty_memlet(
        first_tasklet_independent=True,
        only_empty_memlets=False,
    )
    state: dace.SDFGState = next(iter(sdfg.nodes()))

    count = sdfg.apply_transformations_repeated(
        gtx_transformations.LoopBlocking(blocking_size=5, blocking_parameter="j"),
        validate=True,
        validate_all=True,
    )
    assert count == 1

    scope_dict = state.scope_dict()
    assert scope_dict[mentry] is None
    assert scope_dict[itask] is mentry
    assert scope_dict[tmp] is mentry
    assert scope_dict[task2] is not mentry
    assert scope_dict[task2] is not None
    assert all(
        isinstance(in_edge.src, dace_nodes.MapEntry) and in_edge.src is not mentry
        for in_edge in state.in_edges(task2)
    )


def test_empty_memlet_2():
    sdfg, mentry, dtask, tmp, task2 = _get_sdfg_with_empty_memlet(
        first_tasklet_independent=False,
        only_empty_memlets=False,
    )
    state: dace.SDFGState = next(iter(sdfg.nodes()))

    count = sdfg.apply_transformations_repeated(
        gtx_transformations.LoopBlocking(blocking_size=5, blocking_parameter="j"),
        validate=True,
        validate_all=True,
    )
    assert count == 1

    # Find the inner map entry
    assert all(
        isinstance(out_edge.dst, dace_nodes.MapEntry) for out_edge in state.out_edges(mentry)
    )
    inner_mentry = next(iter(state.out_edges(mentry))).dst

    scope_dict = state.scope_dict()
    assert scope_dict[mentry] is None
    assert scope_dict[inner_mentry] is mentry
    assert scope_dict[dtask] is inner_mentry
    assert scope_dict[tmp] is inner_mentry
    assert scope_dict[task2] is inner_mentry


def test_empty_memlet_3():
    # This is the only interesting case with only empty memlet.
    sdfg, mentry, dtask, tmp, task2 = _get_sdfg_with_empty_memlet(
        first_tasklet_independent=False,
        only_empty_memlets=True,
    )
    state: dace.SDFGState = next(iter(sdfg.nodes()))

    count = sdfg.apply_transformations_repeated(
        gtx_transformations.LoopBlocking(blocking_size=5, blocking_parameter="j"),
        validate=True,
        validate_all=True,
    )
    assert count == 1

    # The top map only has a single output, which is the empty edge, that is holding
    #  the inner map entry in the scope.
    assert all(out_edge.data.is_empty() for out_edge in state.out_edges(mentry))
    assert state.in_degree(mentry) == 0
    assert state.out_degree(mentry) == 1
    assert all(
        isinstance(out_edge.dst, dace_nodes.MapEntry) for out_edge in state.out_edges(mentry)
    )

    inner_mentry = next(iter(state.out_edges(mentry))).dst

    scope_dict = state.scope_dict()
    assert scope_dict[mentry] is None
    assert scope_dict[inner_mentry] is mentry
    assert scope_dict[dtask] is inner_mentry
    assert scope_dict[tmp] is inner_mentry
    assert scope_dict[task2] is inner_mentry


class IndependentPart(Enum):
    NONE = 0
    TASKLET = 1
    NESTED_SDFG = 2


def _make_loop_blocking_sdfg_with_inner_map(
    add_independent_part: IndependentPart,
) -> tuple[dace.SDFG, dace.SDFGState, dace_nodes.MapEntry, dace_nodes.MapEntry]:
    """
    Generate the SDFGs with an inner map.

    The SDFG has an inner map that is classified as dependent. If
    `add_independent_part` is `True` then the SDFG has a part that is independent.
    Note that everything is read from a single connector.

    Return:
        The function will return the SDFG, the state and the map entry for the outer
        and inner map.
    """
    sdfg = dace.SDFG(util.unique_name("sdfg_with_inner_map"))
    state = sdfg.add_state(is_start_block=True)

    for name in "AB":
        sdfg.add_array(name, shape=(10, 10), dtype=dace.float64, transient=False)

    me_out, mx_out = state.add_map("outer_map", ndrange={"__i0": "0:10"})
    me_in, mx_in = state.add_map("inner_map", ndrange={"__i1": "0:10"})
    A, B = (state.add_access(name) for name in "AB")
    tskl = state.add_tasklet(
        "computation", inputs={"__in1", "__in2"}, outputs={"__out"}, code="__out = __in1 + __in2"
    )

    # construct the inner map of the map.
    state.add_edge(A, None, me_out, "IN_A", dace.Memlet("A[0:10, 0:10]"))
    me_out.add_in_connector("IN_A")
    state.add_edge(me_out, "OUT_A", me_in, "IN_A", dace.Memlet("A[__i0, 0:10]"))
    me_out.add_out_connector("OUT_A")
    me_in.add_in_connector("IN_A")
    state.add_edge(me_in, "OUT_A", tskl, "__in1", dace.Memlet("A[__i0, __i1]"))
    me_in.add_out_connector("OUT_A")

    state.add_edge(me_out, "OUT_A", me_in, "IN_A1", dace.Memlet("A[__i0, 0:10]"))
    me_in.add_in_connector("IN_A1")
    state.add_edge(me_in, "OUT_A1", tskl, "__in2", dace.Memlet("A[__i0, 9 - __i1]"))
    me_in.add_out_connector("OUT_A1")

    state.add_edge(tskl, "__out", mx_in, "IN_B", dace.Memlet("B[__i0, __i1]"))
    mx_in.add_in_connector("IN_B")
    state.add_edge(mx_in, "OUT_B", mx_out, "IN_B", dace.Memlet("B[__i0, 0:10]"))
    mx_in.add_out_connector("OUT_B")
    mx_out.add_in_connector("IN_B")
    state.add_edge(mx_out, "OUT_B", B, None, dace.Memlet("B[0:10, 0:10]"))
    mx_out.add_out_connector("OUT_B")

    # If requested add a part that is independent, i.e. is before the inner loop
    if add_independent_part != IndependentPart.NONE:
        sdfg.add_array("C", shape=(10,), dtype=dace.float64, transient=False)
        sdfg.add_scalar("tmp", dtype=dace.float64, transient=True)
        sdfg.add_scalar("tmp2", dtype=dace.float64, transient=True)
        tmp, tmp2, C = (state.add_access(name) for name in ("tmp", "tmp2", "C"))
        state.add_edge(tmp, None, tmp2, None, dace.Memlet("tmp2[0]"))
        state.add_edge(tmp2, None, mx_out, "IN_tmp", dace.Memlet("C[__i0]"))
        mx_out.add_in_connector("IN_tmp")
        state.add_edge(mx_out, "OUT_tmp", C, None, dace.Memlet("C[0:10]"))
        mx_out.add_out_connector("OUT_tmp")
        match add_independent_part:
            case IndependentPart.TASKLET:
                tskli = state.add_tasklet(
                    "independent_comp",
                    inputs={"__field"},
                    outputs={"__out"},
                    code="__out = __field[1, 1]",
                )
                state.add_edge(me_out, "OUT_A", tskli, "__field", dace.Memlet("A[0:10, 0:10]"))
                state.add_edge(tskli, "__out", tmp, None, dace.Memlet("tmp[0]"))
            case IndependentPart.NESTED_SDFG:
                nsdfg_sym, nsdfg_inp, nsdfg_out = ("S", "I", "V")
                nsdfg = _make_conditional_block_sdfg(
                    "independent_comp", nsdfg_sym, nsdfg_inp, nsdfg_out
                )
                nsdfg_node = state.add_nested_sdfg(
                    nsdfg,
                    sdfg,
                    inputs={nsdfg_inp},
                    outputs={nsdfg_out},
                    symbol_mapping={nsdfg_sym: 0},
                )
                state.add_edge(me_out, "OUT_A", nsdfg_node, nsdfg_inp, dace.Memlet("A[1, 1]"))
                state.add_edge(nsdfg_node, nsdfg_out, tmp, None, dace.Memlet("tmp[0]"))
            case _:
                raise NotImplementedError()

    sdfg.validate()
    return sdfg, state, me_out, me_in


def test_loop_blocking_inner_map():
    """
    Tests with an inner map, without an independent part.
    """
    sdfg, state, outer_map, inner_map = _make_loop_blocking_sdfg_with_inner_map(
        IndependentPart.NONE
    )
    assert all(oedge.dst is inner_map for oedge in state.out_edges(outer_map))

    count = sdfg.apply_transformations_repeated(
        gtx_transformations.LoopBlocking(blocking_size=5, blocking_parameter="__i0"),
        validate=True,
        validate_all=True,
    )
    assert count == 1
    assert all(
        oedge.dst is not inner_map and isinstance(oedge.dst, dace_nodes.MapEntry)
        for oedge in state.out_edges(outer_map)
    )
    inner_blocking_map: dace_nodes.MapEntry = next(
        oedge.dst
        for oedge in state.out_edges(outer_map)
        if isinstance(oedge.dst, dace_nodes.MapEntry)
    )
    assert inner_blocking_map is not inner_map

    assert all(oedge.dst is inner_map for oedge in state.out_edges(inner_blocking_map))


@pytest.mark.parametrize("independent_part", [IndependentPart.TASKLET, IndependentPart.NESTED_SDFG])
def test_loop_blocking_inner_map_with_independent_part(independent_part):
    """
    Tests with an inner map with an independent part.
    """
    sdfg, state, outer_map, inner_map = _make_loop_blocking_sdfg_with_inner_map(independent_part)

    # Find the parts that are independent.
    independent_node: dace_nodes.Tasklet | dace_nodes.NestedSDFG = next(
        oedge.dst
        for oedge in state.out_edges(outer_map)
        if isinstance(oedge.dst, (dace_nodes.Tasklet, dace_nodes.NestedSDFG))
    )
    assert independent_node.label == "independent_comp"
    i_access_node: dace_nodes.AccessNode = next(
        oedge.dst for oedge in state.out_edges(independent_node)
    )
    assert i_access_node.data == "tmp"

    count = sdfg.apply_transformations_repeated(
        gtx_transformations.LoopBlocking(blocking_size=5, blocking_parameter="__i0"),
        validate=True,
        validate_all=True,
    )
    assert count == 1
    inner_blocking_map: dace_nodes.MapEntry = next(
        oedge.dst
        for oedge in state.out_edges(outer_map)
        if isinstance(oedge.dst, dace_nodes.MapEntry)
    )
    assert inner_blocking_map is not inner_map

    assert all(
        oedge.dst in {inner_blocking_map, independent_node} for oedge in state.out_edges(outer_map)
    )
    assert state.scope_dict()[i_access_node] is outer_map
    assert all(oedge.dst is inner_blocking_map for oedge in state.out_edges(i_access_node))


def _make_loop_blocking_sdfg_with_independent_inner_map() -> (
    tuple[dace.SDFG, dace.SDFGState, dace_nodes.MapEntry, dace_nodes.MapEntry]
):
    """
    Creates a nested Map that is independent from the blocking parameter.
    """
    sdfg = dace.SDFG(util.unique_name("sdfg_with_inner_independent_map"))
    state = sdfg.add_state(is_start_block=True)

    sdfg.add_array("A", shape=(40, 3), dtype=dace.float64, transient=False)
    for name in "BC":
        sdfg.add_array(name, shape=(40, 8), dtype=dace.float64, transient=False)
    sdfg.add_scalar("t", dtype=dace.float64, transient=True)

    me, mx = state.add_map("main_comp", ndrange={"__i0": "0:40", "__i1": "0:8"})
    ac_A, ac_B, ac_C, ac_t = (state.add_access(name) for name in "ABCt")

    state.add_edge(ac_A, None, me, "IN_A", dace.Memlet("A[0:40, 1]"))
    me.add_in_connector("IN_A")

    # Now the inner map, that is independent. Note that the computation is stupid.
    inner_tlet = state.add_tasklet(
        "independent_tasklet", inputs={"__in"}, code="__out = __in + 3.4", outputs={"__out"}
    )
    inner_me, inner_mx = state.add_map("independent_map", ndrange={"__in_inner": "1"})
    state.add_edge(me, "OUT_A", inner_me, "IN_A", dace.Memlet("A[0:40, 1]"))
    me.add_out_connector("OUT_A")
    inner_me.add_in_connector("IN_A")
    state.add_edge(inner_me, "OUT_A", inner_tlet, "__in", dace.Memlet("A[__i0, __i_inner]"))
    inner_me.add_out_connector("OUT_A")
    state.add_edge(inner_tlet, "__out", inner_mx, "IN_t", dace.Memlet("t[0]"))
    inner_mx.add_in_connector("IN_t")
    state.add_edge(inner_mx, "OUT_t", ac_t, None, dace.Memlet("t[0]"))
    inner_mx.add_out_connector("OUT_t")

    # Create the dependent tasklet.
    dep_tlet = state.add_tasklet(
        "dependent_tasklet",
        inputs={"__in1", "__in2"},
        code="__out = __in1 * __in2",
        outputs={"__out"},
    )
    state.add_edge(ac_B, None, me, "IN_B", dace.Memlet("B[0:40, 0:8]"))
    me.add_in_connector("IN_B")
    state.add_edge(me, "OUT_B", dep_tlet, "__in1", dace.Memlet("B[__i0, __i1]"))
    me.add_out_connector("OUT_B")
    state.add_edge(ac_t, None, dep_tlet, "__in2", dace.Memlet("t[0]"))

    state.add_edge(dep_tlet, "__out", mx, "IN_C", dace.Memlet("C[__i0, __i1]"))
    mx.add_in_connector("IN_C")
    state.add_edge(mx, "OUT_C", ac_C, None, dace.Memlet("C[0:40, 0:8]"))
    mx.add_out_connector("OUT_C")
    sdfg.validate()

    return sdfg, state, me, inner_me


def test_loop_blocking_sdfg_with_independent_inner_map():
    sdfg, state, outer_me, inner_me = _make_loop_blocking_sdfg_with_independent_inner_map()

    count = sdfg.apply_transformations_repeated(
        gtx_transformations.LoopBlocking(blocking_size=2, blocking_parameter="__i1"),
        validate=True,
        validate_all=True,
    )
    assert count == 1

    # Because the inner map is independent of the blocking parameter, the inner
    #  Map has not been relocated inside the inner blocking Map.
    scope_dict = state.scope_dict()
    assert scope_dict[outer_me] is None
    assert scope_dict[inner_me] is outer_me


def _make_loop_blocking_with_reduction(
    reduction_is_dependent: bool,
) -> tuple[dace.SDFG, dace.SDFGState, dace_nodes.MapEntry, dace_nodes.LibraryNode]:
    """
    The SDFG contains an reduction node.

    Depending on `reduction_is_dependent` the node is either dependent or not.
    """
    sdfg = dace.SDFG(
        util.unique_name(
            "sdfg_with_" + ("" if reduction_is_dependent else "in") + "dependent_reduction"
        )
    )
    state = sdfg.add_state(is_start_block=True)

    sdfg.add_array(
        "A",
        shape=((40, 10, 4) if reduction_is_dependent else (40, 4)),
        dtype=dace.float64,
        transient=False,
    )
    sdfg.add_array("B", shape=(40, 10), dtype=dace.float64, transient=False)
    sdfg.add_scalar("t", dtype=dace.float64, transient=True)

    me, mx = state.add_map("outer_map", ndrange={"__i0": "0:40", "__i1": "0:10"})
    ac_A, ac_B, ac_t = (state.add_access(name) for name in ["A", "B", "t"])
    red = state.add_reduce(wcr="lambda a, b: a + b", axes=[len(sdfg.arrays["A"].shape)])
    tlet = state.add_tasklet("comp", inputs={"__in"}, code="__out = __in + 10.0", outputs={"__out"})

    state.add_edge(
        ac_A,
        None,
        me,
        "IN_A",
        dace.Memlet("A[0:40, 0:10, 0:4]" if reduction_is_dependent else "A[0:40, 0:4]"),
    )
    me.add_in_connector("IN_A")
    state.add_edge(
        me,
        "OUT_A",
        red,
        None,
        dace.Memlet("A[__i0, __i1, 0:4]" if reduction_is_dependent else "A[__i0, 0:4]"),
    )
    me.add_out_connector("OUT_A")

    state.add_edge(red, None, ac_t, None, dace.Memlet("t[0]"))
    state.add_edge(ac_t, None, tlet, "__in", dace.Memlet("t[0]"))
    state.add_edge(tlet, "__out", mx, "IN_B", dace.Memlet("B[__i0, __i1]"))
    mx.add_in_connector("IN_B")
    state.add_edge(mx, "OUT_B", ac_B, None, dace.Memlet("B[0:40, 0:10]"))
    mx.add_out_connector("OUT_B")
    sdfg.validate()

    return sdfg, state, me, red


def test_loop_blocking_dependent_reduction():
    sdfg, state, me, red = _make_loop_blocking_with_reduction(reduction_is_dependent=True)
    count = sdfg.apply_transformations_repeated(
        gtx_transformations.LoopBlocking(blocking_size=2, blocking_parameter="__i1"),
        validate=True,
        validate_all=True,
    )
    assert count == 1

    # The reduction node is dependent, so it can not be inside the scope defined
    #  by the outer map.
    scope_dict = state.scope_dict()
    assert scope_dict[red] is not me
    assert scope_dict[red] is not None
    assert isinstance(scope_dict[red], dace_nodes.MapEntry)


def test_loop_blocking_independent_reduction():
    sdfg, state, me, red = _make_loop_blocking_with_reduction(reduction_is_dependent=False)
    count = sdfg.apply_transformations_repeated(
        gtx_transformations.LoopBlocking(blocking_size=2, blocking_parameter="__i1"),
        validate=True,
        validate_all=True,
    )
    assert count == 1

    # Because the reduction does not depend on the blocking parameter, the reduction
    #  node should be inside the outer Map.
    scope_dict = state.scope_dict()
    assert scope_dict[red] is me
    assert state.out_degree(red) == 1
    assert all(
        isinstance(oedge.dst, dace_nodes.AccessNode) and oedge.dst.data == "t"
        for oedge in state.out_edges(red)
    )


def _make_mixed_memlet_sdfg(
    tskl1_independent: bool,
) -> tuple[dace.SDFG, dace.SDFGState, dace_nodes.MapEntry, dace_nodes.Tasklet, dace_nodes.Tasklet]:
    """
    Generates the SDFGs for the mixed Memlet tests.

    The SDFG that is generated has the following structure:
    - `tsklt2`, is always dependent, it has an incoming connection from the
        map entry, and an incoming, but empty, connection with `tskl1`.
    - `tskl1` is connected to the map entry, depending on `tskl1_independent`
        it is independent or dependent, it has an empty connection to `tskl2`,
        thus it is sequenced before.
    - Both have connection to other nodes down stream, but they are dependent.

    Returns:
        A tuple containing the following objects.
        - The SDFG.
        - The SDFG state.
        - The outer map entry node.
        - `tskl1`.
        - `tskl2`.
    """
    sdfg = dace.SDFG(util.unique_name("mixed_memlet_sdfg"))
    state = sdfg.add_state(is_start_block=True)
    names_array = ["A", "B", "C"]
    names_scalar = ["tmp1", "tmp2"]
    for aname in names_array:
        sdfg.add_array(
            aname,
            shape=((10,) if aname == "A" else (10, 10)),
            dtype=dace.float64,
            transient=False,
        )
    for sname in names_scalar:
        sdfg.add_scalar(
            sname,
            dtype=dace.float64,
            transient=True,
        )
    A, B, C, tmp1, tmp2 = (state.add_access(name) for name in names_array + names_scalar)

    me, mx = state.add_map("outer_map", ndrange={"i": "0:10", "j": "0:10"})
    tskl1 = state.add_tasklet(
        "tskl1",
        inputs={"__in1"},
        outputs={"__out"},
        code="__out = __in1" if tskl1_independent else "__out = __in1 + j",
    )
    tskl2 = state.add_tasklet(
        "tskl2",
        inputs={"__in1"},
        outputs={"__out"},
        code="__out = __in1 + 10.0",
    )
    tskl3 = state.add_tasklet(
        "tskl3",
        inputs={"__in1", "__in2"},
        outputs={"__out"},
        code="__out = __in1 + __in2",
    )

    state.add_edge(A, None, me, "IN_A", dace.Memlet("A[0:10]"))
    me.add_in_connector("IN_A")
    state.add_edge(me, "OUT_A", tskl1, "__in1", dace.Memlet("A[i]"))
    me.add_out_connector("OUT_A")
    state.add_edge(tskl1, "__out", tmp1, None, dace.Memlet("tmp1[0]"))

    state.add_edge(B, None, me, "IN_B", dace.Memlet("B[0:10, 0:10]"))
    me.add_in_connector("IN_B")
    state.add_edge(me, "OUT_B", tskl2, "__in1", dace.Memlet("B[i, j]"))
    me.add_out_connector("OUT_B")
    state.add_edge(tskl2, "__out", tmp2, None, dace.Memlet("tmp2[0]"))

    # Add the empty Memlet that sequences `tskl1` before `tskl2`.
    state.add_edge(tskl1, None, tskl2, None, dace.Memlet())

    state.add_edge(tmp1, None, tskl3, "__in1", dace.Memlet("tmp1[0]"))
    state.add_edge(tmp2, None, tskl3, "__in2", dace.Memlet("tmp2[0]"))
    state.add_edge(tskl3, "__out", mx, "IN_C", dace.Memlet("C[i, j]"))
    mx.add_in_connector("IN_C")
    state.add_edge(mx, "OUT_C", C, None, dace.Memlet("C[0:10, 0:10]"))
    mx.add_out_connector("OUT_C")
    sdfg.validate()

    return (sdfg, state, me, tskl1, tskl2)


def _apply_and_run_mixed_memlet_sdfg(sdfg: dace.SDFG) -> None:
    ref = {
        "A": np.array(np.random.rand(10), dtype=np.float64, copy=True),
        "B": np.array(np.random.rand(10, 10), dtype=np.float64, copy=True),
        "C": np.array(np.random.rand(10, 10), dtype=np.float64, copy=True),
    }
    res = copy.deepcopy(ref)
    sdfg(**ref)

    count = sdfg.apply_transformations_repeated(
        gtx_transformations.LoopBlocking(blocking_size=2, blocking_parameter="j"),
        validate=True,
        validate_all=True,
    )
    assert count == 1, f"Expected one application, but git {count}"
    sdfg(**res)
    assert all(np.allclose(ref[name], res[name]) for name in ref)


def _make_conditional_block_sdfg(sdfg_label: str, sym: str, inp: str, out: str):
    sdfg = dace.SDFG(sdfg_label)
    for data in [inp, out]:
        sdfg.add_scalar(data, dtype=dace.float64)

    if_region = dace.sdfg.state.ConditionalBlock("if")
    sdfg.add_node(if_region)
    entry_state = sdfg.add_state("entry", is_start_block=True)
    sdfg.add_edge(entry_state, if_region, dace.InterstateEdge())

    then_body = dace.sdfg.state.ControlFlowRegion("then_body", sdfg=sdfg)
    tstate = then_body.add_state("true_branch", is_start_block=True)
    if_region.add_branch(dace.sdfg.state.CodeBlock(f"{sym} % 2 == 0"), then_body)
    tskli = tstate.add_tasklet("write_0", inputs={"inp"}, outputs={"val"}, code=f"val = inp + 0")
    tstate.add_edge(tstate.add_access(inp), None, tskli, "inp", dace.Memlet(f"{inp}[0]"))
    tstate.add_edge(tskli, "val", tstate.add_access(out), None, dace.Memlet(f"{out}[0]"))

    else_body = dace.sdfg.state.ControlFlowRegion("else_body", sdfg=sdfg)
    fstate = else_body.add_state("false_branch", is_start_block=True)
    if_region.add_branch(dace.sdfg.state.CodeBlock(f"{sym} % 2 != 0"), else_body)
    tskli = fstate.add_tasklet("write_1", inputs={"inp"}, outputs={"val"}, code=f"val = inp + 1")
    fstate.add_edge(fstate.add_access(inp), None, tskli, "inp", dace.Memlet(f"{inp}[0]"))
    fstate.add_edge(tskli, "val", fstate.add_access(out), None, dace.Memlet(f"{out}[0]"))

    return sdfg


def test_loop_blocking_mixed_memlets_1():
    sdfg, state, me, tskl1, tskl2 = _make_mixed_memlet_sdfg(True)
    mx = state.exit_node(me)

    _apply_and_run_mixed_memlet_sdfg(sdfg)
    scope_dict = state.scope_dict()

    # Ensure that `tskl1` is independent.
    assert scope_dict[tskl1] is me

    # The output of `tskl1`, which is `tmp1` should also be classified as independent.
    tmp1 = next(iter(edge.dst for edge in state.out_edges(tskl1) if not edge.data.is_empty()))
    assert scope_dict[tmp1] is me
    assert isinstance(tmp1, dace_nodes.AccessNode)
    assert tmp1.data == "tmp1"

    # Find the inner map.
    inner_map_entry: dace_nodes.MapEntry = scope_dict[tskl2]
    assert inner_map_entry is not me and isinstance(inner_map_entry, dace_nodes.MapEntry)
    inner_map_exit: dace_nodes.MapExit = state.exit_node(inner_map_entry)

    outer_scope = {tskl1, tmp1, inner_map_entry, inner_map_exit, mx}
    for node in state.nodes():
        if scope_dict[node] is None:
            assert (node is me) or (
                isinstance(node, dace_nodes.AccessNode) and node.data in {"A", "B", "C"}
            )
        elif scope_dict[node] is me:
            assert node in outer_scope
        else:
            assert (
                (node is inner_map_exit)
                or (isinstance(node, dace_nodes.AccessNode) and node.data == "tmp2")
                or (isinstance(node, dace_nodes.Tasklet) and node.label in {"tskl2", "tskl3"})
            )


def test_loop_blocking_mixed_memlets_2():
    sdfg, state, me, tskl1, tskl2 = _make_mixed_memlet_sdfg(False)
    mx = state.exit_node(me)

    _apply_and_run_mixed_memlet_sdfg(sdfg)
    scope_dict = state.scope_dict()

    # Because `tskl1` is now dependent, everything is now dependent.
    inner_map_entry = scope_dict[tskl1]
    assert isinstance(inner_map_entry, dace_nodes.MapEntry)
    assert inner_map_entry is not me

    for node in state.nodes():
        if scope_dict[node] is None:
            assert (node is me) or (
                isinstance(node, dace_nodes.AccessNode) and node.data in {"A", "B", "C"}
            )
        elif scope_dict[node] is me:
            assert isinstance(node, dace_nodes.MapEntry) or (node is mx)
        else:
            assert scope_dict[node] is inner_map_entry


def test_loop_blocking_no_independent_nodes():
    import dace

    sdfg = dace.SDFG(util.unique_name("mixed_memlet_sdfg"))
    state = sdfg.add_state(is_start_block=True)
    names = ["A", "B", "C"]
    for aname in names:
        sdfg.add_array(
            aname,
            shape=(10, 10),
            dtype=dace.float64,
            transient=False,
        )
    A = state.add_access("A")
    _, me, mx = state.add_mapped_tasklet(
        "fully_dependent_computation",
        map_ranges={"__i0": "0:10", "__i1": "0:10"},
        inputs={"__in1": dace.Memlet("A[__i0, __i1]")},
        code="__out = __in1 + 10.0",
        outputs={"__out": dace.Memlet("B[__i0, __i1]")},
        external_edges=True,
        input_nodes={A},
    )
    nsdfg_sym, nsdfg_inp, nsdfg_out = ("S", "I", "V")
    nsdfg = _make_conditional_block_sdfg("dependent_component", nsdfg_sym, nsdfg_inp, nsdfg_out)
    nsdfg_node = state.add_nested_sdfg(
        nsdfg, sdfg, inputs={nsdfg_inp}, outputs={nsdfg_out}, symbol_mapping={nsdfg_sym: "__i1"}
    )
    state.add_memlet_path(A, me, nsdfg_node, dst_conn=nsdfg_inp, memlet=dace.Memlet("A[1,1]"))
    state.add_memlet_path(
        nsdfg_node,
        mx,
        state.add_access("C"),
        src_conn=nsdfg_out,
        memlet=dace.Memlet("C[__i0, __i1]"),
    )
    sdfg.validate()

    # Because there is nothing that is independent the transformation will
    #  not apply if `require_independent_nodes` is enabled.
    count = sdfg.apply_transformations_repeated(
        gtx_transformations.LoopBlocking(
            blocking_size=2,
            blocking_parameter="__i1",
            require_independent_nodes=True,
        ),
        validate=True,
        validate_all=True,
    )
    assert count == 0

    # But it will apply once this requirement is lifted.
    count = sdfg.apply_transformations_repeated(
        gtx_transformations.LoopBlocking(
            blocking_size=2,
            blocking_parameter="__i1",
            require_independent_nodes=False,
        ),
        validate=True,
        validate_all=True,
    )
    assert count == 1


def _make_only_last_two_elements_sdfg() -> dace.SDFG:
    sdfg = dace.SDFG(util.unique_name("simple_block_sdfg"))
    state = sdfg.add_state("state", is_start_block=True)
    sdfg.add_symbol("N", dace.int32)
    sdfg.add_symbol("B", dace.int32)
    sdfg.add_symbol("M", dace.int32)

    for name in "acb":
        sdfg.add_array(
            name,
            shape=(20, 10),
            dtype=dace.float64,
        )

    state.add_mapped_tasklet(
        "computation",
        map_ranges={"i": "B:N", "k": "(M-2):M"},
        inputs={
            "__in1": dace.Memlet("a[i, k]"),
            "__in2": dace.Memlet("b[i, k]"),
        },
        code="__out = __in1 + __in2",
        outputs={"__out": dace.Memlet("c[i, k]")},
        external_edges=True,
    )
    sdfg.validate()

    return sdfg


def test_only_last_two_elements_sdfg():
    sdfg = _make_only_last_two_elements_sdfg()

    def ref_comp(a, b, c, B, N, M):
        for i in range(B, N):
            for k in range(M - 2, M):
                c[i, k] = a[i, k] + b[i, k]

    count = sdfg.apply_transformations_repeated(
        gtx_transformations.LoopBlocking(
            blocking_size=1,
            blocking_parameter="k",
            require_independent_nodes=False,
        ),
        validate=True,
        validate_all=True,
    )
    assert count == 1

    ref = {
        "a": np.array(np.random.rand(20, 10), dtype=np.float64),
        "b": np.array(np.random.rand(20, 10), dtype=np.float64),
        "c": np.zeros((20, 10), dtype=np.float64),
        "B": 0,
        "N": 20,
        "M": 6,
    }
    res = copy.deepcopy(ref)

    ref_comp(**ref)
    sdfg(**res)

    assert np.allclose(ref["c"], res["c"])
