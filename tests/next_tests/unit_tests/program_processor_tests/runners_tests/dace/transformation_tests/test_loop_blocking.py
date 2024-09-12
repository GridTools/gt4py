# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Callable
import dace
import copy
import numpy as np
import pytest

from dace.sdfg import nodes as dace_nodes, propagation as dace_propagation

from gt4py.next.program_processors.runners.dace_fieldview import (
    transformations as gtx_transformations,
)
from . import util

pytestmark = [pytest.mark.requires_dace, pytest.mark.usefixtures("set_dace_settings")]


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
        map_ranges=dict(i=f"0:N", j=f"0:M"),
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
    first_tasklet_indiependent: bool,
    only_empty_memlets: bool,
) -> tuple[
    dace.SDFG, dace_nodes.MapEntry, dace_nodes.Tasklet, dace_nodes.AccessNode, dace_nodes.Tasklet
]:
    """Generates an SDFG with an empty tasklet.

    The map contains two (serial) tasklets, connected through an access node.
    The first tasklet has an empty memlet that connects it to the map entry.
    Depending on `first_tasklet_indiependent` the tasklet is either independent
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
        code="__out0 = 1.0" if first_tasklet_indiependent else "__out0 = j",
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
        first_tasklet_indiependent=True,
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
        first_tasklet_indiependent=False,
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
        first_tasklet_indiependent=False,
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
