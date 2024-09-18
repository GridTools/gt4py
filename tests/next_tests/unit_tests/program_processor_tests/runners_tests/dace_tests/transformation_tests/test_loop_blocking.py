# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import copy
from typing import Callable

import numpy as np
import pytest


dace = pytest.importorskip("dace")
from dace.sdfg import nodes as dace_nodes, propagation as dace_propagation

from gt4py.next.program_processors.runners.dace_fieldview import (
    transformations as gtx_transformations,
)

from . import pytestmark
from . import util


def _get_simple_sdfg() -> tuple[dace.SDFG, Callable[[np.ndarray, np.ndarray], np.ndarray]]:
    """Creates a simple SDFG.

    The k blocking transformation can be applied to the SDFG, however no node
    can be taken out. This is because how it is constructed. However, applying
    some simplistic transformations this can be done.
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
    sdfg.apply_transformations_repeated(
        gtx_transformations.LoopBlocking(blocking_size=10, blocking_parameter="j"),
        validate=True,
        validate_all=True,
    )

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
    sdfg.apply_transformations_repeated(
        gtx_transformations.LoopBlocking(blocking_size=10, blocking_parameter="j"),
        validate=True,
        validate_all=True,
    )

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
    ret = sdfg.apply_transformations_repeated(
        gtx_transformations.LoopBlocking(blocking_size=10, blocking_parameter="j"),
        validate=True,
        validate_all=True,
    )
    assert ret == 1, f"Expected that the transformation was applied 1 time, but it was {ret}."

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
