# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dace
from dace.sdfg.state import LoopRegion

from gt4py.next.program_processors.runners.dace.transformations import (
    utils as gtx_transformations_utils,
)

from . import util


def test_find_successor_state():
    sdfg = dace.SDFG(util.unique_name("find_successor_state"))
    state1 = sdfg.add_state(is_start_block=True)
    state2 = sdfg.add_state_after(state1)
    sdfg.validate()

    assert gtx_transformations_utils.find_successor_state(state1) == [state2]

    # `state2` is a terminal control-flow block of the SDFG, thus it has no
    #  successor. However, the function must not walk above the root region
    #  (previously this crashed with an `AttributeError`).
    assert gtx_transformations_utils.find_successor_state(state2) == []


def test_find_successor_state_terminal_loop_region():
    """Terminal inside a `LoopRegion` that is itself terminal.

    The successor of the last state of the loop body is not expressible, thus
    the function must return an empty list and not walk above the root region
    (previously this crashed with an `AttributeError`).
    """
    sdfg = dace.SDFG(util.unique_name("find_successor_state_terminal_loop_region"))
    loop = LoopRegion(
        label="scan_loop",
        condition_expr="i < 2",
        loop_var="i",
        initialize_expr="i = 0",
        update_expr="i = i + 1",
    )
    sdfg.add_node(loop, is_start_block=True)
    body_state1 = loop.add_state("body_state1", is_start_block=True)
    body_state2 = loop.add_state_after(body_state1)
    sdfg.validate()

    assert gtx_transformations_utils.find_successor_state(body_state1) == [body_state2]
    assert gtx_transformations_utils.find_successor_state(body_state2) == []


def test_find_successor_state_non_terminal_loop_region():
    """A non-terminal `LoopRegion` exposes its outer successor.

    When the last state of the loop body has no local successors, the function
    must go up the hierarchy and return the state that follows the loop.
    """
    sdfg = dace.SDFG(util.unique_name("find_successor_state_non_terminal_loop_region"))
    start_state = sdfg.add_state(is_start_block=True)
    loop = LoopRegion(
        label="scan_loop",
        condition_expr="i < 2",
        loop_var="i",
        initialize_expr="i = 0",
        update_expr="i = i + 1",
    )
    sdfg.add_edge(start_state, loop, dace.InterstateEdge())
    body_state1 = loop.add_state("body_state1", is_start_block=True)
    body_state2 = loop.add_state_after(body_state1)
    after_loop = sdfg.add_state("after_loop")
    sdfg.add_edge(loop, after_loop, dace.InterstateEdge())
    sdfg.validate()

    assert gtx_transformations_utils.find_successor_state(body_state1) == [body_state2]
    assert gtx_transformations_utils.find_successor_state(body_state2) == [after_loop]


def test_find_successor_state_nested_sdfg():
    """Terminal state inside a nested SDFG.

    The function must stop at the root region of the nested SDFG and return an
    empty list, instead of attempting to ascend into the surrounding SDFG.
    """
    sdfg = dace.SDFG(util.unique_name("find_successor_state_nested_sdfg"))
    outer_state = sdfg.add_state(is_start_block=True)

    nested_sdfg = dace.SDFG(util.unique_name("nested_sdfg"))
    inner_state1 = nested_sdfg.add_state("inner_state1", is_start_block=True)
    inner_state2 = nested_sdfg.add_state_after(inner_state1)

    outer_state.add_nested_sdfg(nested_sdfg, inputs={}, outputs={})
    sdfg.validate()

    assert gtx_transformations_utils.find_successor_state(inner_state1) == [inner_state2]
    assert gtx_transformations_utils.find_successor_state(inner_state2) == []
