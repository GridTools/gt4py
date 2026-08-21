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
