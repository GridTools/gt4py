# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import numpy as np

from gt4py.next.program_processors.runners.dace_fieldview import (
    transformations as gtx_transformations,
)

# from . import util


# dace = pytest.importorskip("dace")
from dace.sdfg import nodes as dace_nodes
import dace


def _mk_distributed_buffer_sdfg() -> tuple[dace.SDFG, dace.SDFGState]:
    sdfg = dace.SDFG("NAME")  # util.unique_name("distributed_buffer_sdfg"))

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

    return sdfg, state1


def test_distributed_buffer_remover():
    sdfg, state1 = _mk_distributed_buffer_sdfg()
    assert state1.number_of_nodes() == 5
    assert not any(dnode.data == "b" for dnode in state1.data_nodes())

    res = gtx_transformations.gt_reduce_distributed_buffering(sdfg)
    assert res is not None

    # Because the final state has now become empty
    assert sdfg.number_of_nodes() == 3
    assert state1.number_of_nodes() == 6
    assert any(dnode.data == "b" for dnode in state1.data_nodes())
    assert any(dnode.data == "tmp" for dnode in state1.data_nodes())
