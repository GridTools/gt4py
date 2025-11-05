# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

dace = pytest.importorskip("dace")
from dace.sdfg import nodes as dace_nodes

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
)

from . import util

import dace


def _make_simple_double_write_sdfg() -> tuple[
    dace.SDFG, dace.SDFGState, dace_nodes.AccessNode, dace_nodes.AccessNode, dace_nodes.MapExit
]:
    sdfg = dace.SDFG(util.unique_name("double_write_elimination_1"))
    state = sdfg.add_state(is_start_block=True)

    sdfg.add_array(
        "a",
        shape=(10,),
        dtype=dace.float64,
        transient=False,
    )
    sdfg.add_array(
        "b",
        shape=(10,),
        dtype=dace.float64,
        transient=True,
    )
    sdfg.add_array(
        "c",
        shape=(10, 4),
        dtype=dace.float64,
        transient=False,
    )
    a, b, c = (state.add_access(name) for name in "abc")

    _, _, mx = state.add_mapped_tasklet(
        "computation",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("a[__i]")},
        code="__out = __in + 1.0",
        outputs={"__out": dace.Memlet("b[__i]")},
        external_edges=True,
        input_nodes={a},
        output_nodes={b},
    )

    state.add_nedge(b, c, dace.Memlet("b[0:10] -> [0:10, 0]"))
    state.add_nedge(b, c, dace.Memlet("b[0:10] -> [0:10, 2]"))

    sdfg.validate()
    return sdfg, state, b, c, mx


def test_remove_double_write1():
    sdfg, state, b, c, mx = _make_simple_double_write_sdfg()

    pre_ac = util.count_nodes(sdfg, dace_nodes.AccessNode, True)
    assert len(pre_ac) == 3
    assert b in pre_ac
    assert c in pre_ac
    assert b.data in sdfg.arrays
    assert c.data in sdfg.arrays
    assert util.count_nodes(sdfg, dace_nodes.MapExit) == 1
    assert state.out_degree(mx) == 1
    assert state.out_degree(b) == 2
    assert state.in_degree(b) == 1

    ref, res = util.make_sdfg_args(sdfg)
    util.compile_and_run_sdfg(sdfg, **ref)

    nb_applied = sdfg.apply_transformations_repeated(
        gtx_transformations.DoubleWriteRemover(
            single_use_data=None,
        ),
        validate_all=True,
        validate=True,
    )
    assert nb_applied == 1

    after_ac = util.count_nodes(sdfg, dace_nodes.AccessNode, True)
    assert b not in after_ac
    assert b.data not in sdfg.arrays
    # Because there is a new scalar one that was added.
    assert len(after_ac) == 3

    assert util.count_nodes(sdfg, dace_nodes.MapExit) == 1
    assert state.out_degree(mx) == 2
    assert all(oedge.dst is c for oedge in state.out_edges(mx))

    util.compile_and_run_sdfg(sdfg, **res)
    assert util.compare_sdfg_res(ref=ref, res=res)
