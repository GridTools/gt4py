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


def _make_empty_memlets_sdfg() -> tuple[dace.SDFG, dace.SDFGState]:
    sdfg = dace.SDFG(util.unique_name("empty_memlets"))
    state = sdfg.add_state(is_start_block=True)

    anames = ["a", "b", "c"]
    for name in anames:
        sdfg.add_array(
            name,
            shape=(10,),
            dtype=dace.float64,
            transient=False,
        )
    sdfg.arrays["b"].transient = True

    a, b, c = (state.add_access(name) for name in anames)

    state.add_nedge(a, b, dace.Memlet("a[1:1] -> [1:1]"))
    state.add_nedge(a, b, dace.Memlet("a[2:2] -> [2:2]"))
    state.add_nedge(b, c, dace.Memlet("b[4:4] -> [4:4]"))
    state.add_nedge(b, c, dace.Memlet("b[5:5] -> [5:5]"))
    state.add_nedge(a, c, dace.Memlet("a[8:8] -> [8:8]"))
    state.add_nedge(a, c, dace.Memlet("a[9] -> [9]"))

    sdfg.validate()
    return sdfg, state


def test_remove_empty_dataflow():
    sdfg, state = _make_empty_memlets_sdfg()
    assert state.number_of_edges() == 6
    assert util.count_nodes(sdfg, dace_nodes.AccessNode) == 3

    # Out of the three 6 edges only the one between `a` and `c` is valid.
    #  The function will process and clean two nodes, `a` and `b` and thus the
    #  return value will be 2.
    nb_removed = gtx_transformations.gt_eliminate_dead_dataflow(
        sdfg=sdfg,
        run_simplify=False,
        validate=True,
        validate_all=True,
    )

    ac_nodes = util.count_nodes(sdfg, dace_nodes.AccessNode, return_nodes=True)
    assert nb_removed == 2
    assert len(ac_nodes) == 2
    assert {"a", "c"} == {ac.data for ac in ac_nodes}
    assert {"a", "c"} == sdfg.arrays.keys()
    assert state.number_of_edges() == 1

    a_c_edge = list(state.edges())[0]
    assert a_c_edge.data.subset.min_element()[0] == 9
    assert a_c_edge.data.other_subset.min_element()[0] == 9
