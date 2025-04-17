# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest
import copy
import numpy as np

dace = pytest.importorskip("dace")
from dace.sdfg import nodes as dace_nodes

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
)

from . import util

import dace


def _perform_test(
    sdfg: dace.SDFG,
    explected_applies: int,
    removed_transients: set[str] | None = None,
) -> None:
    ref = {
        name: np.array(np.random.rand(*desc.shape), copy=True, dtype=desc.dtype.as_numpy_dtype())
        for name, desc in sdfg.arrays.items()
        if not desc.transient
    }
    res = copy.deepcopy(ref)

    if removed_transients is not None:
        assert removed_transients.issubset(
            name for name, desc in sdfg.arrays.items() if desc.transient
        )

    if explected_applies != 0:
        csdfg_ref = sdfg.compile()
        csdfg_ref(**ref)

    nb_apply = gtx_transformations.gt_split_access_nodes(
        sdfg=sdfg,
        validate=True,
        validate_all=True,
    )
    assert nb_apply == explected_applies

    if explected_applies == 0:
        return

    csdfg_res = sdfg.compile()
    csdfg_res(**res)

    assert all(np.allclose(ref[name], res[name]) for name in ref.keys())

    if removed_transients is not None:
        assert all(
            name not in removed_transients for name, desc in sdfg.arrays.items() if desc.transient
        )


def _make_split_edge_two_ac_producer_one_ac_consumer_1d_sdfg() -> tuple[dace.SDFG, dace.SDFGState]:
    sdfg = dace.SDFG(util.unique_name("split_edge_two_ac_producer_one_ac_consumer_1d"))
    state = sdfg.add_state(is_start_block=True)

    sdfg.add_array("src1", shape=(10,), dtype=dace.float64, transient=False)
    sdfg.add_array("src2", shape=(10,), dtype=dace.float64, transient=False)
    sdfg.add_array("tmp", shape=(20,), dtype=dace.float64, transient=True)
    sdfg.add_array("dst", shape=(20,), dtype=dace.float64, transient=False)

    tmp = state.add_access("tmp")

    state.add_edge(state.add_access("src1"), None, tmp, None, dace.Memlet("src1[0:10] -> [0:10]"))
    state.add_edge(state.add_access("src2"), None, tmp, None, dace.Memlet("src2[0:10] -> [10:20]"))
    state.add_edge(tmp, None, state.add_access("dst"), None, dace.Memlet("tmp[0:20] -> [0:20]"))
    sdfg.validate()

    return sdfg, state


def test_split_edge_two_ac_producer_one_ac_consumer_1d():
    sdfg, state = _make_split_edge_two_ac_producer_one_ac_consumer_1d_sdfg()

    assert state.number_of_edges() == 3

    nb_apply = sdfg.apply_transformations_repeated(
        gtx_transformations.SplitMemlet,
        validate=True,
        validate_all=True,
    )
    assert nb_apply == 1

    assert state.number_of_edges() == 4

    ac_nodes = util.count_nodes(state, dace_nodes.AccessNode, True)
    assert len({ac.data for ac in ac_nodes}) == 4
