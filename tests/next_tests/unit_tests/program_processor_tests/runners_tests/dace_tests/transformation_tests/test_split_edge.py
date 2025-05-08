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

    ref, res = util.make_sdfg_args(sdfg)
    util.evaluate_sdfg(sdfg, ref)

    nb_apply = sdfg.apply_transformations_repeated(
        gtx_transformations.SplitMemlet,
        validate=True,
        validate_all=True,
    )
    assert nb_apply == 1

    assert state.number_of_edges() == 4

    ac_nodes = util.count_nodes(state, dace_nodes.AccessNode, True)
    assert len({ac.data for ac in ac_nodes}) == 4

    tmp = next(ac for ac in ac_nodes if ac.data == "tmp")
    assert state.out_degree(tmp) == 2
    reads_to_tmp = {
        (oedge.data.src_subset.min_element()[0], oedge.data.src_subset.size()[0])
        for oedge in state.out_edges(tmp)
    }
    assert {(0, 10), (10, 10)} == reads_to_tmp

    util.evaluate_sdfg(sdfg, res)
    assert util.compare_sdfg_res(ref, res)


def _make_split_edge_mock_apply_diffusion_to_w_sdfg() -> (
    tuple[dace.SDFG, dace.SDFGState, dace_nodes.AccessNode]
):
    # Test is roughly equivalent to what we see in `apply_diffusion_to_w`
    #  Although instead of Maps we sometimes use direct edges.
    sdfg = dace.SDFG(util.unique_name("split_edge_mock_apply_diffusion_to_w"))

    state = sdfg.add_state(is_start_block=True)

    sdfg.add_array("s1", shape=(30, 80), dtype=dace.float64, transient=False)
    sdfg.add_array("s2", shape=(70, 80), dtype=dace.float64, transient=False)
    sdfg.add_array("tmp", shape=(100, 80), dtype=dace.float64, transient=True)
    sdfg.add_array("dst", shape=(100, 80), dtype=dace.float64, transient=False)

    tmp = state.add_access("tmp")
    dst = state.add_access("dst")

    state.add_nedge(state.add_access("s1"), tmp, dace.Memlet("s1[0:30, 0:80] -> [0:30, 0:80]"))
    # In the original case this is a Map.
    state.add_nedge(state.add_access("s2"), tmp, dace.Memlet("s2[0:70, 0:80] -> [30:100, 0:80]"))

    state.add_nedge(tmp, dst, dace.Memlet("tmp[0:100, 0] -> [0:100, 0]"))
    state.add_nedge(tmp, dst, dace.Memlet("tmp[0:100, 13:80] -> [0:100, 13:80]"))
    state.add_nedge(tmp, dst, dace.Memlet("tmp[0:30, 1:13] -> [0:30, 1:13]"))
    # In the original case this is a Map.
    state.add_nedge(tmp, dst, dace.Memlet("tmp[30:100, 1:13] -> [30:100, 1:13]"))
    sdfg.validate()

    return sdfg, state, tmp


def test_split_edge_mock_apply_diffusion_to_w():
    sdfg, state, tmp_ac = _make_split_edge_mock_apply_diffusion_to_w_sdfg()
    assert not gtx_transformations.SplitAccessNode.can_be_applied_to(sdfg=sdfg, access_node=tmp_ac)

    assert util.count_nodes(state, dace_nodes.AccessNode) == 4
    assert state.out_degree(tmp_ac) == 4

    ref, res = util.make_sdfg_args(sdfg)
    util.evaluate_sdfg(sdfg, ref)

    nb_apply = sdfg.apply_transformations_repeated(
        gtx_transformations.SplitMemlet,
        validate=True,
        validate_all=True,
    )
    assert nb_apply == 1

    assert util.count_nodes(state, dace_nodes.AccessNode) == 4
    assert state.out_degree(tmp_ac) == 6

    assert gtx_transformations.SplitAccessNode.can_be_applied_to(sdfg=sdfg, access_node=tmp_ac)

    util.evaluate_sdfg(sdfg, res)
    assert util.compare_sdfg_res(ref, res)
