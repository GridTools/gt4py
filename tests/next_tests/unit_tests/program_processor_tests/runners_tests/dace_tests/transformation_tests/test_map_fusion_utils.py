# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


import pytest


dace = pytest.importorskip("dace")
from dace import subsets as dace_subsets

from gt4py.next.program_processors.runners.dace.transformations import (
    map_fusion_utils as gtx_map_fusion_utils,
)

import numpy as np

from . import util


def test_copy_map_graph():
    N = dace.symbol("N", dace.int32)
    sdfg = dace.SDFG(util.unique_name("copy_map_graph"))
    A, A_desc = sdfg.add_array("A", [N], dtype=dace.float64)
    B = sdfg.add_datadesc("B", A_desc.clone())
    st = sdfg.add_state()
    A_node = st.add_access(A)
    B_node = st.add_access(B)
    _, first_map_entry, first_map_exit = st.add_mapped_tasklet(
        "plus1",
        {"i": "0:N"},
        code="_out = _inp + 1.0",
        inputs={"_inp": dace.Memlet(data=A, subset="i")},
        outputs={"_out": dace.Memlet(data=B, subset="i")},
        input_nodes={A_node},
        output_nodes={B_node},
        external_edges=True,
    )
    sdfg.validate()
    assert len(st.nodes()) == 5

    # We have created an SDFG with a mapped tasklet, that adds 1 to input array 'A'
    # and writes the result to output array 'B'. We store the SDFG hash.
    sdfg_hash = sdfg.hash_sdfg()

    # We verify that it computes the right thing.
    A_data = np.random.rand(10)
    B1_data = np.zeros_like(A_data)
    B2_data = np.zeros_like(A_data)
    sdfg(A=A_data, B=B1_data, N=10)
    assert np.allclose(B1_data, A_data + 1.0)

    # Now we copy the mapped tasklöet to a new map scope.
    #  Note that we use an empty suffix to have an execat replica, including names.
    second_map_extry, second_map_exit = gtx_map_fusion_utils.copy_map_graph(
        sdfg, st, first_map_entry, first_map_exit, suffix=None
    )
    assert second_map_extry.map is second_map_exit.map
    assert second_map_extry.map is not first_map_entry.map
    assert len(st.nodes()) == 8

    # We delete the original map and verify that the SDFG hash is the same.
    #  Note that this is expected because we use an empty suffix to copy the nodes.
    gtx_map_fusion_utils.delete_map(st, first_map_entry, first_map_exit)
    assert sdfg.hash_sdfg() == sdfg_hash

    # We should still get the same result.
    sdfg(A=A_data, B=B2_data, N=10)
    assert np.allclose(B2_data, A_data + 1.0)


@pytest.mark.parametrize(
    "map_ranges",
    [
        ({"i": "0:N"}, {"i": "0:N"}, {}, {}),  # same ndrange, no splitting
        ({"i": "0:N"}, {"i": "N:N+N"}, {}, {}),  # non-overlapping, no splitting
        ({"i": "0:N"}, {"i": "1:N"}, {"0", "1:N"}, {"1:N"}),  # partially overlapping
        ({"i": "0:M"}, {"i": "1:N"}, {}, {}),  # symbolic ndrange, no splitting
        (
            {"i": "0:M", "j": "0:N-1"},
            {"i": "1:M", "j": "0:N"},
            {"0,0:N-1", "1:M,0:N-1"},
            {"1:M,0:N-1", "1:M,N-1"},
        ),  # overlapping, same ndrange order
        (
            {"i": "0:M", "j": "0:N-1"},
            {"j": "0:N", "i": "1:M"},
            {"0,0:N-1", "1:M,0:N-1"},
            {"0:N-1,1:M", "N-1,1:M"},
        ),  # overlapping, different ndrange order
        (
            {"i": "0:M", "j": "0:N-1", "k": "3:7"},
            {"i": "1:M", "j": "0:N"},
            {},
            {},
        ),  # overlapping but one extra ndrange parameter on first map, no splitting
    ],
)
def test_split_overlapping_map_range(map_ranges):
    first_ndrange, second_ndrange = map_ranges[0:2]

    sdfg = dace.SDFG(util.unique_name("split_overlapping_map_range"))
    st = sdfg.add_state()
    first_map_entry, _ = st.add_map("first", first_ndrange)
    second_map_entry, _ = st.add_map("second", second_ndrange)

    ret = gtx_map_fusion_utils.split_overlapping_map_range(
        first_map_entry.map, second_map_entry.map
    )
    if len(map_ranges[2]) == 0 and len(map_ranges[3]) == 0:
        assert ret is None

    else:
        splitted_range_first_map, splitted_range_second_map = ret
        assert set(splitted_range_first_map) == {
            dace_subsets.Range.from_string(expected_range) for expected_range in map_ranges[2]
        }
        assert set(splitted_range_second_map) == {
            dace_subsets.Range.from_string(expected_range) for expected_range in map_ranges[3]
        }
