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

from . import util


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
            {"0,0:N-1,3:7", "1:M,0:N-1,3:7"},
            {"1:M,0:N-1", "1:M,N-1"},
        ),  # overlapping, different ndrange parameters
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
