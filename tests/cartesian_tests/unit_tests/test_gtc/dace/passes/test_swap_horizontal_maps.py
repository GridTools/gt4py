# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from dace import nodes, subsets
from dace.sdfg.analysis.schedule_tree import treenodes as tn

from gt4py.cartesian.gtc.dace.passes import SwapHorizontalMaps

# Because "dace tests" filter by `requires_dace`, we still need to add the marker.
# This global variable adds the marker to all test functions in this module.
pytestmark = pytest.mark.requires_dace


def test_swap_horizontal_maps() -> None:
    root = tn.ScheduleTreeRoot(name="tester", children=[])
    k_loop = tn.MapScope(
        node=nodes.MapEntry(
            map=nodes.Map("vertical map", ["__k"], subsets.Range.from_string("0:10:2"))
        ),
        children=[],
    )
    k_loop.parent = root
    ji_loop = tn.MapScope(
        node=nodes.MapEntry(
            map=nodes.Map(
                "horizontal maps", ["__j", "__i"], subsets.Range([(0, 4, 1), (0, 7, 2, 2)])
            )
        ),
        children=[],
    )
    ji_loop.parent = k_loop
    k_loop.children.append(ji_loop)
    root.children.append(k_loop)

    flipper = SwapHorizontalMaps()
    flipper.visit(root)

    horizontal_maps = ji_loop.node.map
    assert horizontal_maps.params[0] == "__i"
    assert horizontal_maps.range[0] == (0, 7, 2)
    assert horizontal_maps.range.tile_sizes[0] == 2
    assert horizontal_maps.params[1] == "__j"
    assert horizontal_maps.range[1] == (0, 4, 1)
    assert horizontal_maps.range.tile_sizes[1] == 1
