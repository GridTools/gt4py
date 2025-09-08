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

from gt4py.cartesian.gtc.dace.passes import PushVerticalMapDown

# Because "dace tests" filter by `requires_dace`, we still need to add the marker.
# This global variable adds the marker to all test functions in this module.
pytestmark = pytest.mark.requires_dace


def test_push_vertical_map_down():
    root = tn.ScheduleTreeRoot(name="tester", children=[])
    k_loop = tn.MapScope(
        node=nodes.MapEntry(
            map=nodes.Map("vertical map", ["__k"], subsets.Range.from_string("0:10"))
        ),
        children=[],
    )
    k_loop.parent = root
    ij_loop = tn.MapScope(
        node=nodes.MapEntry(
            map=nodes.Map("horizontal maps", ["__i", "__j"], subsets.Range.from_string("0:5,0:8"))
        ),
        children=[],
    )
    ij_loop.parent = k_loop
    k_loop.children.append(ij_loop)
    root.children.append(k_loop)

    flipper = PushVerticalMapDown()
    flipper.visit(root)

    assert len(root.children) == 1
    assert isinstance(root.children[0], tn.MapScope)
    assert root.children[0].node.map.params == ["__i", "__j"]

    assert len(root.children[0].children) == 1
    assert isinstance(root.children[0].children[0], tn.MapScope)
    assert root.children[0].children[0].node.map.params == ["__k"]
