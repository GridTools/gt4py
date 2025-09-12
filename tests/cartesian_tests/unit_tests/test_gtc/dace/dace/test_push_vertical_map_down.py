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


def test_push_vertical_map_down_multiple_horizontal_maps():
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
            map=nodes.Map(
                "horizontal maps", ["__i_0", "__j_0"], subsets.Range.from_string("0:5,0:8")
            )
        ),
        children=[],
    )
    ij_loop.parent = k_loop
    k_loop.children.append(ij_loop)

    second_loop = tn.MapScope(
        node=nodes.MapEntry(
            map=nodes.Map(
                "horizontal maps", ["__i_1", "__j_1"], subsets.Range.from_string("0:5,0:8")
            )
        ),
        children=[],
    )
    second_loop.parent = k_loop
    k_loop.children.append(second_loop)

    root.children.append(k_loop)

    flipper = PushVerticalMapDown()
    flipper.visit(root)

    assert len(root.children) == 2

    for index, child in enumerate(root.children):
        assert isinstance(child, tn.MapScope)
        assert child.node.map.params == [f"__i_{index}", f"__j_{index}"]
        assert len(child.children) == 1
        assert isinstance(child.children[0], tn.MapScope)
        assert child.children[0].node.map.params == ["__k"]
