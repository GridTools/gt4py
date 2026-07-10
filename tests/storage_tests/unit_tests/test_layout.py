# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from gt4py.storage.cartesian import layout as gt_layout


@pytest.mark.parametrize(
    ["dimensions", "layout"],
    [
        ((), ()),
        (("I",), (0,)),
        (("J",), (0,)),
        (("I", "J"), (0, 1)),
        (("K",), (0,)),
        (("I", "K"), (0, 1)),
        (("J", "K"), (0, 1)),
        (("I", "J", "K"), (0, 1, 2)),
        (("0", "1"), (0, 1)),
        (("I", "0", "1"), (2, 0, 1)),
        (("J", "0", "1"), (2, 0, 1)),
        (("K", "0", "1"), (2, 0, 1)),
        (("I", "J", "0", "1"), (2, 3, 0, 1)),
        (("I", "K", "0", "1"), (2, 3, 0, 1)),
        (("J", "K", "0", "1"), (2, 3, 0, 1)),
        (("I", "J", "K", "0", "1"), (2, 3, 4, 0, 1)),
        (("0",), (0,)),
        (("I", "0"), (1, 0)),
        (("J", "0"), (1, 0)),
        (("K", "0"), (1, 0)),
        (("I", "J", "0"), (1, 2, 0)),
        (("I", "K", "0"), (1, 2, 0)),
        (("J", "K", "0"), (1, 2, 0)),
        (("I", "J", "K", "0"), (1, 2, 3, 0)),
        (("1",), (0,)),
        (("I", "1"), (1, 0)),
        (("J", "1"), (1, 0)),
        (("K", "1"), (1, 0)),
        (("I", "J", "1"), (1, 2, 0)),
        (("I", "K", "1"), (1, 2, 0)),
        (("J", "K", "1"), (1, 2, 0)),
        (("I", "J", "K", "1"), (1, 2, 3, 0)),
    ],
)
def test_IJK_layout(dimensions, layout):
    layout_maker = gt_layout.layout_maker_factory((0, 1, 2))
    assert layout_maker(dimensions) == layout


@pytest.mark.parametrize(
    ["dimensions", "layout"],
    [
        # No cartesian dimensions
        ((), ()),
        (("0",), (0,)),
        (("1",), (0,)),
        (("0", "1"), (0, 1)),
        # One cartesian dimension
        (("I",), (0,)),
        (("I", "0"), (1, 0)),
        (("I", "0", "1"), (2, 0, 1)),
        (("J",), (0,)),
        (("J", "0"), (1, 0)),
        (("J", "0", "1"), (2, 0, 1)),
        (("K",), (0,)),
        (("K", "0"), (1, 0)),
        (("K", "0", "1"), (2, 0, 1)),
        # Two cartesian dimensions
        (("I", "J"), (1, 0)),
        (("I", "J", "0"), (2, 1, 0)),
        (("I", "J", "0", "1"), (3, 2, 0, 1)),
        (("I", "K"), (1, 0)),
        (("I", "K", "0"), (2, 1, 0)),
        (("I", "K", "0", "1"), (3, 2, 0, 1)),
        (("J", "K"), (0, 1)),
        (("J", "K", "0"), (1, 2, 0)),
        (("J", "K", "0", "1"), (2, 3, 0, 1)),
        # Three cartesian dimensions
        (("I", "J", "K"), (2, 0, 1)),
        (("I", "J", "K", "0"), (3, 1, 2, 0)),
        (("I", "J", "K", "0", "1"), (4, 2, 3, 0, 1)),
    ],
)
def test_JKI_layout(dimensions, layout):
    layout_maker = gt_layout.layout_maker_factory((2, 0, 1))
    assert layout_maker(dimensions) == layout


@pytest.mark.parametrize(
    ["dimensions", "layout"],
    [
        # No cartesian dimensions
        ((), ()),
        (("0",), (0,)),
        (("1",), (0,)),
        (("0", "1"), (0, 1)),
        # One cartesian dimension
        (("I",), (0,)),
        (("I", "0"), (1, 0)),
        (("I", "0", "1"), (2, 0, 1)),
        (("J",), (0,)),
        (("J", "0"), (1, 0)),
        (("J", "0", "1"), (2, 0, 1)),
        (("K",), (0,)),
        (("K", "0"), (1, 0)),
        (("K", "0", "1"), (2, 0, 1)),
        # Two cartesian dimensions
        (("I", "J"), (1, 0)),
        (("I", "J", "0"), (2, 1, 0)),
        (("I", "J", "0", "1"), (3, 2, 0, 1)),
        (("I", "K"), (1, 0)),
        (("I", "K", "0"), (2, 1, 0)),
        (("I", "K", "0", "1"), (3, 2, 0, 1)),
        (("J", "K"), (1, 0)),
        (("J", "K", "0"), (2, 1, 0)),
        (("J", "K", "0", "1"), (3, 2, 0, 1)),
        # Three cartesian dimensions
        (("I", "J", "K"), (2, 1, 0)),
        (("I", "J", "K", "0"), (3, 2, 1, 0)),
        (("I", "J", "K", "0", "1"), (4, 3, 2, 0, 1)),
    ],
)
def test_KJI_layout(dimensions, layout):
    layout_maker = gt_layout.layout_maker_factory((2, 1, 0))
    assert layout_maker(dimensions) == layout
