# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest

from gt4py.storage import layout as gt_layout


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
def test_gtcpu_kfirst_layout(dimensions, layout):
    make_layout_map = gt_layout.CPUKFirstLayout["layout_map"]
    assert make_layout_map(dimensions) == layout


@pytest.mark.parametrize(
    ["dimensions", "layout"],
    [
        ((), ()),
        (("I",), (0,)),
        (("J",), (0,)),
        (("I", "J"), (1, 0)),
        (("K",), (0,)),
        (("I", "K"), (1, 0)),
        (("J", "K"), (0, 1)),
        (("I", "J", "K"), (2, 0, 1)),
        (("0", "1"), (1, 0)),
        (("I", "0", "1"), (2, 1, 0)),
        (("J", "0", "1"), (2, 1, 0)),
        (("K", "0", "1"), (2, 1, 0)),
        (("I", "J", "0", "1"), (3, 2, 1, 0)),
        (("I", "K", "0", "1"), (3, 2, 1, 0)),
        (("J", "K", "0", "1"), (2, 3, 1, 0)),
        (("I", "J", "K", "0", "1"), (4, 2, 3, 1, 0)),
        (("0",), (0,)),
        (("I", "0"), (1, 0)),
        (("J", "0"), (1, 0)),
        (("K", "0"), (1, 0)),
        (("I", "J", "0"), (2, 1, 0)),
        (("I", "K", "0"), (2, 1, 0)),
        (("J", "K", "0"), (1, 2, 0)),
        (("I", "J", "K", "0"), (3, 1, 2, 0)),
        (("1",), (0,)),
        (("I", "1"), (1, 0)),
        (("J", "1"), (1, 0)),
        (("K", "1"), (1, 0)),
        (("I", "J", "1"), (2, 1, 0)),
        (("I", "K", "1"), (2, 1, 0)),
        (("J", "K", "1"), (1, 2, 0)),
        (("I", "J", "K", "1"), (3, 1, 2, 0)),
    ],
)
def test_gtcpu_ifirst_layout(dimensions, layout):
    make_layout_map = gt_layout.CPUIFirstLayout["layout_map"]
    assert make_layout_map(dimensions) == layout


@pytest.mark.parametrize(
    ["dimensions", "layout"],
    [
        ((), ()),
        (("I",), (0,)),
        (("J",), (0,)),
        (("I", "J"), (1, 0)),
        (("K",), (0,)),
        (("I", "K"), (1, 0)),
        (("J", "K"), (1, 0)),
        (("I", "J", "K"), (2, 1, 0)),
        (("0", "1"), (1, 0)),
        (("I", "0", "1"), (2, 1, 0)),
        (("J", "0", "1"), (2, 1, 0)),
        (("K", "0", "1"), (2, 1, 0)),
        (("I", "J", "0", "1"), (3, 2, 1, 0)),
        (("I", "K", "0", "1"), (3, 2, 1, 0)),
        (("J", "K", "0", "1"), (3, 2, 1, 0)),
        (("I", "J", "K", "0", "1"), (4, 3, 2, 1, 0)),
        (("0",), (0,)),
        (("I", "0"), (1, 0)),
        (("J", "0"), (1, 0)),
        (("K", "0"), (1, 0)),
        (("I", "J", "0"), (2, 1, 0)),
        (("I", "K", "0"), (2, 1, 0)),
        (("J", "K", "0"), (2, 1, 0)),
        (("I", "J", "K", "0"), (3, 2, 1, 0)),
        (("1",), (0,)),
        (("I", "1"), (1, 0)),
        (("J", "1"), (1, 0)),
        (("K", "1"), (1, 0)),
        (("I", "J", "1"), (2, 1, 0)),
        (("I", "K", "1"), (2, 1, 0)),
        (("J", "K", "1"), (2, 1, 0)),
        (("I", "J", "K", "1"), (3, 2, 1, 0)),
    ],
)
def test_gpu_layout(dimensions, layout):
    make_layout_map = gt_layout.GPULayout["layout_map"]
    assert make_layout_map(dimensions) == layout
