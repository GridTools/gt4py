# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest

from gt4py.backend.gt_backends import gtcuda_layout, gtmc_layout, gtx86_layout


@pytest.mark.parametrize(
    ["dims", "layout"],
    [
        ((), ()),
        (("I",), (0,)),
        (("J",), (0,)),
        (("K",), (0,)),
        (("IJ"), (0, 1)),
        (("IK"), (0, 1)),
        (("JK"), (0, 1)),
        (("01"), (0, 1)),
        (("IJK"), (0, 1, 2)),
        (("I01"), (2, 0, 1)),
        (["J", "K", "0", "1"], (2, 3, 0, 1)),
        (["I", "J", "K", "0", "1"], (2, 3, 4, 0, 1)),
        ("JI", (1, 0)),
        ("KI", (1, 0)),
        ("KJ", (1, 0)),
        ("JKI", (1, 2, 0)),
        ("10", (1, 0)),
        ("0I1", (0, 2, 1)),
        (["0", "1", "J", "K"], (0, 1, 2, 3)),
    ],
)
def test_gtx86_layout(dims, layout):
    assert gtx86_layout(dims) == layout


@pytest.mark.parametrize(
    ["dims", "layout"],
    [
        ((), ()),
        (("I",), (0,)),
        (("J",), (0,)),
        (("K",), (0,)),
        ("IJ", (1, 0)),
        ("IK", (1, 0)),
        ("JK", (0, 1)),
        ("01", (1, 0)),
        ("IJK", (2, 0, 1)),
        ("I01", (2, 1, 0)),
        (["J", "K", "0", "1"], (2, 3, 1, 0)),
        (["I", "J", "K", "0", "1"], (4, 2, 3, 1, 0)),
        ("JI", (0, 1)),
        ("KI", (0, 1)),
        ("KJ", (1, 0)),
        ("JKI", (0, 1, 2)),
        ("10", (0, 1)),
        ("0I1", (1, 2, 0)),
        (["0", "1", "J", "K"], (1, 0, 2, 3)),
    ],
)
def test_mc_layout(dims, layout):
    assert gtmc_layout(dims) == layout


@pytest.mark.parametrize(
    ["dims", "layout"],
    [
        ((), ()),
        (("I",), (0,)),
        (("J",), (0,)),
        (("K",), (0,)),
        (("IJ"), (1, 0)),
        (("IK"), (1, 0)),
        (("JK"), (1, 0)),
        (("01"), (1, 0)),
        (("IJK"), (2, 1, 0)),
        (("I01"), (2, 1, 0)),
        (["J", "K", "0", "1"], (3, 2, 1, 0)),
        (["I", "J", "K", "0", "1"], (4, 3, 2, 1, 0)),
        (("JI"), (0, 1)),
        (("KI"), (0, 1)),
        (("KJ"), (0, 1)),
        (("10"), (0, 1)),
        (("0I1"), (1, 2, 0)),
        (["0", "1", "J", "K"], (1, 0, 3, 2)),
    ],
)
def test_gtcuda_layout(dims, layout):
    assert gtcuda_layout(dims) == layout
