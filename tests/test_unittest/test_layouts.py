# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2022, ETH Zurich
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

from gt4py.backend.gtc_common import make_cuda_layout_map, make_mc_layout_map, make_x86_layout_map


def test_x86_layout():
    assert make_x86_layout_map(()) == ()
    assert make_x86_layout_map(("I",)) == (0,)
    assert make_x86_layout_map(("J",)) == (0,)
    assert make_x86_layout_map(("I", "J")) == (0, 1)
    assert make_x86_layout_map(("K",)) == (0,)
    assert make_x86_layout_map(("I", "K")) == (0, 1)
    assert make_x86_layout_map(("J", "K")) == (0, 1)
    assert make_x86_layout_map(("I", "J", "K")) == (0, 1, 2)
    assert make_x86_layout_map(("0", "1")) == (0, 1)
    assert make_x86_layout_map(("I", "0", "1")) == (2, 0, 1)
    assert make_x86_layout_map(("J", "0", "1")) == (2, 0, 1)
    assert make_x86_layout_map(("K", "0", "1")) == (2, 0, 1)
    assert make_x86_layout_map(("I", "J", "0", "1")) == (2, 3, 0, 1)
    assert make_x86_layout_map(("I", "K", "0", "1")) == (2, 3, 0, 1)
    assert make_x86_layout_map(("J", "K", "0", "1")) == (2, 3, 0, 1)
    assert make_x86_layout_map(("I", "J", "K", "0", "1")) == (2, 3, 4, 0, 1)
    assert make_x86_layout_map(("0",)) == (0,)
    assert make_x86_layout_map(("I", "0")) == (1, 0)
    assert make_x86_layout_map(("J", "0")) == (1, 0)
    assert make_x86_layout_map(("K", "0")) == (1, 0)
    assert make_x86_layout_map(("I", "J", "0")) == (1, 2, 0)
    assert make_x86_layout_map(("I", "K", "0")) == (1, 2, 0)
    assert make_x86_layout_map(("J", "K", "0")) == (1, 2, 0)
    assert make_x86_layout_map(("I", "J", "K", "0")) == (1, 2, 3, 0)
    assert make_x86_layout_map(("1",)) == (0,)
    assert make_x86_layout_map(("I", "1")) == (1, 0)
    assert make_x86_layout_map(("J", "1")) == (1, 0)
    assert make_x86_layout_map(("K", "1")) == (1, 0)
    assert make_x86_layout_map(("I", "J", "1")) == (1, 2, 0)
    assert make_x86_layout_map(("I", "K", "1")) == (1, 2, 0)
    assert make_x86_layout_map(("J", "K", "1")) == (1, 2, 0)
    assert make_x86_layout_map(("I", "J", "K", "1")) == (1, 2, 3, 0)


def test_mc_layout():
    assert make_mc_layout_map(()) == ()
    assert make_mc_layout_map(("I",)) == (0,)
    assert make_mc_layout_map(("J",)) == (0,)
    assert make_mc_layout_map(("I", "J")) == (1, 0)
    assert make_mc_layout_map(("K",)) == (0,)
    assert make_mc_layout_map(("I", "K")) == (1, 0)
    assert make_mc_layout_map(("J", "K")) == (0, 1)
    assert make_mc_layout_map(("I", "J", "K")) == (2, 0, 1)
    assert make_mc_layout_map(("0", "1")) == (1, 0)
    assert make_mc_layout_map(("I", "0", "1")) == (2, 1, 0)
    assert make_mc_layout_map(("J", "0", "1")) == (2, 1, 0)
    assert make_mc_layout_map(("K", "0", "1")) == (2, 1, 0)
    assert make_mc_layout_map(("I", "J", "0", "1")) == (3, 2, 1, 0)
    assert make_mc_layout_map(("I", "K", "0", "1")) == (3, 2, 1, 0)
    assert make_mc_layout_map(("J", "K", "0", "1")) == (2, 3, 1, 0)
    assert make_mc_layout_map(("I", "J", "K", "0", "1")) == (4, 2, 3, 1, 0)
    assert make_mc_layout_map(("0",)) == (0,)
    assert make_mc_layout_map(("I", "0")) == (1, 0)
    assert make_mc_layout_map(("J", "0")) == (1, 0)
    assert make_mc_layout_map(("K", "0")) == (1, 0)
    assert make_mc_layout_map(("I", "J", "0")) == (2, 1, 0)
    assert make_mc_layout_map(("I", "K", "0")) == (2, 1, 0)
    assert make_mc_layout_map(("J", "K", "0")) == (1, 2, 0)
    assert make_mc_layout_map(("I", "J", "K", "0")) == (3, 1, 2, 0)
    assert make_mc_layout_map(("1",)) == (0,)
    assert make_mc_layout_map(("I", "1")) == (1, 0)
    assert make_mc_layout_map(("J", "1")) == (1, 0)
    assert make_mc_layout_map(("K", "1")) == (1, 0)
    assert make_mc_layout_map(("I", "J", "1")) == (2, 1, 0)
    assert make_mc_layout_map(("I", "K", "1")) == (2, 1, 0)
    assert make_mc_layout_map(("J", "K", "1")) == (1, 2, 0)
    assert make_mc_layout_map(("I", "J", "K", "1")) == (3, 1, 2, 0)


def test_cuda_layout():
    assert make_cuda_layout_map(()) == ()
    assert make_cuda_layout_map(("I",)) == (0,)
    assert make_cuda_layout_map(("J",)) == (0,)
    assert make_cuda_layout_map(("I", "J")) == (1, 0)
    assert make_cuda_layout_map(("K",)) == (0,)
    assert make_cuda_layout_map(("I", "K")) == (1, 0)
    assert make_cuda_layout_map(("J", "K")) == (1, 0)
    assert make_cuda_layout_map(("I", "J", "K")) == (2, 1, 0)
    assert make_cuda_layout_map(("0", "1")) == (1, 0)
    assert make_cuda_layout_map(("I", "0", "1")) == (2, 1, 0)
    assert make_cuda_layout_map(("J", "0", "1")) == (2, 1, 0)
    assert make_cuda_layout_map(("K", "0", "1")) == (2, 1, 0)
    assert make_cuda_layout_map(("I", "J", "0", "1")) == (3, 2, 1, 0)
    assert make_cuda_layout_map(("I", "K", "0", "1")) == (3, 2, 1, 0)
    assert make_cuda_layout_map(("J", "K", "0", "1")) == (3, 2, 1, 0)
    assert make_cuda_layout_map(("I", "J", "K", "0", "1")) == (4, 3, 2, 1, 0)
    assert make_cuda_layout_map(("0",)) == (0,)
    assert make_cuda_layout_map(("I", "0")) == (1, 0)
    assert make_cuda_layout_map(("J", "0")) == (1, 0)
    assert make_cuda_layout_map(("K", "0")) == (1, 0)
    assert make_cuda_layout_map(("I", "J", "0")) == (2, 1, 0)
    assert make_cuda_layout_map(("I", "K", "0")) == (2, 1, 0)
    assert make_cuda_layout_map(("J", "K", "0")) == (2, 1, 0)
    assert make_cuda_layout_map(("I", "J", "K", "0")) == (3, 2, 1, 0)
    assert make_cuda_layout_map(("1",)) == (0,)
    assert make_cuda_layout_map(("I", "1")) == (1, 0)
    assert make_cuda_layout_map(("J", "1")) == (1, 0)
    assert make_cuda_layout_map(("K", "1")) == (1, 0)
    assert make_cuda_layout_map(("I", "J", "1")) == (2, 1, 0)
    assert make_cuda_layout_map(("I", "K", "1")) == (2, 1, 0)
    assert make_cuda_layout_map(("J", "K", "1")) == (2, 1, 0)
    assert make_cuda_layout_map(("I", "J", "K", "1")) == (3, 2, 1, 0)
