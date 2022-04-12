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

from gt4py.backend.gtc_backend.common import (
    make_cuda_layout_map,
    make_mc_layout_map,
    make_x86_layout_map,
)


def test_x86_layout():
    assert make_x86_layout_map(()) == ()
    assert make_x86_layout_map((0,)) == (None,)
    assert make_x86_layout_map((1,)) == (0,)
    assert make_x86_layout_map((0, 0)) == (None, None)
    assert make_x86_layout_map((1, 0)) == (0, None)
    assert make_x86_layout_map((0, 1)) == (None, 0)
    assert make_x86_layout_map((1, 1)) == (0, 1)
    assert make_x86_layout_map((0, 0, 0)) == (None, None, None)
    assert make_x86_layout_map((1, 0, 0)) == (0, None, None)
    assert make_x86_layout_map((0, 1, 0)) == (None, 0, None)
    assert make_x86_layout_map((0, 0, 1)) == (None, None, 0)
    assert make_x86_layout_map((1, 1, 0)) == (0, 1, None)
    assert make_x86_layout_map((1, 0, 1)) == (0, None, 1)
    assert make_x86_layout_map((0, 1, 1)) == (None, 0, 1)
    assert make_x86_layout_map((1, 1, 1)) == (0, 1, 2)
    assert make_x86_layout_map((0, 0, 0, 1, 1)) == (None, None, None, 0, 1)
    assert make_x86_layout_map((1, 0, 0, 1, 1)) == (2, None, None, 0, 1)
    assert make_x86_layout_map((0, 1, 0, 1, 1)) == (None, 2, None, 0, 1)
    assert make_x86_layout_map((0, 0, 1, 1, 1)) == (None, None, 2, 0, 1)
    assert make_x86_layout_map((1, 1, 0, 1, 1)) == (2, 3, None, 0, 1)
    assert make_x86_layout_map((1, 0, 1, 1, 1)) == (2, None, 3, 0, 1)
    assert make_x86_layout_map((0, 1, 1, 1, 1)) == (None, 2, 3, 0, 1)
    assert make_x86_layout_map((1, 1, 1, 1, 1)) == (2, 3, 4, 0, 1)
    assert make_x86_layout_map((0, 0, 0, 1, 0)) == (None, None, None, 0, None)
    assert make_x86_layout_map((1, 0, 0, 1, 0)) == (1, None, None, 0, None)
    assert make_x86_layout_map((0, 1, 0, 1, 0)) == (None, 1, None, 0, None)
    assert make_x86_layout_map((0, 0, 1, 1, 0)) == (None, None, 1, 0, None)
    assert make_x86_layout_map((1, 1, 0, 1, 0)) == (1, 2, None, 0, None)
    assert make_x86_layout_map((1, 0, 1, 1, 0)) == (1, None, 2, 0, None)
    assert make_x86_layout_map((0, 1, 1, 1, 0)) == (None, 1, 2, 0, None)
    assert make_x86_layout_map((1, 1, 1, 1, 0)) == (1, 2, 3, 0, None)
    assert make_x86_layout_map((0, 0, 0, 0, 1)) == (None, None, None, None, 0)
    assert make_x86_layout_map((1, 0, 0, 0, 1)) == (1, None, None, None, 0)
    assert make_x86_layout_map((0, 1, 0, 0, 1)) == (None, 1, None, None, 0)
    assert make_x86_layout_map((0, 0, 1, 0, 1)) == (None, None, 1, None, 0)
    assert make_x86_layout_map((1, 1, 0, 0, 1)) == (1, 2, None, None, 0)
    assert make_x86_layout_map((1, 0, 1, 0, 1)) == (1, None, 2, None, 0)
    assert make_x86_layout_map((0, 1, 1, 0, 1)) == (None, 1, 2, None, 0)
    assert make_x86_layout_map((1, 1, 1, 0, 1)) == (1, 2, 3, None, 0)
    assert make_x86_layout_map((0, 0, 0, 0, 0)) == (None, None, None, None, None)
    assert make_x86_layout_map((1, 0, 0, 0, 0)) == (0, None, None, None, None)
    assert make_x86_layout_map((0, 1, 0, 0, 0)) == (None, 0, None, None, None)
    assert make_x86_layout_map((0, 0, 1, 0, 0)) == (None, None, 0, None, None)
    assert make_x86_layout_map((1, 1, 0, 0, 0)) == (0, 1, None, None, None)
    assert make_x86_layout_map((1, 0, 1, 0, 0)) == (0, None, 1, None, None)
    assert make_x86_layout_map((0, 1, 1, 0, 0)) == (None, 0, 1, None, None)
    assert make_x86_layout_map((1, 1, 1, 0, 0)) == (0, 1, 2, None, None)


def test_mc_layout():
    assert make_mc_layout_map(()) == ()
    assert make_mc_layout_map((0,)) == (None,)
    assert make_mc_layout_map((1,)) == (0,)
    assert make_mc_layout_map((0, 0)) == (None, None)
    assert make_mc_layout_map((1, 0)) == (0, None)
    assert make_mc_layout_map((0, 1)) == (None, 0)
    assert make_mc_layout_map((1, 1)) == (1, 0)
    assert make_mc_layout_map((0, 0, 0)) == (None, None, None)
    assert make_mc_layout_map((1, 0, 0)) == (0, None, None)
    assert make_mc_layout_map((0, 1, 0)) == (None, 0, None)
    assert make_mc_layout_map((0, 0, 1)) == (None, None, 0)
    assert make_mc_layout_map((1, 1, 0)) == (1, 0, None)
    assert make_mc_layout_map((1, 0, 1)) == (1, None, 0)
    assert make_mc_layout_map((0, 1, 1)) == (None, 0, 1)
    assert make_mc_layout_map((1, 1, 1)) == (2, 0, 1)
    assert make_mc_layout_map((0, 0, 0, 1, 1)) == (None, None, None, 1, 0)
    assert make_mc_layout_map((1, 0, 0, 1, 1)) == (2, None, None, 1, 0)
    assert make_mc_layout_map((0, 1, 0, 1, 1)) == (None, 2, None, 1, 0)
    assert make_mc_layout_map((0, 0, 1, 1, 1)) == (None, None, 2, 1, 0)
    assert make_mc_layout_map((1, 1, 0, 1, 1)) == (3, 2, None, 1, 0)
    assert make_mc_layout_map((1, 0, 1, 1, 1)) == (3, None, 2, 1, 0)
    assert make_mc_layout_map((0, 1, 1, 1, 1)) == (None, 2, 3, 1, 0)
    assert make_mc_layout_map((1, 1, 1, 1, 1)) == (4, 2, 3, 1, 0)
    assert make_mc_layout_map((0, 0, 0, 1, 0)) == (None, None, None, 0, None)
    assert make_mc_layout_map((1, 0, 0, 1, 0)) == (1, None, None, 0, None)
    assert make_mc_layout_map((0, 1, 0, 1, 0)) == (None, 1, None, 0, None)
    assert make_mc_layout_map((0, 0, 1, 1, 0)) == (None, None, 1, 0, None)
    assert make_mc_layout_map((1, 1, 0, 1, 0)) == (2, 1, None, 0, None)
    assert make_mc_layout_map((1, 0, 1, 1, 0)) == (2, None, 1, 0, None)
    assert make_mc_layout_map((0, 1, 1, 1, 0)) == (None, 1, 2, 0, None)
    assert make_mc_layout_map((1, 1, 1, 1, 0)) == (3, 1, 2, 0, None)
    assert make_mc_layout_map((0, 0, 0, 0, 1)) == (None, None, None, None, 0)
    assert make_mc_layout_map((1, 0, 0, 0, 1)) == (1, None, None, None, 0)
    assert make_mc_layout_map((0, 1, 0, 0, 1)) == (None, 1, None, None, 0)
    assert make_mc_layout_map((0, 0, 1, 0, 1)) == (None, None, 1, None, 0)
    assert make_mc_layout_map((1, 1, 0, 0, 1)) == (2, 1, None, None, 0)
    assert make_mc_layout_map((1, 0, 1, 0, 1)) == (2, None, 1, None, 0)
    assert make_mc_layout_map((0, 1, 1, 0, 1)) == (None, 1, 2, None, 0)
    assert make_mc_layout_map((1, 1, 1, 0, 1)) == (3, 1, 2, None, 0)
    assert make_mc_layout_map((0, 0, 0, 0, 0)) == (None, None, None, None, None)
    assert make_mc_layout_map((1, 0, 0, 0, 0)) == (0, None, None, None, None)
    assert make_mc_layout_map((0, 1, 0, 0, 0)) == (None, 0, None, None, None)
    assert make_mc_layout_map((0, 0, 1, 0, 0)) == (None, None, 0, None, None)
    assert make_mc_layout_map((1, 1, 0, 0, 0)) == (1, 0, None, None, None)
    assert make_mc_layout_map((1, 0, 1, 0, 0)) == (1, None, 0, None, None)
    assert make_mc_layout_map((0, 1, 1, 0, 0)) == (None, 0, 1, None, None)
    assert make_mc_layout_map((1, 1, 1, 0, 0)) == (2, 0, 1, None, None)


def test_cuda_layout():
    assert make_cuda_layout_map(()) == ()
    assert make_cuda_layout_map((0,)) == (None,)
    assert make_cuda_layout_map((1,)) == (0,)
    assert make_cuda_layout_map((0, 0)) == (None, None)
    assert make_cuda_layout_map((1, 0)) == (0, None)
    assert make_cuda_layout_map((0, 1)) == (None, 0)
    assert make_cuda_layout_map((1, 1)) == (1, 0)
    assert make_cuda_layout_map((0, 0, 0)) == (None, None, None)
    assert make_cuda_layout_map((1, 0, 0)) == (0, None, None)
    assert make_cuda_layout_map((0, 1, 0)) == (None, 0, None)
    assert make_cuda_layout_map((0, 0, 1)) == (None, None, 0)
    assert make_cuda_layout_map((1, 1, 0)) == (1, 0, None)
    assert make_cuda_layout_map((1, 0, 1)) == (1, None, 0)
    assert make_cuda_layout_map((0, 1, 1)) == (None, 1, 0)
    assert make_cuda_layout_map((1, 1, 1)) == (2, 1, 0)
    assert make_cuda_layout_map((0, 0, 0, 1, 1)) == (None, None, None, 1, 0)
    assert make_cuda_layout_map((1, 0, 0, 1, 1)) == (2, None, None, 1, 0)
    assert make_cuda_layout_map((0, 1, 0, 1, 1)) == (None, 2, None, 1, 0)
    assert make_cuda_layout_map((0, 0, 1, 1, 1)) == (None, None, 2, 1, 0)
    assert make_cuda_layout_map((1, 1, 0, 1, 1)) == (3, 2, None, 1, 0)
    assert make_cuda_layout_map((1, 0, 1, 1, 1)) == (3, None, 2, 1, 0)
    assert make_cuda_layout_map((0, 1, 1, 1, 1)) == (None, 3, 2, 1, 0)
    assert make_cuda_layout_map((1, 1, 1, 1, 1)) == (4, 3, 2, 1, 0)
    assert make_cuda_layout_map((0, 0, 0, 1, 0)) == (None, None, None, 0, None)
    assert make_cuda_layout_map((1, 0, 0, 1, 0)) == (1, None, None, 0, None)
    assert make_cuda_layout_map((0, 1, 0, 1, 0)) == (None, 1, None, 0, None)
    assert make_cuda_layout_map((0, 0, 1, 1, 0)) == (None, None, 1, 0, None)
    assert make_cuda_layout_map((1, 1, 0, 1, 0)) == (2, 1, None, 0, None)
    assert make_cuda_layout_map((1, 0, 1, 1, 0)) == (2, None, 1, 0, None)
    assert make_cuda_layout_map((0, 1, 1, 1, 0)) == (None, 2, 1, 0, None)
    assert make_cuda_layout_map((1, 1, 1, 1, 0)) == (3, 2, 1, 0, None)
    assert make_cuda_layout_map((0, 0, 0, 0, 1)) == (None, None, None, None, 0)
    assert make_cuda_layout_map((1, 0, 0, 0, 1)) == (1, None, None, None, 0)
    assert make_cuda_layout_map((0, 1, 0, 0, 1)) == (None, 1, None, None, 0)
    assert make_cuda_layout_map((0, 0, 1, 0, 1)) == (None, None, 1, None, 0)
    assert make_cuda_layout_map((1, 1, 0, 0, 1)) == (2, 1, None, None, 0)
    assert make_cuda_layout_map((1, 0, 1, 0, 1)) == (2, None, 1, None, 0)
    assert make_cuda_layout_map((0, 1, 1, 0, 1)) == (None, 2, 1, None, 0)
    assert make_cuda_layout_map((1, 1, 1, 0, 1)) == (3, 2, 1, None, 0)
    assert make_cuda_layout_map((0, 0, 0, 0, 0)) == (None, None, None, None, None)
    assert make_cuda_layout_map((1, 0, 0, 0, 0)) == (0, None, None, None, None)
    assert make_cuda_layout_map((0, 1, 0, 0, 0)) == (None, 0, None, None, None)
    assert make_cuda_layout_map((0, 0, 1, 0, 0)) == (None, None, 0, None, None)
    assert make_cuda_layout_map((1, 1, 0, 0, 0)) == (1, 0, None, None, None)
    assert make_cuda_layout_map((1, 0, 1, 0, 0)) == (1, None, 0, None, None)
    assert make_cuda_layout_map((0, 1, 1, 0, 0)) == (None, 1, 0, None, None)
    assert make_cuda_layout_map((1, 1, 1, 0, 0)) == (2, 1, 0, None, None)
