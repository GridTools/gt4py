# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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

import numpy as np

from gt4py.storage.located_field import array_as_located_field


def test_located_field_1d():
    foo = array_as_located_field("foo")(np.zeros((1,)))

    foo[0] = 42

    assert foo.__gt_dims__[0] == "foo"
    assert foo[0] == 42


def test_located_field_2d():
    foo = array_as_located_field("foo", "bar")(np.zeros((1, 1), dtype=np.float64))

    foo[0, 0] = 42

    assert foo.__gt_dims__[0] == "foo"
    assert foo[0, 0] == 42
    assert foo.dtype == np.float64
