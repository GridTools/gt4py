# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2020, CSCS - Swiss National Supercomputing Center, ETH Zurich
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


from __future__ import annotations

import pytest


class TestFrozenList:
    def test_immutability(self):
        from eve.type_definitions import FrozenList

        fl = FrozenList([0, 1, 2, 3, 4, 5])

        with pytest.raises(TypeError, match="object does not support item assignment"):
            fl[2] = -2

    def test_instance_check(self):
        from eve.type_definitions import FrozenList

        assert isinstance(FrozenList([0, 1, 2, 3, 4, 5]), FrozenList)
        assert isinstance((), FrozenList)
        assert not isinstance([], FrozenList)


def test_sentinel():
    from eve.type_definitions import NOTHING

    values = [0, 1, 2, NOTHING, 4, 6]

    assert values.index(NOTHING) == 3
    assert values[values.index(NOTHING)] is NOTHING
