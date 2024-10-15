# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest


class TestFrozenList:
    def test_immutability(self):
        from gt4py.eve.type_definitions import FrozenList

        fl = FrozenList([0, 1, 2, 3, 4, 5])

        with pytest.raises(TypeError, match="object does not support item assignment"):
            fl[2] = -2

    def test_instance_check(self):
        from gt4py.eve.type_definitions import FrozenList

        assert isinstance(FrozenList([0, 1, 2, 3, 4, 5]), FrozenList)
        assert isinstance((), FrozenList)
        assert not isinstance([], FrozenList)


def test_sentinel():
    from gt4py.eve.type_definitions import NOTHING

    values = [0, 1, 2, NOTHING, 4, 6]

    assert values.index(NOTHING) == 3
    assert values[values.index(NOTHING)] is NOTHING
