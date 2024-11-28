# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from gt4py.cartesian.gtc.gtcpp import gtcpp_codegen
from gt4py.cartesian.gtc.gtcpp.gtcpp import GTLevel


@pytest.mark.parametrize("root,expected", [(GTLevel(splitter=0, offset=5), 6)])
def test_offset_limit(root, expected):
    assert gtcpp_codegen._offset_limit(root) == expected
