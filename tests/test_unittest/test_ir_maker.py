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

import ast

from gt4py.frontend.gtscript_frontend import IRMaker
from gt4py.frontend.nodes import BinaryOperator, BinOpExpr


def test_AugAssign():
    ir_maker = IRMaker(None, None, None, domain=None)
    aug_assign = ast.parse("a += 1", feature_version=(3, 9)).body[0]

    _, result = ir_maker.visit_AugAssign(aug_assign)

    assert isinstance(result.value, BinOpExpr)
    assert result.value.op == BinaryOperator.ADD
