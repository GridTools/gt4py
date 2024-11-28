# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import ast

from gt4py.cartesian.frontend.gtscript_frontend import PYTHON_AST_VERSION, IRMaker
from gt4py.cartesian.frontend.nodes import BinaryOperator, BinOpExpr


def test_AugAssign():
    ir_maker = IRMaker(None, None, None, None, domain=None)
    aug_assign = ast.parse("a += 1", feature_version=PYTHON_AST_VERSION).body[0]

    _, result = ir_maker.visit_AugAssign(aug_assign)

    assert isinstance(result.value, BinOpExpr)
    assert result.value.op == BinaryOperator.ADD
