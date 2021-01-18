# -*- coding: utf-8 -*-
import ast

from gt4py.frontend.gtscript_frontend import IRMaker
from gt4py.ir.nodes import BinaryOperator, BinOpExpr


def test_AugAssign():
    ir_maker = IRMaker(None, None, None, domain=None, extra_temp_decls=None)
    aug_assign = ast.parse("a += 1").body[0]

    _, result = ir_maker.visit_AugAssign(aug_assign)

    assert isinstance(result.value, BinOpExpr)
    assert result.value.op == BinaryOperator.ADD
