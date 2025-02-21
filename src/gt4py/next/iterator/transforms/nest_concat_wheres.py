# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im


class NestConcatWheres(PreserveLocationVisitor, NodeTranslator):
    @classmethod
    def apply(cls, node: ir.Node):
        return cls().visit(node)

    def visit_FunCall(self, node: ir.FunCall) -> ir.FunCall:
        node = self.generic_visit(node)

        if cpm.is_call_to(node, "concat_where"):
            cond_expr, field_a, field_b = node.args
            # TODO: don't duplicate exprs here
            if cpm.is_call_to(cond_expr, ("and_")):
                conds = cond_expr.args
                return im.concat_where(
                    conds[0], im.concat_where(conds[1], field_a, field_b), field_b
                )
            if cpm.is_call_to(cond_expr, ("or_")):
                conds = cond_expr.args
                return im.concat_where(
                    conds[0], field_a, im.concat_where(conds[1], field_a, field_b)
                )
            if cpm.is_call_to(cond_expr, ("eq")):
                cond1 = im.less(cond_expr.args[0], cond_expr.args[1])
                cond2 = im.greater(cond_expr.args[0], cond_expr.args[1])
                return im.concat_where(cond1, field_b, im.concat_where(cond2, field_b, field_a))

        return node
