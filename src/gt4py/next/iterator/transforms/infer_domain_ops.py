# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.next import common
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im


class InferDomainOps(PreserveLocationVisitor, NodeTranslator):
    @classmethod
    def apply(cls, node: ir.Node):
        return cls().visit(node)

    def visit_FunCall(self, node: ir.FunCall) -> ir.FunCall:
        if isinstance(node, ir.FunCall) and cpm.is_call_to(
            node, ir.BINARY_MATH_COMPARISON_BUILTINS
        ):
            if isinstance(node.args[0], ir.AxisLiteral) and isinstance(node.args[1], ir.Literal):
                dim = common.Dimension(value=node.args[0].value, kind=common.DimensionKind.VERTICAL)
                value = int(node.args[1].value)
                reverse = False
            elif isinstance(node.args[0], ir.Literal) and isinstance(node.args[1], ir.AxisLiteral):
                dim = common.Dimension(value=node.args[1].value, kind=common.DimensionKind.VERTICAL)
                value = int(node.args[0].value)
                reverse = True
            else:
                raise ValueError(f"{node.args} need to be a 'ir.AxisLiteral' and an 'ir.Literal'.")

            match node.fun.id:
                case ir.SymbolRef("less"):
                    if reverse:
                        min = value + 1
                        max = "inf"
                    else:
                        min = "neg_inf"
                        max = value - 1
                case ir.SymbolRef("less_equal"):
                    if reverse:
                        min = value
                        max = "inf"
                    else:
                        min = "neg_inf"
                        max = value
                case ir.SymbolRef("greater"):
                    if reverse:
                        min = "neg_inf"
                        max = value - 1
                    else:
                        min = value + 1
                        max = "inf"
                case ir.SymbolRef("greater_equal"):
                    if reverse:
                        min = "neg_inf"
                        max = value
                    else:
                        min = value
                        max = "inf"
                case ir.SymbolRef("eq"):
                    min = max = value
                case ir.SymbolRef("not_eq"):
                    min1 = "neg_inf"
                    max1 = value - 1
                    min2 = value + 1
                    max2 = "inf"
                    return im.call("and_")(
                        im.domain(common.GridType.CARTESIAN, {dim: (min1, max1)}),
                        im.domain(common.GridType.CARTESIAN, {dim: (min2, max2)}),
                    )
                case _:
                    raise NotImplementedError

            return im.domain(common.GridType.CARTESIAN, {dim: (min, max)})

        return self.generic_visit(node)
