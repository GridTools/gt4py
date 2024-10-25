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
        if cpm.is_call_to(node, ir.BINARY_MATH_COMPARISON_BUILTINS):  # TODO: add tests
            arg1, arg2 = node.args
            fun = node.fun
            if isinstance(arg1, ir.AxisLiteral) and isinstance(arg2, ir.Literal):
                dim = common.Dimension(value=arg2.value, kind=common.DimensionKind.VERTICAL)
                value = int(arg2.value)
                reverse = False
            elif isinstance(arg1, ir.Literal) and isinstance(arg2, ir.AxisLiteral):
                dim = common.Dimension(value=arg2.value, kind=common.DimensionKind.VERTICAL)
                value = int(arg1.value)
                reverse = True
            else:
                raise ValueError(f"{node.args} need to be a 'ir.AxisLiteral' and an 'ir.Literal'.")
            assert isinstance(fun, ir.SymRef)
            min_: int | str
            max_: int | str
            match fun.id:
                case ir.SymbolRef("less"):
                    if reverse:
                        min_ = value + 1
                        max_ = "inf"
                    else:
                        min_ = "neg_inf"
                        max_ = value - 1
                case ir.SymbolRef("less_equal"):
                    if reverse:
                        min_ = value
                        max_ = "inf"
                    else:
                        min_ = "neg_inf"
                        max_ = value
                case ir.SymbolRef("greater"):
                    if reverse:
                        min_ = "neg_inf"
                        max_ = value - 1
                    else:
                        min_ = value + 1
                        max_ = "inf"
                case ir.SymbolRef("greater_equal"):
                    if reverse:
                        min_ = "neg_inf"
                        max_ = value
                    else:
                        min_ = value
                        max_ = "inf"
                case ir.SymbolRef("eq"):
                    min_ = max_ = value
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

            return im.domain(common.GridType.CARTESIAN, {dim: (min_, max_)})

        return self.generic_visit(node)
