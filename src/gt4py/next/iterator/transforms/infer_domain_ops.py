# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.next import common
from gt4py.next.iterator import builtins, ir
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    domain_utils,
    ir_makers as im,
)
from gt4py.next.iterator.transforms.constant_folding import ConstantFolding


class InferDomainOps(PreserveLocationVisitor, NodeTranslator):
    @classmethod
    def apply(cls, node: ir.Node):
        return cls().visit(node)

    def visit_FunCall(self, node: ir.FunCall) -> ir.Node:
        node = self.generic_visit(node)
        if (
            cpm.is_call_to(node, builtins.BINARY_MATH_COMPARISON_BUILTINS)
            and any(isinstance(arg, ir.AxisLiteral) for arg in node.args)
            and any(isinstance(arg, (ir.Literal, ir.SymRef)) for arg in node.args)
        ):  # TODO: add tests
            arg1, arg2 = node.args
            fun = node.fun
            if isinstance(arg1, ir.AxisLiteral):
                dim = common.Dimension(value=arg1.value, kind=arg1.kind)
                reverse = False
                if isinstance(arg2, ir.Literal):
                    value = int(arg2.value)
                elif isinstance(arg2, ir.SymRef):
                    value = arg2
            elif isinstance(arg2, ir.AxisLiteral):
                dim = common.Dimension(value=arg2.value, kind=arg2.kind)
                reverse = True
                if isinstance(arg1, ir.Literal):
                    value = int(arg1.value)
                elif isinstance(arg1, ir.SymRef):
                    value = arg1
            else:
                raise ValueError(f"{node.args} need to be a 'ir.AxisLiteral' and an 'ir.Literal'.")
            assert isinstance(fun, ir.SymRef)
            min_: int | ir.NegInfinityLiteral
            max_: int | ir.InfinityLiteral
            match fun.id:
                case ir.SymbolRef("less"):
                    if reverse:
                        min_ = im.plus(value, 1)
                        max_ = ir.InfinityLiteral()
                    else:
                        min_ = ir.NegInfinityLiteral()
                        max_ = im.minus(value, 1)
                case ir.SymbolRef("less_equal"):
                    if reverse:
                        min_ = value
                        max_ = ir.InfinityLiteral()
                    else:
                        min_ = ir.NegInfinityLiteral()
                        max_ = value
                case ir.SymbolRef("greater"):
                    if reverse:
                        min_ = ir.NegInfinityLiteral()
                        max_ = im.minus(value, 1)
                    else:
                        min_ = im.plus(value, 1)
                        max_ = ir.InfinityLiteral()
                case ir.SymbolRef("greater_equal"):
                    if reverse:
                        min_ = ir.NegInfinityLiteral()
                        max_ = value
                    else:
                        min_ = value
                        max_ = ir.InfinityLiteral()
                case ir.SymbolRef("eq"):
                    min_ = max_ = value
                case ir.SymbolRef("not_eq"):
                    min1 = ir.NegInfinityLiteral()
                    max1 = im.minus(value, 1)
                    min2 = im.plus(value, 1)
                    max2 = ir.InfinityLiteral()
                    return im.call("and_")(
                        im.domain(common.GridType.CARTESIAN, {dim: (min1, max1)}),
                        im.domain(common.GridType.CARTESIAN, {dim: (min2, max2)}),
                    )
                case _:
                    raise NotImplementedError
            return im.domain(
                common.GridType.CARTESIAN,
                {dim: (min_, im.plus(max_, 1))}
                if not isinstance(max_, ir.InfinityLiteral)
                else {dim: (min_, max_)},
            )
        if cpm.is_call_to(node, builtins.BINARY_LOGICAL_BUILTINS) and all(
            isinstance(arg, (ir.Literal, ir.FunCall)) for arg in node.args
        ):
            if cpm.is_call_to(node, "and_"):
                # TODO: domain promotion
                return ConstantFolding.apply(
                    domain_utils.domain_intersection(
                        *[domain_utils.SymbolicDomain.from_expr(arg) for arg in node.args]
                    ).as_expr()
                )

            else:
                raise NotImplementedError

        return self.generic_visit(node)
