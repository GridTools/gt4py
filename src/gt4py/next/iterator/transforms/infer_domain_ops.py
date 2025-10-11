# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import dataclasses

from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.next import common
from gt4py.next.iterator import builtins, ir as itir
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    domain_utils,
    ir_makers as im,
    misc as ir_misc,
)
from gt4py.next.iterator.type_system import inference
from gt4py.next.type_system import type_specifications as ts


@dataclasses.dataclass
class InferDomainOps(PreserveLocationVisitor, NodeTranslator):
    grid_type: common.GridType

    @classmethod
    def apply(cls, program: itir.Program):
        return cls(grid_type=ir_misc.grid_type_from_program(program)).visit(program, recurse=True)

    def visit_FunCall(self, node: itir.FunCall, **kwargs) -> itir.Node:
        if kwargs["recurse"]:
            node = self.generic_visit(node, **kwargs)

        # e.g. `IDim < a`
        if cpm.is_call_to(node, builtins.BINARY_MATH_COMPARISON_BUILTINS) and any(
            isinstance(arg, itir.AxisLiteral) for arg in node.args
        ):
            arg1, arg2 = node.args
            if isinstance(arg2, itir.AxisLiteral):
                # take complementary operation if we have e.g. `0 < IDim` use `IDim > 0`
                complementary_op = {
                    "less": "greater",
                    "less_equal": "greater_equal",
                    "greater": "less",
                    "greater_equal": "less_equal",
                    "eq": "eq",
                    "not_eq": "not_eq",
                }
                return self.visit(
                    im.call(complementary_op[node.fun.id])(arg2, arg1),
                    **{**kwargs, "recurse": False},
                )

            inference.reinfer(arg1)
            assert isinstance(arg1.type, ts.DimensionType)
            dim: common.Dimension = arg1.type.dim
            value: itir.Expr = arg2

            if cpm.is_call_to(node, ("less", "less_equal", "greater", "greater_equal", "eq")):
                min_: itir.Expr
                max_: itir.Expr

                # `IDim < 1`
                if cpm.is_call_to(node, "less"):
                    min_ = itir.InfinityLiteral.NEGATIVE
                    max_ = value
                # `IDim <= 1`
                elif cpm.is_call_to(node, "less_equal"):
                    min_ = itir.InfinityLiteral.NEGATIVE
                    max_ = im.plus(value, 1)
                # `IDim > 1`
                elif cpm.is_call_to(node, "greater"):
                    min_ = im.plus(value, 1)
                    max_ = itir.InfinityLiteral.POSITIVE
                # `IDim >= 1`
                elif cpm.is_call_to(node, "greater_equal"):
                    min_ = value
                    max_ = itir.InfinityLiteral.POSITIVE
                # `IDim == 1`
                elif cpm.is_call_to(node, "eq"):
                    min_ = value
                    max_ = im.plus(value, 1)

                domain = domain_utils.SymbolicDomain(
                    self.grid_type,
                    ranges={dim: domain_utils.SymbolicRange(start=min_, stop=max_)},
                )

                return domain.as_expr()
            elif cpm.is_call_to(node, "not_eq"):
                # `IDim != a` -> `IDim < a & IDim > a`
                return self.visit(
                    im.call("and_")(
                        self.visit(
                            im.less(im.axis_literal(dim), value), **(kwargs | {"recurse": False})
                        ),
                        self.visit(
                            im.greater(im.axis_literal(dim), value), **(kwargs | {"recurse": False})
                        ),
                    ),
                    **(kwargs | {"recurse": False}),
                )
            else:
                raise AssertionError()

        return node
