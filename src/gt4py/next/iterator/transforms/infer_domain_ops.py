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
from gt4py.next.iterator import builtins, ir
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    domain_utils,
    ir_makers as im,
)
from gt4py.next.iterator.transforms.constant_folding import ConstantFolding
from gt4py.next.program_processors.codegens.gtfn.itir_to_gtfn_ir import _get_gridtype
from gt4py.next.type_system import type_specifications as ts


@dataclasses.dataclass
class InferDomainOps(PreserveLocationVisitor, NodeTranslator):
    grid_type: common.GridType

    @classmethod
    def apply(cls, program: ir.Program):
        # TODO: move _get_gridtype
        return cls(grid_type=_get_gridtype(program.body)).visit(program, recurse=True)

    def visit_FunCall(self, node: ir.FunCall, **kwargs) -> ir.Node:
        if kwargs["recurse"]:
            node = self.generic_visit(node, **kwargs)

        # IDim < a
        if (
            cpm.is_call_to(node, builtins.BINARY_MATH_COMPARISON_BUILTINS)
            and any(isinstance(arg, ir.AxisLiteral) for arg in node.args)
            and any(isinstance(arg, ir.Expr) for arg in node.args)
        ):  # TODO: add tests
            arg1, arg2 = node.args
            if isinstance(arg2, ir.AxisLiteral):
                # take complementary operation if we have e.g. `0 < IDim` use `IDim > 0`
                complementary_op = {
                    "less": "greater",
                    "less_equal": "greater_equal",
                    "greater": "greater_equal",
                    "greater_equal": "less_equal",
                    "eq": "eq",
                    "not_eq": "not_eq",
                }
                return self.visit(
                    im.call(complementary_op[node.fun.id])(arg2, arg1),
                    **{**kwargs, "recurse": False},
                )

            assert isinstance(arg1.type, ts.DimensionType)
            dim: common.Dimension = arg1.type.dim
            value: ir.Expr = arg2

            if cpm.is_call_to(node, ("less", "less_equal", "greater", "greater_equal", "eq")):
                min_: int | ir.InfinityLiteral
                max_: int | ir.InfinityLiteral

                # `IDim < 1`
                if cpm.is_call_to(node, "less"):
                    min_ = ir.InfinityLiteral.NEGATIVE
                    max_ = value
                # `IDim <= 1`
                elif cpm.is_call_to(node, "less_equal"):
                    min_ = ir.InfinityLiteral.NEGATIVE
                    max_ = im.plus(value, 1)
                # `IDim > 1`
                elif cpm.is_call_to(node, "greater"):
                    min_ = im.plus(value, 1)
                    max_ = ir.InfinityLiteral.POSITIVE
                # `IDim >= 1`
                elif cpm.is_call_to(node, "greater_equal"):
                    min_ = value
                    max_ = ir.InfinityLiteral.POSITIVE
                # `IDim == 1`  # TODO: isn't this removed before and rewritten as two concat_where?
                elif cpm.is_call_to(node, "eq"):
                    min_ = value
                    max_ = im.plus(value, 1)

                domain = domain_utils.SymbolicDomain(
                    self.grid_type,
                    ranges={dim: domain_utils.SymbolicRange(start=min_, stop=max_)},
                )

                return domain.as_expr()
            elif cpm.is_call_to(node, "not_eq"):
                # `IDim != a -> `IDim < a & IDim > a`
                return im.call("and_")(
                    self.visit(im.less(dim, value), **kwargs),
                    self.visit(im.greater(dim, value), **kwargs),
                )
            else:
                raise AssertionError()

        # if cpm.is_call_to(node, builtins.BINARY_LOGICAL_BUILTINS) and all(
        #     isinstance(arg.type, ts.DomainType) for arg in node.args
        # ):
        #     if cpm.is_call_to(node, "and_"):
        #         # TODO: domain promotion
        #         return ConstantFolding.apply(
        #             domain_utils.domain_intersection(
        #                 *[domain_utils.SymbolicDomain.from_expr(arg) for arg in node.args]
        #             ).as_expr()
        #         )
        #     else:
        #         raise NotImplementedError()

        return node
