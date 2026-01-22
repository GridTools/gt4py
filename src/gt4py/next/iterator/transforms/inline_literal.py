# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Mapping

from gt4py import eve
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im


class ReplaceLiterals(eve.PreserveLocationVisitor, eve.NodeTranslator):
    PRESERVED_ANNEX_ATTRS = ("type", "domain")

    def visit_FunCall(self, node: ir.FunCall, *, symbol_map: Mapping[str, ir.Literal]):
        node = self.generic_visit(node, symbol_map=symbol_map)

        if cpm.is_call_to(node, "deref"):
            assert len(node.args) == 1
            if (
                isinstance(node.args[0], ir.SymRef)
                and (symbol_name := str(node.args[0].id)) in symbol_map
            ):
                return symbol_map[symbol_name]
        return node


class InlineLiteral(eve.NodeTranslator):
    """Inline literal values (constants) into lambda expressions."""

    PRESERVED_ANNEX_ATTRS = ("domain", "type")

    def visit_FunCall(self, node: ir.FunCall) -> ir.Node:
        node = self.generic_visit(node)

        if cpm.is_applied_as_fieldop(node):
            assert len(node.fun.args) == 1 and isinstance(node.fun.args[0], ir.Lambda)
            lambda_node = node.fun.args[0]
            symbol_map = {}
            fun_args = []
            lambda_params = []
            for lambda_param, fun_arg in zip(lambda_node.params, node.args, strict=True):
                if isinstance(fun_arg, ir.Literal):
                    symbol_name = str(lambda_param.id)
                    symbol_map[symbol_name] = fun_arg
                else:
                    fun_args.append(fun_arg)
                    lambda_params.append(lambda_param)

            if symbol_map:
                lambda_expr = ReplaceLiterals().visit(lambda_node.expr, symbol_map=symbol_map)
                return im.as_fieldop(im.lambda_(*lambda_params)(lambda_expr))(*fun_args)

        return node

    @classmethod
    def apply(cls, node: ir.Node) -> ir.Node:
        return cls().visit(node)
