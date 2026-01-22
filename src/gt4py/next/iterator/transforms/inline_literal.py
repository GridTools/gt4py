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
        if cpm.is_call_to(node, "deref"):
            assert len(node.args) == 1
            if (
                isinstance(node.args[0], ir.SymRef)
                and (symbol_name := str(node.args[0].id)) in symbol_map
            ):
                return symbol_map[symbol_name]

        return self.generic_visit(node, symbol_map=symbol_map)

    def visit_SymRef(self, node: ir.SymRef, *, symbol_map: Mapping[str, ir.Literal]):
        return symbol_map.get(str(node.id), node)


class InlineLiteral(eve.NodeTranslator):
    """Inline literal arguments (constants) of field operators into the lambda expression."""

    PRESERVED_ANNEX_ATTRS = ("domain", "type")

    def visit_FunCall(self, node: ir.FunCall) -> ir.Node:
        node = self.generic_visit(node)

        if cpm.is_applied_as_fieldop(node):
            assert len(node.fun.args) in {1, 2}

            lambda_params = []
            if isinstance(node.fun.args[0], ir.Lambda):
                lambda_node = node.fun.args[0]
            elif cpm.is_call_to(node.fun.args[0], "scan"):
                assert isinstance(node.fun.args[0].args[0], ir.Lambda)
                lambda_node = node.fun.args[0].args[0]
                lambda_params.append(lambda_node.params[0])
            else:
                return node

            fun_args = []
            symbol_map = {}
            pstart = len(lambda_params)
            for lambda_param, fun_arg in zip(lambda_node.params[pstart:], node.args, strict=True):
                if isinstance(fun_arg, ir.Literal):
                    symbol_name = str(lambda_param.id)
                    symbol_map[symbol_name] = fun_arg
                else:
                    fun_args.append(fun_arg)
                    lambda_params.append(lambda_param)

            if symbol_map:
                domain = node.fun.args[1] if len(node.fun.args) == 2 else None
                lambda_expr = ReplaceLiterals().visit(lambda_node.expr, symbol_map=symbol_map)
                lambda_node = im.lambda_(*lambda_params)(lambda_expr)
                if isinstance(node.fun.args[0], ir.Lambda):
                    return im.as_fieldop(lambda_node, domain)(*fun_args)
                else:
                    scan_expr = im.scan(
                        lambda_node, node.fun.args[0].args[1], node.fun.args[0].args[2]
                    )
                    return im.as_fieldop(scan_expr, domain)(*fun_args)
        return node

    @classmethod
    def apply(cls, node: ir.Node) -> ir.Node:
        return cls().visit(node)
