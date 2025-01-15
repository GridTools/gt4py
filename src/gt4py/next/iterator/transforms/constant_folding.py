# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.next.iterator import embedded, ir
from gt4py.next.iterator.ir_utils import ir_makers as im


class ConstantFolding(PreserveLocationVisitor, NodeTranslator):
    @classmethod
    def apply(cls, node: ir.Node) -> ir.Node:
        return cls().visit(node)

    def visit_FunCall(self, node: ir.FunCall):
        # visit depth-first such that nested constant expressions (e.g. `(1+2)+3`) are properly folded
        new_node = self.generic_visit(node)

        if (
            isinstance(new_node.fun, ir.SymRef)
            and new_node.fun.id in ["minimum", "maximum"]
            and new_node.args[0] == new_node.args[1]
        ):  # `minimum(a, a)` -> `a`
            return new_node.args[0]
        if isinstance(new_node.fun, ir.SymRef) and new_node.fun.id in ["minimum", "maximum"]:
            if new_node.args[0] == new_node.args[1]:
                return new_node.args[0]
            if isinstance(new_node.args[0], ir.FunCall) and isinstance(new_node.args[1], ir.SymRef):
                fun_call, sym_ref = new_node.args
            elif isinstance(new_node.args[0], ir.SymRef) and isinstance(
                new_node.args[1], ir.FunCall
            ):
                sym_ref, fun_call = new_node.args
            else:
                return new_node
            if fun_call.fun.id in ["plus", "minus"]:
                if fun_call.args[0] == sym_ref:
                    if new_node.fun.id == "minimum":
                        if fun_call.fun.id == "plus":
                            return sym_ref if fun_call.args[1].value >= "0" else fun_call
                        elif fun_call.fun.id == "minus":
                            return fun_call if fun_call.args[1].value > "0" else sym_ref
                    elif new_node.fun.id == "maximum":
                        if fun_call.fun.id == "plus":
                            return fun_call if fun_call.args[1].value > "0" else sym_ref
                        elif fun_call.fun.id == "minus":
                            return sym_ref if fun_call.args[1].value >= "0" else fun_call
                return new_node.args[0]
        if (
            isinstance(new_node.fun, ir.SymRef)
            and new_node.fun.id == "if_"
            and isinstance(new_node.args[0], ir.Literal)
        ):  # `if_(True, true_branch, false_branch)` -> `true_branch`
            if new_node.args[0].value == "True":
                new_node = new_node.args[1]
            else:
                new_node = new_node.args[2]

        if (
            isinstance(new_node, ir.FunCall)
            and isinstance(new_node.fun, ir.SymRef)
            and len(new_node.args) > 0
            and all(isinstance(arg, ir.Literal) for arg in new_node.args)
        ):  # `1 + 1` -> `2`
            try:
                if new_node.fun.id in ir.ARITHMETIC_BUILTINS:
                    fun = getattr(embedded, str(new_node.fun.id))
                    arg_values = [
                        getattr(embedded, str(arg.type))(arg.value)  # type: ignore[attr-defined] # arg type already established in if condition
                        for arg in new_node.args
                    ]
                    new_node = im.literal_from_value(fun(*arg_values))
            except ValueError:
                pass  # happens for inf and neginf

        return new_node
