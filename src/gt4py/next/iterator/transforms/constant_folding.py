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
            if new_node.fun.id in ir.ARITHMETIC_BUILTINS:
                fun = getattr(embedded, str(new_node.fun.id))
                arg_values = [getattr(embedded, str(arg.type))(arg.value) for arg in new_node.args]  # type: ignore[attr-defined] # arg type already established in if condition
                new_node = im.literal_from_value(fun(*arg_values))

        return new_node
