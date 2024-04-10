# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.next.iterator import embedded, ir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.ir_utils.common_pattern_matcher import is_call_to


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
        ):
            # `minimum(a, a)` -> `a`
            if new_node.args[0] == new_node.args[1]:
                return new_node.args[0]

        if is_call_to(new_node, "minimum"):
            left, right = new_node.args
            if isinstance(left, ir.Literal) and left.value == "inf":
                return right
            if isinstance(right, ir.Literal) and right.value == "inf":
                return left

        if is_call_to(new_node, "maximum"):
            left, right = new_node.args
            if isinstance(left, ir.Literal) and left.value == "neginf":
                return right
            if isinstance(right, ir.Literal) and right.value == "neginf":
                return left

        if (
            isinstance(new_node.fun, ir.SymRef)
            and new_node.fun.id == "if_"
            and isinstance(new_node.args[0], ir.Literal)
        ):  # `if_(True, true_branch, false_branch)` -> `true_branch`
            if new_node.args[0].value == "True":
                new_node = new_node.args[1]
            else:
                new_node = new_node.args[2]

        # TODO: doesn't work for 1+(1+s)
        if (
            isinstance(new_node, ir.FunCall)
            and isinstance(new_node.fun, ir.SymRef)
            and len(new_node.args) > 0
            and all(isinstance(arg, ir.Literal) for arg in new_node.args)
        ):  # `1 + 1` -> `2`
            try:
                if new_node.fun.id in ir.ARITHMETIC_BUILTINS:
                    fun = getattr(embedded, str(new_node.fun.id))
                    arg_values = [getattr(embedded, str(arg.type))(arg.value) for arg in new_node.args]  # type: ignore[attr-defined] # arg type already established in if condition
                    new_node = im.literal_from_value(fun(*arg_values))
            except ValueError:
                pass  # happens for inf and neginf

        return new_node
