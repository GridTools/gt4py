# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.next.iterator import builtins, embedded, ir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im


class ConstantFolding(PreserveLocationVisitor, NodeTranslator):
    PRESERVED_ANNEX_ATTRS = (
        "type",
        "domain",
    )

    @classmethod
    def apply(cls, node: ir.Node) -> ir.Node:
        return cls().visit(node)

    def visit_FunCall(self, node: ir.FunCall) -> ir.Node:
        # visit depth-first such that nested constant expressions (e.g. `(1+2)+3`) are properly folded
        new_node = self.generic_visit(node)

        if (
            cpm.is_call_to(new_node, ("minimum", "maximum"))
            and new_node.args[0] == new_node.args[1]
        ):  # `minimum(a, a)` -> `a`
            return new_node.args[0]

        if cpm.is_call_to(new_node, "plus"):
            for arg in new_node.args:
                # `a + inf` -> `inf`
                if arg == ir.InfinityLiteral.POSITIVE:
                    return ir.InfinityLiteral.POSITIVE
                # `a + (-inf)` -> `-inf`
                if arg == ir.InfinityLiteral.NEGATIVE:
                    return ir.InfinityLiteral.NEGATIVE

        if cpm.is_call_to(new_node, "minimum"):
            a, b = new_node.args
            for arg, other_arg in ((a, b), (b, a)):
                # `minimum(inf, a)` -> `a`
                if arg == ir.InfinityLiteral.POSITIVE:
                    return other_arg
                # `minimum(-inf, a)` -> `-inf`
                if arg == ir.InfinityLiteral.NEGATIVE:
                    return ir.InfinityLiteral.NEGATIVE

        if cpm.is_call_to(new_node, "maximum"):
            a, b = new_node.args
            for arg, other_arg in ((a, b), (b, a)):
                # `maximum(inf, a)` -> `inf`
                if arg == ir.InfinityLiteral.POSITIVE:
                    return ir.InfinityLiteral.POSITIVE
                # `maximum(-inf, a)` -> `a`
                if arg == ir.InfinityLiteral.NEGATIVE:
                    return other_arg

        if cpm.is_call_to(new_node, ("less", "less_equal")):
            a, b = new_node.args
            # `-inf < v` -> `True`
            # `v < inf` -> `True`
            if a == ir.InfinityLiteral.NEGATIVE or b == ir.InfinityLiteral.POSITIVE:
                return im.literal_from_value(True)
            # `inf < v` -> `False`
            # `v < -inf ` -> `False`
            if a == ir.InfinityLiteral.POSITIVE or b == ir.InfinityLiteral.NEGATIVE:
                return im.literal_from_value(False)

        if cpm.is_call_to(new_node, ("greater", "greater_equal")):
            a, b = new_node.args
            # `inf > v` -> `True`
            # `v > -inf ` -> `True`
            if a == ir.InfinityLiteral.POSITIVE or b == ir.InfinityLiteral.NEGATIVE:
                return im.literal_from_value(True)
            # `-inf > v` -> `False`
            # `v > inf` -> `False`
            if a == ir.InfinityLiteral.NEGATIVE or b == ir.InfinityLiteral.POSITIVE:
                return im.literal_from_value(False)

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
            if new_node.fun.id in builtins.ARITHMETIC_BUILTINS:
                fun = getattr(embedded, str(new_node.fun.id))
                arg_values = [
                    getattr(embedded, str(arg.type))(arg.value)
                    # type: ignore[attr-defined] # arg type already established in if condition
                    for arg in new_node.args
                ]
                new_node = im.literal_from_value(fun(*arg_values))

        return new_node
