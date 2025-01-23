# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import enum
import functools
import operator
from typing import Optional

from gt4py import eve
from gt4py.next.iterator import builtins, embedded, ir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im


@dataclasses.dataclass(frozen=True)
class ConstantFolding(eve.PreserveLocationVisitor, eve.NodeTranslator):
    class Flag(enum.Flag):
        # e.g. `literal + symref` -> `symref + literal` and
        # `literal + funcall` -> `funcall + literal` and
        # `symref + funcall` -> `funcall + symref`
        CANONICALIZE_FUNCALL_SYMREF_LITERAL = enum.auto()

        # `minus(arg0, arg1) -> plus(im.call("neg")(arg0), arg1)`
        CANONICALIZE_MINUS = enum.auto()

        # `(sym + 1) + 1` -> `sym + 2`
        FOLD_FUNCALL_LITERAL = enum.auto()

        # `maximum(maximum(sym, 1), sym)` -> `maximum(sym, 1)`
        FOLD_MIN_MAX_FUNCALL_SYMREF_LITERAL = enum.auto()

        # `maximum(plus(sym, 1), sym)` -> `plus(sym, 1)` and
        # `maximum(plus(sym, 1), plus(sym, -1))` -> `plus(sym, 1)`
        FOLD_MIN_MAX_PLUS = enum.auto()

        #  `sym + 0` -> `sym`
        FOLD_SYMREF_PLUS_ZERO = enum.auto()

        # `sym + 1 + (sym + 2)` -> `sym + sym + 2 + 1`
        CANONICALIZE_PLUS_SYMREF_LITERAL = enum.auto()

        # `1 + 1` -> `2`
        FOLD_ARITHMETIC_BUILTINS = enum.auto()

        # `minimum(a, a)` -> `a`
        FOLD_MIN_MAX_LITERALS = enum.auto()

        # `if_(True, true_branch, false_branch)` -> `true_branch`
        FOLD_IF = enum.auto()

        @classmethod
        def all(self):  # TODO -> ConstantFolding.Flag:
            return functools.reduce(operator.or_, self.__members__.values())

    flags: Flag = Flag.all()  # noqa: RUF009 [function-call-in-dataclass-default-argument]

    @classmethod
    def apply(cls, node: ir.Node) -> ir.Node:
        node = cls().visit(node)
        return node

    def visit_FunCall(self, node: ir.FunCall, **kwargs):
        # visit depth-first such that nested constant expressions (e.g. `(1+2)+3`) are properly folded
        node = self.generic_visit(node, **kwargs)
        return self.fp_transform(node, **kwargs)

    def fp_transform(self, node: ir.Node, **kwargs) -> ir.Node:
        while True:
            new_node = self.transform(node, **kwargs)
            if new_node is None:
                break
            assert new_node != node
            node = new_node
        return node

    def transform(self, node: ir.Node, **kwargs) -> Optional[ir.Node]:
        if not isinstance(node, ir.FunCall):
            return None

        for transformation in self.Flag:
            if transformation:
                assert isinstance(transformation.name, str)
                method = getattr(self, f"transform_{transformation.name.lower()}")
                result = method(node)
                if result is not None:
                    assert (
                        result is not node
                    )  # transformation should have returned None, since nothing changed
                    return result
        return None

    def transform_canonicalize_funcall_symref_literal(
        self, node: ir.FunCall, **kwargs
    ) -> Optional[ir.Node]:
        # e.g. `literal + symref` -> `symref + literal` and
        # `literal + funcall` -> `funcall + literal` and
        # `symref + funcall` -> `funcall + symref`
        if isinstance(node.fun, ir.SymRef) and cpm.is_call_to(
            node, ("plus", "multiplies", "minimum", "maximum")
        ):
            if (
                isinstance(node.args[1], (ir.SymRef, ir.FunCall))
                and isinstance(node.args[0], ir.Literal)
            ) or (isinstance(node.args[1], ir.FunCall) and isinstance(node.args[0], ir.SymRef)):
                return im.call(node.fun.id)(node.args[1], node.args[0])
        return None

    def transform_canonicalize_minus(self, node: ir.FunCall, **kwargs) -> Optional[ir.Node]:
        # `minus(arg0, arg1) -> plus(im.call("neg")(arg0), arg1)`
        if cpm.is_call_to(node, "minus"):
            return self.visit(im.plus(im.call("neg")(node.args[1]), node.args[0]))
        return None

    def transform_fold_funcall_literal(self, node: ir.FunCall, **kwargs) -> Optional[ir.Node]:
        # `(sym + 1) + 1` -> `sym + 2`
        if cpm.is_call_to(node, "plus"):
            if cpm.is_call_to(node.args[0], "plus") and isinstance(node.args[1], ir.Literal):
                fun_call, literal = node.args
                if (
                    isinstance(fun_call.args[0], (ir.SymRef, ir.FunCall))  # type: ignore[attr-defined] # assured by if above
                    and isinstance(fun_call.args[1], ir.Literal)  # type: ignore[attr-defined] # assured by if above
                ):
                    return self.visit(im.plus(fun_call.args[0], im.plus(fun_call.args[1], literal)))  # type: ignore[attr-defined] # assured by if above
        return None

    def transform_fold_min_max_funcall_symref_literal(
        self, node: ir.FunCall, **kwargs
    ) -> Optional[ir.Node]:
        # `maximum(maximum(sym, 1), sym)` -> `maximum(sym, 1)`
        if isinstance(node.fun, ir.SymRef) and cpm.is_call_to(node, ("minimum", "maximum")):
            if cpm.is_call_to(node.args[0], ("maximum", "minimum")):
                fun_call, arg1 = node.args
                if arg1 == fun_call.args[0]:  # type: ignore[attr-defined] # assured by if above
                    return self.visit(im.call(fun_call.fun.id)(fun_call.args[1], arg1))  # type: ignore[attr-defined] # assured by if above
                if arg1 == fun_call.args[1]:  # type: ignore[attr-defined] # assured by if above
                    return self.visit(im.call(fun_call.fun.id)(fun_call.args[0], arg1))  # type: ignore[attr-defined] # assured by if above
        return None

    def transform_fold_min_max_plus(self, node: ir.FunCall, **kwargs) -> Optional[ir.Node]:
        if isinstance(node.fun, ir.SymRef) and cpm.is_call_to(node, ("minimum", "maximum")):
            arg0, arg1 = node.args
            # `maximum(plus(sym, 1), sym)` -> `plus(sym, 1)`
            if cpm.is_call_to(arg0, "plus"):
                if arg0.args[0] == arg1:
                    return self.visit(im.plus(arg0.args[0], im.call(node.fun.id)(arg0.args[1], 0)))
            # `maximum(plus(sym, 1), plus(sym, -1))` -> `plus(sym, 1)`
            if cpm.is_call_to(arg0, "plus") and cpm.is_call_to(arg1, "plus"):
                if arg0.args[0] == arg1.args[0]:
                    return self.visit(
                        im.plus(arg0.args[0], im.call(node.fun.id)(arg0.args[1], arg1.args[1]))
                    )
        return None

    def transform_fold_symref_plus_zero(self, node: ir.FunCall, **kwargs) -> Optional[ir.Node]:
        # `sym + 0` -> `sym`
        if (
            cpm.is_call_to(node, "plus")
            and isinstance(node.args[1], ir.Literal)
            and node.args[1].value.isdigit()
            and int(node.args[1].value) == 0
        ):
            return node.args[0]
        return None

    def transform_canonicalize_plus_symref_literal(
        self, node: ir.FunCall, **kwargs
    ) -> Optional[ir.Node]:
        # `sym1 + 1 + (sym2 + 2)` -> `sym1 + sym2 + 2 + 1`
        if cpm.is_call_to(node, "plus"):
            if (
                cpm.is_call_to(node.args[0], "plus")
                and cpm.is_call_to(node.args[1], "plus")
                and isinstance(node.args[0].args[1], ir.Literal)
                and isinstance(node.args[1].args[1], ir.Literal)
            ):
                return self.visit(
                    im.plus(
                        im.plus(node.args[0].args[0], node.args[1].args[0]),
                        im.plus(node.args[0].args[1], node.args[1].args[1]),
                    )
                )
        return None

    def transform_fold_arithmetic_builtins(self, node: ir.FunCall, **kwargs) -> Optional[ir.Node]:
        # `1 + 1` -> `2`
        if (
            isinstance(node, ir.FunCall)
            and isinstance(node.fun, ir.SymRef)
            and len(node.args) > 0
            and all(isinstance(arg, ir.Literal) for arg in node.args)
        ):
            try:
                if node.fun.id in builtins.ARITHMETIC_BUILTINS:
                    fun = getattr(embedded, str(node.fun.id))
                    arg_values = [
                        getattr(embedded, str(arg.type))(arg.value)  # type: ignore[attr-defined] # arg type already established in if condition
                        for arg in node.args
                    ]
                    return im.literal_from_value(fun(*arg_values))
            except ValueError:
                pass  # happens for inf and neginf
        return None

    def transform_fold_min_max_literals(self, node: ir.FunCall, **kwargs) -> Optional[ir.Node]:
        # `minimum(a, a)` -> `a`
        if cpm.is_call_to(node, ("minimum", "maximum")):
            if node.args[0] == node.args[1]:
                return node.args[0]
        return None

    def transform_fold_if(self, node: ir.FunCall, **kwargs) -> Optional[ir.Node]:
        # `if_(True, true_branch, false_branch)` -> `true_branch`
        if cpm.is_call_to(node, "if_") and isinstance(node.args[0], ir.Literal):
            if node.args[0].value == "True":
                return node.args[1]
            else:
                return node.args[2]
        return None
