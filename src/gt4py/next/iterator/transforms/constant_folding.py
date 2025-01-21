# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.next.iterator import embedded, ir
import functools
import operator
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.type_system import type_specifications as ts, type_translation

import dataclasses
import enum
from typing import Optional



class ConvertMinusToUnary(PreserveLocationVisitor, NodeTranslator):
    def visit(self, node: ir.Node):
        node = self.generic_visit(node)

        # im.minus(1, im.ref("a")) -> im.plus(im.call("neg)(im.ref("a")), 1)
        if cpm.is_call_to(node, "minus"):
            if isinstance(node.args[1], (ir.SymRef, ir.FunCall)):
                node = im.plus(im.call("neg")(node.args[1]), node.args[0])
        return node

class ConvertUnaryToMinus(PreserveLocationVisitor, NodeTranslator):

    def visit(self, node: ir.Node):

        if isinstance(node, ir.FunCall) and cpm.is_call_to(node.args[0], "neg"):
            manipulated_first_arg = False
            if node.args[0].args[0].type:
                zero = im.literal(str(0), node.args[0].args[0].type)
            else:
                zero = im.literal_from_value(0.0) # TODO: fix datatype
            if cpm.is_call_to(node, "minus"):
                node = im.plus(node.args[1], node.args[0].args[0])
                manipulated_first_arg = True
            elif cpm.is_call_to(node, "plus"):
                node = im.minus(node.args[1], node.args[0].args[0])
                manipulated_first_arg = True
            elif cpm.is_call_to(node, "multiplies"):
                node = im.multiplies_(im.minus(zero,node.args[0].args[0]), node.args[1])
                manipulated_first_arg = True
            elif cpm.is_call_to(node, "divides"):
                node = im.divides_(im.minus(zero,node.args[0].args[0]), node.args[1])
                manipulated_first_arg = True
            elif cpm.is_call_to(node, "minimum"):
                node = im.call("minimum")(im.minus(zero,node.args[0].args[0]), node.args[1])
                manipulated_first_arg = True
            elif cpm.is_call_to(node, "maximum"):
                node = im.call("maximum")(im.minus(zero,node.args[0].args[0]), node.args[1])
                manipulated_first_arg = True
            if manipulated_first_arg:
                node = self.visit(node)
        elif isinstance(node, ir.FunCall) and len(node.args) > 2 and cpm.is_call_to(node.args[1], "neg"):
            if node.args[0].args[0].type:
                zero = im.literal(str(0), node.args[0].args[0].type)
            else:
                zero = im.literal_from_value(0.0) # TODO: fix datatype
            if cpm.is_call_to(node, "minus"):
                node = im.plus(node.args[0], node.args[1].args[0])
            elif cpm.is_call_to(node, "plus"):
                node = im.minus(node.args[0], node.args[1].args[0])
            elif cpm.is_call_to(node, "multiplies"):
                node = im.multiplies_(node.args[0], im.minus(zero,node.args[1].args[0]))
            elif cpm.is_call_to(node, "divides"):
                node = im.divides_(node.args[0], im.minus(zero, node.args[1].args[0]))
            elif cpm.is_call_to(node, "minimum"):
                node = im.call("minimum")(im.node.args[1], im.minus(zero,node.args[1].args[0]))
            elif cpm.is_call_to(node, "maximum"):
                node = im.call("maximum")(im.node.args[1], im.minus(zero,node.args[1].args[0]))

        return self.generic_visit(node)



@dataclasses.dataclass(frozen=True)
class ConstantFolding(PreserveLocationVisitor, NodeTranslator):
    class Flag(enum.Flag):
        # literal + symref -> symref + literal
        CANONICALIZE_SYMREF_LITERAL = enum.auto()

        # literal + funcall -> funcall + literal
        CANONICALIZE_FUNCALL_LITERAL = enum.auto()

        # `__out_size_1 + 1 + 1` -> `__out_size_1 + 2`
        FOLD_FUNCALL_LITERAL = enum.auto()

        # `maximum(1, __out_size_1)` -> `maximum(__out_size_1, 1)` and  `maximum(__out_size_1, maximum(__out_size_1, 1))` -> `maximum(maximum(__out_size_1, 1), __out_size_1)`
        CANONICALIZE_MIN_MAX_FUNCALL_SYMREF_LITERAL = enum.auto()

        # `maximum(maximum(__out_size_1, 1), __out_size_1)` -> `maximum(__out_size_1, 1)`
        FOLD_MIN_MAX_FUNCALL_SYMREF_LITERAL = enum.auto()

        # `minus(__out_size_1, literal) -> plus(__out_size_1,-literal)`
        CANONICALIZE_MINUS_SYMREF_LITERAL = enum.auto()

        # `maximum(plus(__out_size_1, 1), __out_size_1)` -> `plus(__out_size_1, 1)`
        # and `maximum(plus(__out_size_1, 1), plus(__out_size_1, -1))` -> `plus(__out_size_1, 1)`
        FOLD_MIN_MAX_PLUS_MINUS = enum.auto()

        #  `__out_size_1 + 0` -> `__out_size_1`
        FOLD_SYMREF_PLUS_MINUS_ZERO = enum.auto()

        # `sym + 1 + (sym + 2)` -> `sym + sym + 2 + 1`
        CANONICALIZE_PLUS_SYMREF_LITERAL = enum.auto()

        # `1 + 1` -> `2`
        FOLD_ARITHMETIC_BUILTINS = enum.auto()

        # `neg(1)` -> `-1`
        CANONICALIZE_NEG_LITERAL = enum.auto()

        # `minimum(a, a)` -> `a`
        FOLD_MIN_MAX_LITERALS = enum.auto()

        # `if_(True, true_branch, false_branch)` -> `true_branch`
        FOLD_IF = enum.auto()

        @classmethod
        def all(self): # TODO: -> ConstantFolding.Flag
            return functools.reduce(operator.or_, self.__members__.values())

    flags: Flag = Flag.all()


    @classmethod
    def apply(cls, node: ir.Node, flags: Optional[Flag] = None) -> ir.Node:
        flags = flags or cls.flags

        node = ConvertMinusToUnary().visit(node)
        node = cls().visit(node, flags=flags) #TODO: remove flags?
        node = ConvertUnaryToMinus().visit(node)
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
            if self.flags & transformation:
                assert isinstance(transformation.name, str)
                method = getattr(self, f"transform_{transformation.name.lower()}")
                result = method(node)
                if result is not None:
                    assert (
                            result is not node
                    )  # transformation should have returned None, since nothing changed
                    return result
        return None

    def transform_canonicalize_symref_literal(self, node: ir.FunCall, **kwargs) -> Optional[ir.Node]:
        # literal + symref -> symref + literal
        if cpm.is_call_to(node, ("plus", "multiplies")):
            if cpm.is_call_to(node, ("plus", "times")):
                if isinstance(node.args[1], ir.SymRef) and isinstance(node.args[0], ir.Literal):
                    return im.call(node.fun.id)(node.args[1], node.args[0])
        return None

    def transform_canonicalize_funcall_literal(self, node: ir.FunCall, **kwargs) -> Optional[ir.Node]:
        # literal + funcall -> funcall + literal
        if cpm.is_call_to(node, ("plus", "multiplies")):
            if cpm.is_call_to(node, ("plus", "multiplies")):
                if isinstance(node.args[1], ir.FunCall) and isinstance(node.args[0], ir.Literal):
                    return im.call(node.fun.id)(node.args[1], node.args[0])
        return None

    def transform_fold_funcall_literal(self, node: ir.FunCall, **kwargs) -> Optional[ir.Node]:
        # `__out_size_1 + 1 + 1` -> `__out_size_1 + 2`
        if cpm.is_call_to(node, ("plus", "minus")):
            if isinstance(node.args[0], ir.FunCall) and isinstance(
                node.args[1], ir.Literal
            ):
                fun_call, literal = node.args
                if cpm.is_call_to(fun_call, ("plus", "minus")):
                    if isinstance(fun_call.args[0], (ir.SymRef, ir.FunCall)) and isinstance(
                        fun_call.args[1], ir.Literal
                    ):
                        return self.visit(im.plus(
                            fun_call.args[0],
                            self.visit(im.call(node.fun.id)(fun_call.args[1], literal))))
        return None

    def transform_canonicalize_min_max_funcall_symref_literal(self, node: ir.FunCall, **kwargs) -> Optional[ir.Node]:
        # `maximum(1, __out_size_1)` -> `maximum(__out_size_1, 1)` and  `maximum(__out_size_1, maximum(__out_size_1, 1))` -> `maximum(maximum(__out_size_1, 1), __out_size_1)`
        if cpm.is_call_to(node, ("minimum", "maximum")):
            if ((isinstance(node.args[0], ir.Literal) and isinstance(node.args[1], (ir.SymRef, ir.FunCall))) or
                    (isinstance(node.args[0], ir.SymRef) and isinstance(node.args[1], ir.FunCall))):
                return im.call(node.fun.id)(node.args[1], node.args[0])
        return None


    def transform_fold_min_max_funcall_symref_literal(self, node: ir.FunCall, **kwargs) -> Optional[ir.Node]:
        # `maximum(maximum(__out_size_1, 1), __out_size_1)` -> `maximum(__out_size_1, 1)`
        if cpm.is_call_to(node, ("minimum", "maximum")):
            if isinstance(node.args[0], ir.FunCall):
                fun_call, arg1,  = node.args
                if cpm.is_call_to(fun_call, ("maximum", "minimum")):
                        if arg1 == fun_call.args[0]:
                            return self.visit(im.call(fun_call.fun.id)(fun_call.args[1], arg1))
                        if arg1 == fun_call.args[1]:
                            return self.visit(im.call(fun_call.fun.id)(fun_call.args[0], arg1))
        return None

    def transform_canonicalize_minus_symref_literal(self, node: ir.FunCall, **kwargs) -> Optional[ir.Node]:
        # `minus(__out_size_1, literal) -> plus(__out_size_1,-literal)`
        if cpm.is_call_to(node, "minus") and isinstance(node.args[0], (ir.SymRef, ir.FunCall)) and isinstance(node.args[1], ir.Literal):
            return self.visit(im.plus(node.args[0], im.minus(0, node.args[1])))
        return None

    def transform_fold_min_max_plus_minus(self, node: ir.FunCall, **kwargs) -> Optional[ir.Node]:
        if cpm.is_call_to(node, ("minimum", "maximum")):
            arg0, arg1 = node.args
            # `maximum(plus(__out_size_1, 1), __out_size_1)` -> `plus(__out_size_1, 1)`
            if cpm.is_call_to(arg0, ("plus", "minus")) and isinstance(arg1, (ir.SymRef, ir.FunCall)):
                if arg0.args[0] == arg1:
                    return self.visit(im.call(arg0.fun.id)(arg0.args[0], im.call(node.fun.id)(0, arg0.args[1])))
            # `maximum(plus(__out_size_1, 1), plus(__out_size_1, -1))` -> `plus(__out_size_1, 1)`
            if cpm.is_call_to(arg0, ("plus", "minus")) and cpm.is_call_to(arg1, ("plus", "minus")):
                if arg0.args[0] == arg1.args[0]:
                    return self.visit(im.call(arg0.fun.id)(arg0.args[0], im.call(node.fun.id)(arg0.args[1], arg1.args[1])))
        return None

    def transform_fold_symref_plus_minus_zero(self, node: ir.FunCall, **kwargs) -> Optional[ir.Node]:
        # `__out_size_1 + 0` -> `__out_size_1`
        if cpm.is_call_to(node, ("plus", "minus")) and isinstance(node.args[0], (ir.SymRef, ir.FunCall)) and isinstance(node.args[1], ir.Literal) and node.args[1].value.isdigit() and int(node.args[1].value) == 0:
            return node.args[0]
        return None

    def transform_canonicalize_plus_symref_literal(self, node: ir.FunCall, **kwargs) -> Optional[ir.Node]:
        # `sym1 + 1 + (sym2 + 2)` -> `sym1 + sym2 + 2 + 1`
        if cpm.is_call_to(node, "plus"):
            if cpm.is_call_to(node.args[0], "plus") and cpm.is_call_to(node.args[1], "plus") and isinstance(node.args[0].args[1], ir.Literal) and isinstance(node.args[1].args[1], ir.Literal):
                return self.visit(im.plus(im.plus(node.args[0].args[0], node.args[1].args[0]),im.plus(node.args[0].args[1], node.args[1].args[1])))
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
                if node.fun.id in ir.ARITHMETIC_BUILTINS and not cpm.is_call_to(node, "neg"):
                    fun = getattr(embedded, str(node.fun.id))
                    arg_values = [
                        getattr(embedded, str(arg.type))(arg.value)
                        # type: ignore[attr-defined] # arg type already established in if condition
                        for arg in node.args
                    ]
                    return im.literal_from_value(fun(*arg_values))
            except ValueError:
                pass  # happens for inf and neginf
        return None

    def transform_canonicalize_neg_literal(self, node: ir.FunCall, **kwargs) -> Optional[ir.Node]:
        # `neg(1)` -> `-1`
        if cpm.is_call_to(node, ("neg")):
            if isinstance(node.args[0], ir.Literal):
                return self.visit(im.minus(0, int(node.args[0].value)))
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

        # # `maximum(maximum(__out_size_1, 1), maximum(1, __out_size_1))` -> `maximum(__out_size_1, 1)`
        # if cpm.is_call_to(new_node, ("minimum", "maximum")):
        #     if all(cpm.is_call_to(arg, "maximum") for arg in new_node.args) or all(
        #         cpm.is_call_to(arg, "minimum") for arg in new_node.args
        #     ):
        #         if (
        #             new_node.args[0].args[0] == new_node.args[1].args[1]
        #             and new_node.args[0].args[1] == new_node.args[1].args[0]
        #         ):
        #             new_node = new_node.args[0]
        # # `maximum(maximum(__out_size_1, 1), __out_size_1)` -> `maximum(__out_size_1, 1)`
        # if cpm.is_call_to(new_node, ("minimum", "maximum")):
        #     match = False
        #     if isinstance(new_node.args[0], ir.FunCall) and isinstance(
        #         new_node.args[1], (ir.Literal, ir.SymRef)
        #     ):
        #         fun_call, sym_lit = new_node.args
        #         match = True
        #     elif isinstance(new_node.args[0], (ir.Literal, ir.SymRef)) and isinstance(
        #         new_node.args[1], ir.FunCall
        #     ):
        #         match = True
        #         sym_lit, fun_call = new_node.args
        #     if match and cpm.is_call_to(fun_call, ("maximum", "minimum")):
        #         if isinstance(fun_call.args[0], ir.SymRef) and isinstance(
        #             fun_call.args[1], ir.Literal
        #         ):
        #             if sym_lit == fun_call.args[0]:
        #                 new_node = im.call(fun_call.fun.id)(sym_lit, fun_call.args[1])
        #             if sym_lit == fun_call.args[1]:
        #                 new_node = im.call(fun_call.fun.id)(fun_call.args[0], sym_lit)
        #         if isinstance(fun_call.args[0], ir.Literal) and isinstance(
        #             fun_call.args[1], ir.SymRef
        #         ):
        #             if sym_lit == fun_call.args[0]:
        #                 new_node = im.call(fun_call.fun.id)(fun_call.args[1], sym_lit)
        #             if sym_lit == fun_call.args[1]:
        #                 new_node = im.call(fun_call.fun.id)(sym_lit, fun_call.args[0])
        # # `maximum(plus(__out_size_1, 1), minus(__out_size_1, 1))` -> `plus(__out_size_1, 1)`
        # if cpm.is_call_to(new_node, ("minimum", "maximum")):
        #     if all(cpm.is_call_to(arg, ("plus", "minus")) for arg in new_node.args):
        #         if new_node.args[0].args[0] == new_node.args[1].args[0]:
        #             new_node = im.plus(
        #                 new_node.args[0].args[0],
        #                 self.visit(
        #                     im.call(new_node.fun.id)(
        #                         im.call(new_node.args[0].fun.id)(0, new_node.args[0].args[1]),
        #                         im.call(new_node.args[1].fun.id)(0, new_node.args[1].args[1]),
        #                     )
        #                 ),
        #             )
        #     # `maximum(plus(__out_size_1, 1), __out_size_1)` -> `plus(__out_size_1, 1)`
        #     match = False
        #     if isinstance(new_node.args[0], ir.FunCall) and isinstance(new_node.args[1], ir.SymRef):
        #         fun_call, sym_ref = new_node.args
        #         match = True
        #     elif isinstance(new_node.args[0], ir.SymRef) and isinstance(
        #         new_node.args[1], ir.FunCall
        #     ):
        #         match = True
        #         sym_ref, fun_call = new_node.args
        #     if match and fun_call.fun.id in ["plus", "minus"]:
        #         if fun_call.args[0] == sym_ref:
        #             if new_node.fun.id == "minimum":
        #                 if fun_call.fun.id == "plus":
        #                     new_node = sym_ref if int(fun_call.args[1].value) >= 0 else fun_call
        #                 elif fun_call.fun.id == "minus":
        #                     new_node = fun_call if int(fun_call.args[1].value) > 0 else sym_ref
        #             elif new_node.fun.id == "maximum":
        #                 if fun_call.fun.id == "plus":
        #                     new_node = fun_call if int(fun_call.args[1].value) > 0 else sym_ref
        #                 elif fun_call.fun.id == "minus":
        #                     new_node = sym_ref if int(fun_call.args[1].value) >= 0 else fun_call

        # return new_node
