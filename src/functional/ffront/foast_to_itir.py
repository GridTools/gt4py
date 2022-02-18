# GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
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

from collections import namedtuple
from typing import Optional

import functional
from eve import NodeTranslator, NodeVisitor
from functional.ffront import field_operator_ast as foast
from functional.ffront import itir_makers as im
from functional.ffront.fbuiltins import FUN_BUILTIN_NAMES
from functional.ffront.type_info import TypeInfo
from functional.iterator import ir as itir


FieldAccess = namedtuple("FieldAccess", ["name", "shift"])


class FieldCollector(NodeVisitor):
    def __init__(self):
        self._fields = set()
        self.in_shift = False

    def visit_FunCall(self, node: itir.FunCall, **kwargs) -> None:
        if isinstance(node.fun, itir.FunCall) and node.fun.fun.id == "shift":
            self._fields.add(FieldAccess(node.args[0].id, node.fun.args[0]))
        elif isinstance(node.fun, itir.FunCall) and node.fun.fun.id == "lift":
            self.generic_visit(node.fun)
        else:
            self.generic_visit(node)

    def visit_SymRef(self, node: itir.SymRef, **kwargs) -> None:
        if not hasattr(functional.iterator.builtins, node.id):
            self._fields.add(FieldAccess(node.id, None))

    def getFields(self):
        return list(sorted(self._fields))


class FieldOperatorLowering(NodeTranslator):
    """
    Lower FieldOperator AST (FOAST) to Iterator IR (ITIR).

    Examples
    --------
    >>> from functional.ffront.func_to_foast import FieldOperatorParser
    >>> from functional.common import Field
    >>>
    >>> float64 = float
    >>>
    >>> def fieldop(inp: Field[..., "float64"]):
    ...    return inp
    >>>
    >>> parsed = FieldOperatorParser.apply_to_function(fieldop)
    >>> lowered = FieldOperatorLowering.apply(parsed)
    >>> type(lowered)
    <class 'functional.iterator.ir.FunctionDefinition'>
    >>> lowered.id
    'fieldop'
    >>> lowered.params
    [Sym(id='inp')]
    """

    class lifted_lambda:
        def __init__(self, *params):
            self.params = params

        def __call__(self, expr):
            return im.lift_(im.lambda__(*self.params)(expr))(*self.params)

    @classmethod
    def apply(cls, node: foast.FieldOperator) -> itir.FunctionDefinition:
        return cls().visit(node)

    def visit_FieldOperator(self, node: foast.FieldOperator, **kwargs) -> itir.FunctionDefinition:
        symtable = node.symtable_
        params = self.visit(node.params, symtable=symtable)
        return itir.FunctionDefinition(
            id=node.id,
            params=params,
            expr=self._visit_body(node.body, params=params, symtable=symtable),
        )

    def _visit_body(
        self, body: list[foast.Stmt], params: Optional[list[itir.Sym]] = None, **kwargs
    ) -> itir.FunCall:
        *assigns, return_stmt = body
        current_expr = self.visit(return_stmt, **kwargs)

        for assign in reversed(assigns):
            current_expr = im.let(*self._visit_assign(assign, **kwargs))(current_expr)

        return im.deref_(current_expr)

    def _visit_assign(self, node: foast.Assign, **kwargs) -> tuple[itir.Sym, itir.Expr]:
        sym = self.visit(node.target, **kwargs)
        expr = self.visit(node.value, **kwargs)
        return sym, expr

    def visit_Return(self, node: foast.Return, **kwargs) -> itir.Expr:
        return self.visit(node.value)

    def visit_Symbol(self, node: foast.Symbol, **kwargs) -> itir.Sym:
        return im.sym(node.id)

    def visit_Name(self, node: foast.Name, **kwargs) -> itir.Sym:
        return im.ref(node.id)

    def _lift_lambda(self, node):
        def is_field(expr: foast.Expr) -> bool:
            return TypeInfo(expr.type).is_field_type

        param_names = list(
            node.iter_tree().if_isinstance(foast.Name).filter(is_field).getattr("id").unique()
        )
        return self.lifted_lambda(*param_names)

    def visit_Subscript(self, node: foast.Subscript, **kwargs) -> itir.FunCall:
        return im.tuple_get_(node.index, self.visit(node.value, **kwargs))

    def visit_TupleExpr(self, node: foast.TupleExpr, **kwargs) -> itir.FunCall:
        return im.make_tuple_(*self.visit(node.elts, **kwargs))

    def visit_UnaryOp(self, node: foast.UnaryOp, **kwargs) -> itir.FunCall:
        # TODO(tehrengruber): extend iterator ir to support unary operators
        zero_arg = [itir.IntLiteral(value=0)] if node.op is not foast.UnaryOperator.NOT else []
        return self._lift_lambda(node)(
            im.call_(node.op.value)(*[*zero_arg, im.deref_(self.visit(node.operand, **kwargs))])
        )

    def visit_BinOp(self, node: foast.BinOp, **kwargs) -> itir.FunCall:
        return self._lift_lambda(node)(
            im.call_(node.op.value)(
                im.deref_(self.visit(node.left, **kwargs)),
                im.deref_(self.visit(node.right, **kwargs)),
            )
        )

    def visit_Compare(self, node: foast.Compare, **kwargs) -> itir.FunCall:

        return self._lift_lambda(node)(
            im.call_(node.op.value)(
                im.deref_(self.visit(node.left, **kwargs)),
                im.deref_(self.visit(node.right, **kwargs)),
            )
        )

    def _visit_shift(self, node: foast.Call, **kwargs) -> itir.FunCall:
        if isinstance(node.args[0], foast.Subscript):
            return im.shift_(node.args[0].value.id, node.args[0].index)(
                self.visit(node.func, **kwargs)
            )
        else:
            return im.shift_(node.args[0].id)(self.visit(node.func, **kwargs))

    def _extract_fields(self, expr: itir.Expr) -> list[str]:
        fc = FieldCollector()
        fc.visit(expr)
        return fc.getFields()

    def _make_lambda_rhs(self, expr: itir.Expr) -> itir.FunCall:
        return im.plus_(im.ref("base"), expr)

    def _make_reduce(self, *args, **kwargs) -> itir.FunCall:
        red_rhs = self.visit(args[0], **kwargs)

        # get all fields referenced (technically this could also be external symbols)
        # this assumes that the same field doesn't appear both shifted and not shifted,
        # which should be checked for before the lowering
        rhs_fields = self._extract_fields(red_rhs[0])

        rhs_lambda = im.lambda__("base", *[im.sym(field.name) for field in rhs_fields])(
            self._make_lambda_rhs(red_rhs[0])
        )
        # the arguments passed to the lambda are either a symref to the field, or, if the
        # field is shifted in the original expr, the shifted field
        lambda_args = []
        for field in rhs_fields:
            if field.shift is not None:
                lambda_args.append(field.name)
            else:
                lambda_args.append(im.ref(field.name))

        reduce_call = im.lift_(im.call_("reduce")(rhs_lambda, 0))(*lambda_args)

        return reduce_call

    def visit_Call(self, node: foast.Call, **kwargs) -> itir.FunCall:
        if TypeInfo(node.func.type).is_field_type:
            return self._visit_shift(node, **kwargs)
        elif node.func.id in FUN_BUILTIN_NAMES:
            return self._make_reduce(node.args, **kwargs)
        return self._lift_lambda(node)(
            im.call_(self.visit(node.func, **kwargs))(*self.visit(node.args, **kwargs))
        )
