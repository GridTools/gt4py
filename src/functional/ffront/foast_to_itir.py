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

from typing import Optional

from eve import NodeTranslator
from functional.ffront import field_operator_ast as foast
from functional.ffront import mockitir as mi
from functional.ffront.type_info import TypeInfo
from functional.iterator import ir as itir


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

    @classmethod
    def apply(cls, node: foast.FieldOperator) -> itir.FunctionDefinition:
        return cls().visit(node)

    def visit_FieldOperator(self, node: foast.FieldOperator, **kwargs) -> itir.FunctionDefinition:
        symtable = node.symtable_
        params = self.visit(node.params, symtable=symtable)
        return itir.FunctionDefinition(
            id=node.id,
            params=params,
            expr=self.body_visit(node.body, params=params, symtable=symtable),
        )

    def body_visit(
        self, body: list[foast.Stmt], params: Optional[list[itir.Sym]] = None, **kwargs
    ) -> itir.FunCall:
        *assigns, return_stmt = body
        current_expr = self.visit(return_stmt, **kwargs)

        for assign in assigns[-1::-1]:
            current_expr = mi.let(*self.visit(assign, **kwargs))(current_expr)

        return mi.deref(current_expr)

    def _make_lambda_param_names(self, node: foast.Expr, **kwargs) -> list[str]:
        def is_field(expr: foast.Expr) -> bool:
            return TypeInfo(expr.type).is_field_type

        names = list(
            node.iter_tree().if_isinstance(foast.Name).filter(is_field).getattr("id").unique()
        )
        return [name for name in names]

    def visit_Assign(self, node: foast.Assign, **kwargs) -> tuple[itir.Sym, itir.Expr]:
        sym = self.visit(node.target, **kwargs)
        expr = self.visit(node.value, **kwargs)
        return sym, expr

    def visit_Return(self, node: foast.Return, **kwargs) -> itir.Expr:
        return self.visit(node.value)

    def visit_Symbol(self, node: foast.Symbol, **kwargs) -> itir.Sym:
        return mi.sym(node.id)

    def visit_Name(self, node: foast.Name, **kwargs) -> itir.SymRef:
        return mi.ref(node.id)

    def visit_Subscript(self, node: foast.Subscript, **kwargs) -> itir.FunCall:
        param_names = self._make_lambda_param_names(node, **kwargs)
        return mi.lift(
            mi.lambda_(*param_names)(
                mi.tuple_get(mi.deref(self.visit(node.value, **kwargs)), node.index)
            )
        )(*param_names)

    def visit_TupleExpr(self, node: foast.TupleExpr, **kwargs) -> itir.FunCall:
        args = [mi.deref(arg) for arg in self.visit(node.elts, **kwargs)]
        param_names = self._make_lambda_param_names(node, **kwargs)
        return mi.lift(mi.lambda_(*param_names)(mi.make_tuple(*args)))(*param_names)

    def visit_UnaryOp(self, node: foast.UnaryOp, **kwargs) -> itir.FunCall:
        # todo(tehrengruber): extend iterator ir to support unary operators
        zero_arg = [itir.IntLiteral(value=0)] if node.op is not foast.UnaryOperator.NOT else []
        param_names = self._make_lambda_param_names(node)
        return mi.lift(
            mi.lambda_(*param_names)(
                mi.call(node.op.value)(*[*zero_arg, mi.deref(self.visit(node.operand, **kwargs))])
            )
        )(*param_names)

    def visit_BinOp(self, node: foast.BinOp, **kwargs) -> itir.FunCall:
        param_names = self._make_lambda_param_names(node)
        return mi.lift(
            mi.lambda_(*param_names)(
                mi.call(node.op.value)(
                    mi.deref(self.visit(node.left, **kwargs)),
                    mi.deref(self.visit(node.right, **kwargs)),
                )
            )
        )(*param_names)

    def visit_Compare(self, node: foast.Compare, **kwargs) -> itir.FunCall:
        param_names = self._make_lambda_param_names(node, **kwargs)
        return mi.lift(
            mi.lambda_(*param_names)(
                mi.call(node.op.value)(
                    mi.deref(self.visit(node.left, **kwargs)),
                    mi.deref(self.visit(node.right, **kwargs)),
                )
            )
        )(*param_names)

    def visit_shift(self, node: foast.Call, **kwargs) -> itir.FunCall:
        return mi.shift(node.args[0].value.id, node.args[0].index)(self.visit(node.func, **kwargs))

    def visit_Call(self, node: foast.Call, **kwargs) -> itir.FunCall:
        param_names = self._make_lambda_param_names(node, **kwargs)
        if TypeInfo(node.func.type).is_field_type:
            return self.visit_shift(node, **kwargs)
        return mi.lift(
            mi.lambda_(*param_names)(
                mi.call(self.visit(node.func, **kwargs))(*self.visit(node.args, **kwargs))
            )
        )(*param_names)
