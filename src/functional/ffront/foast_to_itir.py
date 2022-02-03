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

from typing import Iterator, Optional, Union

import factory

from eve import NodeTranslator
from functional.ffront import common_types
from functional.ffront import field_operator_ast as foast
from functional.iterator import ir as itir


class ItirSymRefFactory(factory.Factory):
    class Meta:
        model = itir.SymRef


class ItirFunCallFactory(factory.Factory):
    """
    Readability enhancing shortcut for itir.FunCall constructor.

    Usage:
    ------

    >>> ItirFunCallFactory(name="plus")
    FunCall(fun=SymRef(id='plus'), args=[])
    """

    class Meta:
        model = itir.FunCall

    class Params:
        name: Optional[str] = None

    fun = factory.LazyAttribute(lambda obj: ItirSymRefFactory(id=obj.name))
    args = factory.List([])


class ItirDerefFactory(ItirFunCallFactory):
    """Readability enhancing shortcut constructing deref itir builtins."""

    class Params:
        name = "deref"


class ItirShiftFactory(ItirFunCallFactory):
    """Readability enhancing shortcut constructing shift itir builtins."""

    class Params:
        name = "shift"
        shift_args: list[Union[itir.OffsetLiteral, itir.IntLiteral]] = []

    fun = factory.LazyAttribute(lambda obj: ItirFunCallFactory(name=obj.name, args=obj.shift_args))


class ItirLiftedLambdaCallFactory(ItirFunCallFactory):
    class Params:
        lambda_expr: Optional[itir.Lambda] = None

    fun = factory.LazyAttribute(  # lifted lambda
        lambda obj: ItirFunCallFactory(name="lift", args__0=obj.lambda_expr)
    )


def _name_is_field(name: foast.Name) -> bool:
    return isinstance(name.type, common_types.FieldType)


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
    >>> lowered.expr.args[0].fun.args[0].expr
    FunCall(fun=SymRef(id='deref'), args=[SymRef(id='inp')])
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
        self,
        stmts: list[foast.Stmt],
        params: Optional[list[itir.Sym]] = None,
        **kwargs,
    ) -> itir.Expr:
        lambdas: dict[str, itir.Lambda] = {}
        for stmt in stmts[:-1]:
            lambdas.update(self.visit(stmt, lambdas=lambdas, **kwargs))
        return self.visit(stmts[-1], lambdas=lambdas, **kwargs)

    def _make_lifted_lambda_call(self, node: foast.Expr, **kwargs) -> itir.Lambda:
        param_names = self._make_lambda_param_names(node)
        lambdas = kwargs.get("lambdas", {})
        lambda_params = [itir.Sym(id=name.replace("$", "__")) for name in param_names]
        #  lifted_lambda_args = [itir.SymRef(id=name) for name in param_names if name not in lambdas else lambdas[name]]
        lifted_lambda_args = []
        for name in param_names:
            if name in lambdas:
                lifted_lambda_args.append(lambdas[name])
            else:
                lifted_lambda_args.append(itir.SymRef(id=name))

        return ItirLiftedLambdaCallFactory(
            lambda_expr=itir.Lambda(params=lambda_params, expr=self.visit(node, **kwargs)),
            args=lifted_lambda_args,
        )

    def _make_lambda_param_names(self, node: foast.Expr, **kwargs) -> list[str]:
        def is_not_offset(expr: foast.Expr) -> bool:
            return not isinstance(expr.type, common_types.OffsetType)

        return list(node.iter_tree().if_isinstance(foast.Name).filter(is_not_offset).getattr("id"))

    def visit_Assign(self, node: foast.Assign, **kwargs) -> dict[str, itir.FunCall]:
        return {
            node.target.id: self._make_lifted_lambda_call(
                node.value,
                **kwargs,
            )
        }

    def visit_Return(self, node: foast.Return, **kwargs) -> itir.Expr:
        result = self._make_lifted_lambda_call(
            node.value,
            **kwargs,
        )
        return ItirDerefFactory(args__0=result)

    def visit_FieldSymbol(self, node: foast.FieldSymbol, **kwargs) -> itir.Sym:
        return itir.Sym(id=node.id)

    def visit_Name(
        self,
        node: foast.Name,
        **kwargs,
    ) -> itir.FunCall:
        return ItirDerefFactory(args__0=itir.SymRef(id=node.id.replace("$", "__")))

    def visit_Subscript(self, node: foast.Subscript, **kwargs) -> itir.FunCall:
        return ItirFunCallFactory(
            name="tuple_get",
            args=[self.visit(node.value, **kwargs), itir.IntLiteral(value=node.index)],
        )

    def visit_TupleExpr(self, node: foast.TupleExpr, **kwargs) -> itir.FunCall:
        return ItirFunCallFactory(name="make_tuple", args=self.visit(node.elts, **kwargs))

    def visit_UnaryOp(self, node: foast.UnaryOp, **kwargs) -> itir.FunCall:
        zero_arg = [itir.IntLiteral(value=0)] if node.op is not foast.UnaryOperator.NOT else []
        return ItirFunCallFactory(
            name=node.op.value,
            args=[*zero_arg, self.visit(node.operand, **kwargs)],
        )

    def visit_BinOp(self, node: foast.BinOp, **kwargs) -> itir.FunCall:
        new_fun = itir.SymRef(id=node.op.value)
        return ItirFunCallFactory(
            fun=new_fun,
            args=[self.visit(node.left, **kwargs), self.visit(node.right, **kwargs)],
        )

    def visit_Compare(self, node: foast.Compare, **kwargs) -> itir.FunCall:
        return ItirFunCallFactory(
            name=node.op.value,
            args=[self.visit(node.left, **kwargs), self.visit(node.right, **kwargs)],
        )

    def visit_Shift(self, node: foast.Shift, **kwargs) -> itir.FunCall:
        shift_args = list(self._gen_shift_args(node.offsets))

        return ItirDerefFactory(
            args__0=ItirShiftFactory(
                shift_args=shift_args,
                args__0=self._make_lifted_lambda_call(
                    node.expr,
                    **kwargs,
                ),
            )
        )

    def _make_shift_args(
        self, node: foast.Subscript, **kwargs
    ) -> tuple[itir.OffsetLiteral, itir.IntLiteral]:
        return (itir.OffsetLiteral(value=node.value.id), itir.IntLiteral(value=node.index))

    def _gen_shift_args(
        self, args: list[foast.Subscript], **kwargs
    ) -> Iterator[Union[itir.OffsetLiteral, itir.IntLiteral]]:
        for arg in args:
            name, offset = self._make_shift_args(arg)
            yield name
            yield offset

    def visit_Call(self, node: foast.Call, **kwargs) -> itir.FunCall:
        new_fun = (
            itir.SymRef(id=node.func.id)
            if isinstance(node.func, foast.Name)  # name called, e.g. my_fieldop(a, b)
            else self.visit(node.func, **kwargs)  # expression called, e.g. local_op[...](a, b)
        )
        return ItirDerefFactory(
            args__0=ItirFunCallFactory(
                fun=new_fun,
                args=self.visit(node.args, **kwargs),
            )
        )
