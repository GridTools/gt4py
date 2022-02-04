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


class AssignResolver(NodeTranslator):
    """
    Inline a sequence of assignments into a final return statement.

    >>> from functional.ffront.func_to_foast import FieldOperatorParser
    >>> from functional.common import Field
    >>>
    >>> float64 = float
    >>>
    >>> def fieldop(inp: Field[..., "float64"]):
    ...     tmp1 = inp
    ...     tmp2 = tmp1
    ...     return tmp2
    >>>
    >>> fieldop_foast_expr = AssignResolver.apply(FieldOperatorParser.apply_to_function(fieldop).body)
    >>> fieldop_foast_expr  # doctest: +ELLIPSIS
    Return(location=..., value=Name(location=..., id='inp'))
    """

    @classmethod
    def apply(
        cls, nodes: list[foast.Expr], *, params: Optional[list[itir.Sym]] = None
    ) -> foast.Expr:
        names: dict[str, foast.Expr] = {}
        parser = cls()
        for node in nodes[:-1]:
            names.update(parser.visit(node, names=names))
        return foast.Return(
            value=parser.visit(nodes[-1].value, names=names), location=nodes[-1].location
        )

    def visit_Assign(
        self,
        node: foast.Assign,
        *,
        names: Optional[dict[str, foast.Expr]] = None,
    ) -> dict[str, itir.Expr]:
        return {node.target.id: self.visit(node.value, names=names)}

    def visit_Name(
        self,
        node: foast.Name,
        *,
        names: Optional[dict[str, foast.Expr]] = None,
    ):
        names = names or {}
        if node.id in names:
            return names[node.id]
        return node


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
    >>> lowered.expr
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
        exprs: list[foast.Expr],
        params: Optional[list[itir.Sym]] = None,
        **kwargs,
    ) -> itir.Expr:
        return self.visit(AssignResolver.apply(exprs), **kwargs)

    def visit_Return(self, node: foast.Return, **kwargs) -> itir.Expr:
        result = self.visit(node.value, **kwargs)
        return result

    def visit_FieldSymbol(
        self, node: foast.FieldSymbol, *, symtable: dict[str, foast.Symbol], **kwargs
    ) -> itir.Sym:
        return itir.Sym(id=node.id)

    def visit_Name(
        self,
        node: foast.Name,
        *,
        symtable: dict[str, foast.Symbol],
        shift_args: Optional[list[Union[itir.OffsetLiteral, itir.IntLiteral]]] = None,
        **kwargs,
    ) -> Union[itir.SymRef, itir.FunCall]:
        # always shift a single field, not a field expression
        if _name_is_field(node) and shift_args:
            result = ItirDerefFactory(
                args__0=ItirShiftFactory(shift_args=shift_args, args__0=itir.SymRef(id=node.id))
            )
            return result
        return ItirDerefFactory(args__0=itir.SymRef(id=node.id))

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

    def visit_Shift(self, node: foast.Shift, **kwargs) -> itir.Expr:
        shift_args = list(self._gen_shift_args(node.offsets))
        return self.visit(node.expr, shift_args=shift_args, **kwargs)

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
        return ItirFunCallFactory(
            fun=new_fun,
            args=self.visit(node.args, **kwargs),
        )
