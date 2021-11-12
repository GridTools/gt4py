#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
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

from typing import List, Optional

from eve import NodeTranslator
from functional.ffront import field_operator_ast as foast
from functional.iterator import ir as itir


class AssignResolver(NodeTranslator):
    """
    Inline a sequence of assignments into a final return statement.

    >>> from functional.ffront.func_to_foast import FieldOperatorParser
    >>>
    >>> def fieldop(inp):
    ...     tmp1 = inp
    ...     tmp2 = tmp1
    ...     return tmp2
    >>>
    >>> fieldop_foast_expr = AssignResolver.apply(FieldOperatorParser.apply(fieldop).body)
    >>> fieldop_foast_expr  # doctest: +ELLIPSIS
    Return(location=..., value=SymRef(location=..., id='inp'))
    """

    @classmethod
    def apply(
        cls, nodes: List[foast.Expr], *, params: Optional[list[itir.Sym]] = None
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
        return foast.SymRef(id=node.id, location=node.location)


class FieldOperatorLowering(NodeTranslator):
    """
    Lower FieldOperator AST (FOAST) to Iterator IR (ITIR).

    >>> from functional.ffront.func_to_foast import FieldOperatorParser
    >>>
    >>> def fieldop(inp):
    ...    return inp
    >>>
    >>> parsed = FieldOperatorParser.apply(fieldop)
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

    def visit_FieldOperator(self, node: foast.FieldOperator) -> itir.FunctionDefinition:
        params = self.visit(node.params)
        return itir.FunctionDefinition(
            id=node.id, params=params, expr=self.body_visit(node.body, params=params)
        )

    def body_visit(
        self, exprs: List[foast.Expr], params: Optional[List[itir.Sym]] = None
    ) -> itir.Expr:
        return self.visit(AssignResolver.apply(exprs))

    def visit_Return(self, node: foast.Return) -> itir.Expr:
        return self.visit(node.value)

    def visit_Sym(self, node: foast.Sym) -> itir.Sym:
        return itir.Sym(id=node.id)

    def visit_SymRef(self, node: foast.SymRef) -> itir.FunCall:
        return itir.FunCall(fun=itir.SymRef(id="deref"), args=[itir.SymRef(id=node.id)])

    def visit_Name(self, node: foast.Name) -> itir.SymRef:
        return itir.SymRef(id=node.id)

    def visit_Subscript(self, node: foast.Subscript) -> itir.FunCall:
        return itir.FunCall(
            fun=itir.SymRef(id="tuple_get"),
            args=[self.visit(node.value), itir.IntLiteral(value=node.index)],
        )

    def visit_Tuple(self, node: foast.Tuple) -> itir.FunCall:
        return itir.FunCall(
            fun=itir.SymRef(id="make_tuple"), args=[self.visit(i) for i in node.elts]
        )

    def visit_UnaryOp(self, node: foast.UnaryOp) -> itir.FunCall:
        zero_arg = [itir.IntLiteral(value=0)] if node.op is not foast.UnaryOperator.NOT else []
        return itir.FunCall(
            fun=itir.SymRef(id=node.op.value),
            args=[*zero_arg, self.visit(node.operand)],
        )

    def visit_BinOp(self, node: foast.BinOp) -> itir.FunCall:
        return itir.FunCall(
            fun=itir.SymRef(id=node.op.value), args=[self.visit(node.left), self.visit(node.right)]
        )

    def visit_Compare(self, node: foast.Compare) -> itir.FunCall:
        return itir.FunCall(
            fun=itir.SymRef(id=node.op.value), args=[self.visit(node.left), self.visit(node.right)]
        )

    def visit_Call(self, node: foast.Call) -> itir.FunCall:
        new_fun = (
            itir.SymRef(id=node.func.id)
            if isinstance(node.func, foast.SymRef)
            else self.visit(node.func)
        )
        return itir.FunCall(
            fun=new_fun,
            args=[self.visit(arg) for arg in node.args],
        )
