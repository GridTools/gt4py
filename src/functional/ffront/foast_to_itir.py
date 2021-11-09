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
from functional.iterator import ir as iir


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
    >>> fieldop_foast_expr
    Return(value=SymRef(id='inp'))
    """

    @classmethod
    def apply(
        cls, nodes: List[foast.Expr], *, params: Optional[list[iir.Sym]] = None
    ) -> foast.Expr:
        names: dict[str, foast.Expr] = {}
        parser = cls()
        for node in nodes[:-1]:
            names.update(parser.visit(node, names=names))
        return foast.Return(value=parser.visit(nodes[-1].value, names=names))

    def visit_Assign(
        self,
        node: foast.Assign,
        *,
        names: Optional[dict[str, foast.Expr]] = None,
    ) -> dict[str, iir.Expr]:
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
        return foast.SymRef(id=node.id)


class FieldOperatorLowering(NodeTranslator):
    """
    Lower FieldOperator IR / AST (FOIR) to Iterator IR (ITIR).

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
    def apply(cls, node: foast.FieldOperator) -> iir.FunctionDefinition:
        return cls().visit(node)

    def visit_FieldOperator(self, node: foast.FieldOperator) -> iir.FunctionDefinition:
        params = self.visit(node.params)
        return iir.FunctionDefinition(
            id=node.id, params=params, expr=self.body_visit(node.body, params=params)
        )

    def body_visit(
        self, exprs: List[foast.Expr], params: Optional[List[iir.Sym]] = None
    ) -> iir.Expr:
        return self.visit(AssignResolver.apply(exprs))

    def visit_Return(self, node: foast.Return) -> iir.Expr:
        return self.visit(node.value)

    def visit_Sym(self, node: foast.Sym) -> iir.Sym:
        return iir.Sym(id=node.id)

    def visit_SymRef(self, node: foast.SymRef) -> iir.FunCall:
        return iir.FunCall(fun=iir.SymRef(id="deref"), args=[iir.SymRef(id=node.id)])

    def visit_Subscript(self, node: foast.Subscript) -> iir.FunCall:
        return iir.FunCall(
            fun=iir.SymRef(id="tuple_get"),
            args=[self.visit(node.expr), iir.IntLiteral(value=node.index)],
        )

    def visit_Tuple(self, node: foast.Tuple) -> iir.FunCall:
        return iir.FunCall(fun=iir.SymRef(id="make_tuple"), args=[self.visit(i) for i in node.elts])
