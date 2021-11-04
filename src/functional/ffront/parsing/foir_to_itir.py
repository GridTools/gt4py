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
from functional.ffront import field_operator_ir as foir
from functional.iterator import ir as iir


class SymExprResolver(NodeTranslator):
    @classmethod
    def parse(cls, nodes: List[foir.Expr], *, params: Optional[list[iir.Sym]] = None) -> foir.Expr:
        names: dict[str, foir.Expr] = {}
        parser = cls()
        for node in nodes[:-1]:
            names.update(parser.visit(node, names=names))
        return foir.Return(value=parser.visit(nodes[-1], names=names))

    def visit_SymExpr(
        self,
        node: foir.SymExpr,
        *,
        names: Optional[dict[str, foir.Expr]] = None,
    ) -> dict[str, iir.Expr]:
        return {node.id: self.visit(node.expr, names=names)}

    def visit_Name(
        self,
        node: foir.Name,
        *,
        names: Optional[dict[str, foir.Expr]] = None,
    ):
        names = names or {}
        if node.id in names:
            return names[node.id]
        return foir.SymRef(id=node.id)


class FieldOperatorLowering(NodeTranslator):
    @classmethod
    def parse(cls, node: foir.FieldOperator) -> iir.FunctionDefinition:
        return cls().visit(node)

    def visit_FieldOperator(self, node: foir.FieldOperator) -> iir.FunctionDefinition:
        params = self.visit(node.params)
        return iir.FunctionDefinition(
            id=node.id, params=params, expr=self.body_visit(node.body, params=params)
        )

    def body_visit(
        self, exprs: List[foir.Expr], params: Optional[List[iir.Sym]] = None
    ) -> iir.Expr:
        return self.visit(SymExprResolver.parse(exprs))

    def visit_Return(self, node: foir.Return) -> iir.Expr:
        return self.visit(node.value)

    def visit_Sym(self, node: foir.Sym) -> iir.Sym:
        return iir.Sym(id=node.id)

    def visit_SymRef(self, node: foir.SymRef) -> iir.FunCall:
        return iir.FunCall(fun=iir.SymRef(id="deref"), args=[iir.SymRef(id=node.id)])
