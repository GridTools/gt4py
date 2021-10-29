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

import ast
import inspect
import textwrap
from typing import Callable, List, Optional

from eve import NodeTranslator
from functional.ffront import field_operator_ir as foir
from functional.iterator import ir as iir


class EmptyReturnException(Exception):
    ...


class FieldOperatorSyntaxError(SyntaxError):
    def __init__(self, msg="", *, lineno=0, offset=0, filename=None):
        self.msg = "Invalid Field Operator Syntax: " + msg
        self.lineno = lineno
        self.offset = offset
        self.filename = filename


class FieldOperatorParser(ast.NodeVisitor):
    @classmethod
    def parse(cls, func: Callable) -> foir.FieldOperator:
        if inspect.getabsfile(func) == "<string>":
            raise ValueError(
                "Can not create field operator from a function that is not in a source file!"
            )
        source = textwrap.dedent(inspect.getsource(func))
        result = None
        try:
            result = cls().visit(ast.parse(source).body[0])
        except SyntaxError as err:
            err.filename = inspect.getabsfile(func)
            err.lineno = (err.lineno or 1) + inspect.getsourcelines(func)[1] - 1
            raise err

        return result

    def visit_FunctionDef(self, node: ast.FunctionDef) -> foir.FieldOperator:
        return foir.FieldOperator(
            id=node.name,
            params=self.visit(node.args),
            body=self.visit_stmt_list(node.body),
        )

    def visit_arguments(self, node: ast.arguments) -> foir.Sym:
        return [foir.Sym(id=arg.arg) for arg in node.args]

    def visit_Assign(self, node: ast.Assign) -> foir.SymExpr:
        target = node.targets[0]  # can there be more than one element?
        if isinstance(target, ast.Tuple):
            raise FieldOperatorSyntaxError(
                "Unpacking not allowed!",
                lineno=node.lineno,
                offset=node.col_offset,
            )
        if not isinstance(target, ast.Name):
            raise FieldOperatorSyntaxError(
                "Can only assign to names!",
                lineno=target.lineno,
                offset=target.col_offset,
            )
        return foir.SymExpr(
            id=target.id,
            expr=self.visit(node.value),
        )

    def visit_Return(self, node: ast.Return) -> foir.Return:
        if not node.value:
            raise FieldOperatorSyntaxError(
                "Empty return not allowed", lineno=node.lineno, offset=node.col_offset
            )
        return foir.Return(value=self.visit(node.value))

    def visit_stmt_list(self, nodes: List[ast.stmt]) -> List[foir.Expr]:
        if not isinstance(last_node := nodes[-1], ast.Return):
            raise FieldOperatorSyntaxError(
                msg="Field operator must return a field expression on the last line!",
                lineno=last_node.lineno,
                offset=last_node.col_offset,
            )
        return [self.visit(node) for node in nodes]

    def visit_Name(self, node: ast.Name) -> foir.Name:
        return foir.Name(id=node.id)

    def generic_visit(self, node) -> None:
        raise FieldOperatorSyntaxError(
            lineno=node.lineno,
            offset=node.col_offset,
        )


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
