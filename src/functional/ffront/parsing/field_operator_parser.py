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
from typing import Callable, List

from functional.ffront import field_operator_ir as foir
from functional.ffront.parsing.parsing import FieldOperatorSyntaxError, get_ast_from_func


class FieldOperatorParser(ast.NodeVisitor):
    @classmethod
    def parse(cls, func: Callable) -> foir.FieldOperator:
        result = None
        try:
            result = cls().visit(get_ast_from_func(func))
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
