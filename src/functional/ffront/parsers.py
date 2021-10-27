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
from typing import Any, Callable, List

from functional.iterator import ir


class EmptyReturnException(Exception):
    ...


class FieldOperatorParser(ast.NodeVisitor):
    @classmethod
    def parse(cls, func: Callable) -> ir.FunctionDefinition:
        source = textwrap.dedent(inspect.getsource(func))
        result = cls().visit(ast.parse(source).body[0])

        return result

    def visit_arguments(self, node: ast.arguments) -> List[ir.Sym]:
        return [ir.Sym(id=arg.arg) for arg in node.args]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ir.FunctionDefinition:
        # TODO (ricoh): visit the whole node body,
        # generic_visit seems to do nothing on lists.
        return ir.FunctionDefinition(
            id=node.name, params=self.visit(node.args), expr=self.visit(node.body[0])
        )

    def visit_Return(self, node: ast.Return):
        if not node.value:
            raise EmptyReturnException("fundef may not have empty returns")
        return self.visit(node.value)

    def visit_Expr(self, node: ast.Expr) -> Any:
        return self.visit(node.value)

    def visit_Name(self, node: ast.Name) -> str:
        return ir.FunCall(fun=ir.SymRef(id="deref"), args=[ir.SymRef(id=node.id)])
