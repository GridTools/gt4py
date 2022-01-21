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
from typing import Optional


class StringifyAnnotationsPass(ast.NodeTransformer):
    """
    Rename variables in assignments to avoid overwriting.

    Mutates the python AST, variable names will not be valid python names anymore.
    This pass must be run before any passes that linearize unpacking assignments.


    Example
    -------
    Function ``foo()`` in the following example keeps overwriting local variable ``a``

    >>> import ast, inspect
    >>> from functional.ffront.ast_passes.single_static_assign import SingleStaticAssignPass

    >>> def foo():
    ...     a = 1
    ...     a = 2 + a
    ...     a = 3 + a
    ...     return a

    >>> print(ast.unparse(
    ...     SingleStaticAssignPass.apply(
    ...         ast.parse(inspect.getsource(foo))
    ...     )
    ... ))
    def foo():
        a$0 = 1
        a$1 = 2 + a$0
        a$2 = 3 + a$1
        return a$2

    Note that each variable name is assigned only once and never updated / overwritten.

    Note also that after parsing, running the pass and unparsing we get invalid but
    readable python code. This is ok because this pass is not intended for
    python-to-python translation.
    """

    @classmethod
    def apply(cls, node: ast.AST) -> ast.AST:
        return cls().visit(node)

    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.annotation = self._stringify_annotation(node.annotation)
        result = self.generic_visit(node)
        assert isinstance(result, ast.arg)
        return result

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AnnAssign:
        node.annotation = self._stringify_annotation(node.annotation)
        result = self.generic_visit(node)
        assert isinstance(result, ast.AnnAssign)
        return result

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        node.returns = self._stringify_annotation(node.returns)
        result = self.generic_visit(node)
        assert isinstance(result, ast.FunctionDef)
        return result

    @staticmethod
    def _stringify_annotation(node: Optional[ast.AST]):
        return ast.Constant(value=ast.unparse(node), kind=None) if node else node
