# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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
import dataclasses


@dataclasses.dataclass(kw_only=True)
class RemoveDocstrings(ast.NodeTransformer):
    """
    Remove function docstrings from an AST.

    Example
    -------
    >>> import ast, inspect
    >>> def example_docstring():
    ...     a = 1
    ...     "This is a docstring"
    ...     def example_docstring_2():
    ...          a = 2.0
    ...          "This is a new docstring"
    ...          return a
    ...     a = example_docstring_2()
    ...     return a
    >>> print(ast.unparse(
    ...     RemoveDocstrings.apply(
    ...         ast.parse(inspect.getsource(example_docstring))
    ...     )
    ... ))
    def example_docstring():
        a = 1
    <BLANKLINE>
        def example_docstring_2():
            a = 2.0
            return a
        a = example_docstring_2()
        return a
    """

    @classmethod
    def apply(cls, node: ast.AST) -> ast.AST:
        return cls().visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        for obj in node.body:
            if (
                isinstance(obj, ast.Expr)
                and isinstance(obj.value, ast.Constant)
                and isinstance(obj.value.value, str)
            ):
                node.body.remove(obj)

        return self.generic_visit(node)
