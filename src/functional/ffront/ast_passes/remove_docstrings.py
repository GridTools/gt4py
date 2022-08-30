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
import dataclasses


@dataclasses.dataclass(kw_only=True)
class RemoveDocstrings(ast.NodeTransformer):
    """
    Remove ast docstring from body of ast.FunctionDef.

    Docstrings appear as type ast.Expr with ast.Constant value of type string.
    If such patterns is detected, this entry in the node.body list is removed.

    Example
    -------
    >>> import ast, inspect

    >>> def example_docstring():
    ...     a = 1
    ...     "This is a docstring"
    ...     return a

    This will return an ast.FunctionDef containing elements in the body of type:
     ast.Assign, ast.Expr, and ast.Return. The docstring is the second.
    This element in the body is then removed.

    >>> print(ast.unparse(
    ...     RemoveDocstrings.apply(
    ...         ast.parse(inspect.getsource(example_docstring))
    ...     )
    ... ))
    def example_docstring():
        a = 1
        return a
    """

    @classmethod
    def apply(cls, node):
        return cls().visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        for obj in node.body:
            if (
                isinstance(obj, ast.Expr)
                and isinstance(obj.value, ast.Constant)
                and isinstance(obj.value.value, str)
            ):
                node.body.remove(obj)

        return ast.FunctionDef(
            name=node.name,
            args=node.args,
            body=[self.visit(obj) for obj in node.body],
            decorator_list=node.decorator_list,
            returns=node.returns,
            type_comment=node.type_comment,
            lineno=node.lineno,
            col_offset=node.col_offset,
        )
