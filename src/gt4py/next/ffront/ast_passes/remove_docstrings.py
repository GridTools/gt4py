# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

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
    ...
    ...     def example_docstring_2():
    ...         a = 2.0
    ...         "This is a new docstring"
    ...         return a
    ...
    ...     a = example_docstring_2()
    ...     return a
    >>> print(ast.unparse(RemoveDocstrings.apply(ast.parse(inspect.getsource(example_docstring)))))
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
        node.body = [obj for obj in node.body if not _is_const_str_expr(obj)]
        return self.generic_visit(node)


def _is_const_str_expr(obj: ast.stmt) -> bool:
    return (
        isinstance(obj, ast.Expr)
        and isinstance(obj.value, ast.Constant)
        and isinstance(obj.value.value, str)
    )
