# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import ast
from typing import Any, Optional


class StringifyAnnotationsPass(ast.NodeTransformer):
    """
    AST pass transforming type annotations to string :class:`ast.Constant`.

    Mutates the python AST replacing arbitrary AST subtrees in the annotation
    fields of the supported nodes with its string representation
    using :func:`ast.unparse()`.


    Example
    -------
    >>> import ast, inspect
    >>> from typing import Union

    >>> def foo(a: Union[int, float]) -> float:
    ...     tmp: Union[int, float] = a + 1
    ...     result: float = float(tmp)
    ...     return result

    >>> ast_node = ast.parse(inspect.getsource(foo))
    >>> print(
    ...     f"_ {ast_node.body[0].args.args[0].annotation}"
    ... )  # first character of output can not be ignored
    _ ...Subscript...

    >>> ast_node = StringifyAnnotationsPass.apply(ast_node)
    >>> print(f"_ {ast_node.body[0].args.args[0].annotation}")
    _ ...Constant...
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
    def _stringify_annotation(node: Optional[ast.AST]) -> Any:
        return ast.Constant(value=ast.unparse(node), kind=None) if node else node
