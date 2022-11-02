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
import copy
from collections.abc import Iterator


class NodeYielder(ast.NodeTransformer):
    @classmethod
    def apply(cls, node: ast.AST) -> ast.AST:
        result = list(cls().visit(node))
        if len(result) != 1:
            raise ValueError("AST was split or lost during the pass. Use `.visit()` instead.")
        return result[0]

    def visit(self, node: ast.AST) -> Iterator[ast.AST]:
        result = super().visit(node)
        if isinstance(result, ast.AST):
            yield result
        else:
            yield from result

    def generic_visit(self, node: ast.AST) -> Iterator[ast.AST]:  # type: ignore[override]
        """Override generic visit to deal with generators."""
        for field, old_value in ast.iter_fields(node):
            if isinstance(old_value, list):
                new_values = [i for j in old_value for i in self.visit(j)]
                old_value[:] = new_values
            elif isinstance(old_value, ast.AST):
                new_node, *_ = list(self.visit(old_value)) or (None,)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        yield node


class SingleAssignTargetPass(NodeYielder):
    """
    Split multi target assignments.

    Requires AST in SSA form (see ``SingleStaticAssignPass``) to yield correct results.

    Example
    -------
    >>> import ast, inspect
    >>>
    >>> def foo():
    ...     a = b = 1
    ...     return a, b
    >>>
    >>> print(ast.unparse(
    ...     SingleAssignTargetPass.apply(
    ...         ast.parse(inspect.getsource(foo))
    ...     )
    ... ))
    def foo():
        a = 1
        b = 1
        return (a, b)

    """

    def visit_Assign(self, node: ast.Assign) -> Iterator[ast.Assign]:
        for target in node.targets:
            new_assign = copy.copy(node)
            new_assign.targets = [target]
            yield new_assign
