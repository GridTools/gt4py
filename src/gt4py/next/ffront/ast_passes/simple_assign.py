# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import ast
import copy
from collections.abc import Iterator


class NodeYielder(ast.NodeTransformer):
    @classmethod
    def apply(cls, node: ast.AST) -> ast.AST:
        result = list(cls().visit(node))
        if len(result) != 1:
            raise ValueError("AST was split or lost during the pass, use '.visit()' instead.")
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
    >>> print(ast.unparse(SingleAssignTargetPass.apply(ast.parse(inspect.getsource(foo)))))
    def foo():
        __sat_tmp0 = 1
        a = __sat_tmp0
        b = __sat_tmp0
        return (a, b)

    """

    unique_name_id: int = 0

    def _unique_symbol_name(self) -> ast.Name:
        name = ast.Name(id=f"__sat_tmp{self.unique_name_id}", ctx=ast.Store())
        self.unique_name_id += 1
        return name

    def visit_Assign(self, node: ast.Assign) -> Iterator[ast.Assign]:
        # ignore regular assignments
        if len(node.targets) == 1:
            yield node
            return

        synthetic_target = self._unique_symbol_name()
        ast.copy_location(synthetic_target, node)
        synthetic_assign = copy.copy(node)
        synthetic_assign.targets = [synthetic_target]
        yield synthetic_assign

        for target in node.targets:
            new_assign = copy.copy(node)
            new_assign.targets = [target]
            new_assign.value = ast.Name(id=synthetic_target.id, ctx=ast.Load())
            yield new_assign
