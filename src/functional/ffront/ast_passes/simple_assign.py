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
        if len(node.targets) > 1:
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
        else:
            yield node


class UnpackedAssignPass(NodeYielder):
    """
    Explicitly unpack assignments.

    Requires AST in SSA form and assumes only single target assigns, check the following passes.

     * ``SingleStaticAssignPass``
     * ``SingleAssignTargetPass``

    Example
    -------
    >>> import ast, inspect

    >>> def foo():
    ...     a0 = 1
    ...     b0 = 5
    ...     a1, b1 = b0, a0
    ...     return a1, b1

    >>> print(ast.unparse(
    ...     UnpackedAssignPass.apply(
    ...         ast.parse(inspect.getsource(foo))
    ...     )
    ... ))
    def foo():
        a0 = 1
        b0 = 5
        __tuple_tmp_0 = (b0, a0)
        a1 = __tuple_tmp_0[0]
        b1 = __tuple_tmp_0[1]
        return (a1, b1)

    which would not have been equivalent had the input AST not been in SSA form.
    """

    unique_name_id: int = 0

    def visit_Assign(self, node: ast.Assign) -> Iterator[ast.Assign]:
        if len(node.targets) != 1:
            raise ValueError(
                "AST contains multi target assigns. Please run `SingleAssignTargetPass` first."
            )
        if isinstance(target := node.targets[0], (ast.Tuple, ast.List)):
            yield from self._unpack_assignment(node, targets=target.elts)
        else:
            yield node

    def _unique_tuple_name(self) -> ast.Name:
        name = ast.Name(id=f"__tuple_tmp_{self.unique_name_id}", ctx=ast.Store())
        self.unique_name_id += 1
        return name

    def _unpack_assignment(
        self, node: ast.Assign, *, targets: list[ast.expr]  # targets passed here for typing
    ) -> Iterator[ast.Assign]:
        tuple_name = self._unique_tuple_name()
        tuple_assign = ast.Assign(targets=[tuple_name], value=node.value)
        ast.copy_location(tuple_name, node)
        ast.copy_location(tuple_assign, node)
        yield from self.visit_Assign(tuple_assign)

        for index, subtarget in enumerate(targets):
            new_assign = copy.copy(node)
            new_assign.targets = [subtarget]
            new_assign.value = ast.Subscript(
                ctx=ast.Load(),  # <- ctx is mandatory for ast.Subscript, Load() for rhs.
                value=tuple_name,
                slice=ast.Constant(value=index),
            )
            ast.copy_location(new_assign.value, node.value)
            ast.copy_location(new_assign.value.slice, node.value)
            yield from self.visit_Assign(new_assign)
