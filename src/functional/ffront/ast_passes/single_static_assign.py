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

from functional.ffront.fbuiltins import TYPE_BUILTIN_NAMES


def _make_assign(target: str, source: str, location_node: ast.AST):
    result = ast.Assign(
        targets=[ast.Name(ctx=ast.Store(), id=target)], value=ast.Name(ctx=ast.Load(), id=source)
    )
    for node in ast.walk(result):
        ast.copy_location(node, location_node)
    return result


class SingleStaticAssignPass(ast.NodeTransformer):
    """
    Rename variables in assignments to avoid overwriting.

    Mutates the python AST, variable names will not be valid python names anymore.
    This pass must be run before any passes that linearize unpacking assignments.

    Example
    -------
    Function ``foo()`` in the following example keeps overwriting local variable ``a``

    >>> import ast, inspect

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
        a__0 = 1
        a__1 = 2 + a__0
        a__2 = 3 + a__1
        return a__2

    Note that each variable name is assigned only once (per branch) and never updated / overwritten.

    Note also that after parsing, running the pass and unparsing we get invalid but
    readable python code. This is ok because this pass is not intended for
    python-to-python translation.

    WARNING: This pass is not intended as a general-purpose SSA transformation.
    The pass does not support any general Python AST. Known limitations include:
        * Nested functions aren't supported
        * Returns inside if blocks aren't supported
        * While loops aren't supported
    """

    class RhsRenamer(ast.NodeTransformer):
        """
        Rename right hand side names.

        Only read from parent visitor state, should not modify.
        """

        @classmethod
        def apply(cls, name_counter, separator, node):
            return cls(name_counter, separator).visit(node)

        def __init__(self, name_counter, separator):
            super().__init__()
            self.name_counter: dict[str, int] = name_counter
            self.separator: str = separator

        def visit_Name(self, node: ast.Name) -> ast.Name:
            if node.id in self.name_counter:
                node.id = f"{node.id}{self.separator}{self.name_counter[node.id]}"
            return node

    @classmethod
    def apply(cls, node: ast.AST) -> ast.AST:
        return cls().visit(node)

    def __init__(self, separator="__"):
        super().__init__()
        self.name_counter: dict[str, int] = {}
        self.separator: str = separator

    def _rename(self, node):
        return self.RhsRenamer.apply(self.name_counter, self.separator, node)

    def visit_FunctionDef(self, node: ast.FunctionDef):

        # For practical purposes, this is sufficient, but really not general at all.
        # However, the algorithm was never intended to be general.

        pre_assigns: list[ast.stmt] = [
            _make_assign(arg.arg, arg.arg, arg) for arg in node.args.args
        ]
        # This forces versioning for parameters which is required for if-stmt support
        node.body = pre_assigns + node.body
        node.body = [self.visit(stmt) for stmt in node.body]

        return node

    def visit_If(self, node: ast.If):
        prev_name_counter = self.name_counter

        node.test = self._rename(node.test)

        self.name_counter = {**prev_name_counter}
        node.body = [self.visit(el) for el in node.body]
        body_name_counter = self.name_counter

        self.name_counter = {**prev_name_counter}
        node.orelse = [self.visit(el) for el in node.orelse]
        orelse_name_counter = self.name_counter

        all_names = set(body_name_counter.keys()) & set(orelse_name_counter.keys())
        # We can't continue with the name counter of the else branch,
        # in case a variable was only defined in the else branch
        self.name_counter = {}

        # ensure both branches conclude with the same unique names
        for name in all_names:
            body_count = body_name_counter[name]
            orelse_count = orelse_name_counter[name]

            if body_count < orelse_count:
                new_assign = _make_assign(
                    f"{name}{self.separator}{orelse_count}",
                    f"{name}{self.separator}{body_count}",
                    node,
                )
                node.body.append(new_assign)
            elif body_count > orelse_count:
                new_assign = _make_assign(
                    f"{name}{self.separator}{body_count}",
                    f"{name}{self.separator}{orelse_count}",
                    node,
                )
                node.orelse.append(new_assign)

            self.name_counter[name] = max(body_count, orelse_count)

        return node

    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        # first update rhs names to reference the latest version
        node.value = self._rename(node.value)
        # then update lhs to create new names
        node.targets = [self.visit(target) for target in node.targets]
        return node

    def visit_Return(self, node: ast.Return) -> ast.Return:
        node.value = self._rename(node.value) if node.value else None
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AnnAssign:
        if node.value:
            node.value = self._rename(node.value)
            node.target = self.visit(node.target)
        elif isinstance(node.target, ast.Name):
            target_id = node.target.id
            node.target = self.visit(node.target)
            self.name_counter[target_id] -= 1
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id in TYPE_BUILTIN_NAMES:
            return node
        elif node.id in self.name_counter:
            self.name_counter[node.id] += 1
            node.id = f"{node.id}{self.separator}{self.name_counter[node.id]}"
        else:
            self.name_counter[node.id] = 0
            node.id = f"{node.id}{self.separator}0"
        return node
