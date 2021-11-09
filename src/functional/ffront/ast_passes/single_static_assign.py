#!/usr/bin/env python
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
from dataclasses import dataclass, field


class SingleStaticAssignPass(ast.NodeTransformer):
    """
    Rename variables in assignments to avoid overwriting.

    Mutates the python AST, variable names will not be valid python names anymore.
    This pass must be run before any passes that linearize unpacking assignments.


    Example
    -------
    Function ``foo()`` in the following example keeps overwriting local variable ``a``

    >>> import ast
    >>> from functional.ffront.func_to_foir import get_ast_from_func

    >>> def foo():
    ...     a = 1
    ...     a = 2 + a
    ...     a = 3 + a
    ...     return a

    >>> print(ast.unparse(
    ...     SingleStaticAssignPass.apply(
    ...         get_ast_from_func(foo)
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

    class RhsRenamer(ast.NodeTransformer):
        """
        Rename right hand side names.

        Only read from parent visitor state, should not modify.
        """

        def __init__(self, state):
            super().__init__()
            self.state = state

        def visit_Name(self, node: ast.Name) -> ast.Name:
            if node.id in self.state.name_counter:
                node.id = f"{node.id}${self.state.name_counter[node.id]}"
            return node

    @dataclass
    class State:
        name_counter: dict[str, int] = field(default_factory=dict)

    @classmethod
    def apply(cls, node: ast.AST) -> ast.AST:
        return cls().visit(node)

    def __init__(self):
        super().__init__()
        self.state = self.State()

    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        # first update rhs names to reference the latest version
        node.value = self.RhsRenamer(self.state).visit(node.value)
        # then update lhs to create new names
        node.targets = [self.visit(target) for target in node.targets]
        return node

    def visit_Return(self, node: ast.Return) -> ast.Return:
        node.value = self.RhsRenamer(self.state).visit(node.value) if node.value else None
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AnnAssign:
        if node.value:
            node.value = self.RhsRenamer(self.state).visit(node.value)
            node.target = self.visit(node.target)
        elif isinstance(node.target, ast.Name):
            target_id = node.target.id
            node.target = self.visit(node.target)
            self.state.name_counter[target_id] -= 1
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id in self.state.name_counter:
            self.state.name_counter[node.id] += 1
            node.id = f"{node.id}${self.state.name_counter[node.id]}"
        else:
            self.state.name_counter[node.id] = 0
            node.id = f"{node.id}$0"
        return node
