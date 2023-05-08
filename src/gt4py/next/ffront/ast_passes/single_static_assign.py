# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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
from __future__ import annotations

import ast
import dataclasses
import typing

from gt4py.next.ffront.fbuiltins import TYPE_BUILTIN_NAMES


ASTNodeT = typing.TypeVar("ASTNodeT", bound=ast.AST)

_UNIQUE_NAME_SEPERATOR = "ᐞ"


def unique_name(name: str, num_assignments: int):
    """Given the name of a variable return a unique name after the given amount of assignments."""
    if num_assignments >= 0:
        return f"{name}{_UNIQUE_NAME_SEPERATOR}{num_assignments}"
    assert num_assignments == -1
    return name


def _make_assign(target: str, source: str, location_node: ast.AST) -> ast.Assign:
    result = ast.Assign(
        targets=[ast.Name(ctx=ast.Store(), id=target)], value=ast.Name(ctx=ast.Load(), id=source)
    )
    for node in ast.walk(result):
        ast.copy_location(node, location_node)
    return result


def _is_guaranteed_to_return(node: ast.stmt | list[ast.stmt]) -> bool:
    if isinstance(node, list):
        return any(_is_guaranteed_to_return(child) for child in node)
    if isinstance(node, ast.Return):
        return True
    if isinstance(node, ast.If):
        return _is_guaranteed_to_return(node.body) and _is_guaranteed_to_return(node.orelse)
    return False


@dataclasses.dataclass
class _AssignmentTracker:
    """Helper class to keep track of the number of assignments to a variable."""

    _counts: dict[str, int] = dataclasses.field(default_factory=dict)

    def define(self, name: str) -> None:
        if name in self.names():
            raise ValueError(f"Variable {name} is already defined.")
        self._counts[name] = -1

    def assign(self, name: str) -> None:
        self._counts[name] = self.count(name) + 1

    def count(self, name: str):
        return self._counts.get(name, -1)

    def names(self):
        return self._counts.keys()

    def copy(self):
        return _AssignmentTracker({**self._counts})


def _merge_assignment_tracker(a: _AssignmentTracker, b: _AssignmentTracker) -> _AssignmentTracker:
    common_names = set(a.names()) & set(b.names())
    return _AssignmentTracker({k: max(a.count(k), b.count(k)) for k in common_names})


@dataclasses.dataclass
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
        * While loops aren't supported
    """
    assignment_tracker: _AssignmentTracker = dataclasses.field(default_factory=_AssignmentTracker)

    @classmethod
    def apply(cls, node: ASTNodeT) -> ASTNodeT:
        return cls().visit(node)

    def _rename(self, node: ASTNodeT) -> ASTNodeT:
        for child_node in ast.walk(node):
            if isinstance(child_node, ast.Name):
                child_node.id = unique_name(
                    child_node.id, self.assignment_tracker.count(child_node.id)
                )
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        # For practical purposes, this is sufficient, but really not general at all.
        # However, the algorithm was never intended to be general.

        old_versioning = self.assignment_tracker.copy()

        for arg in node.args.args:
            self.assignment_tracker.define(arg.arg)

        node.body = [self.visit(stmt) for stmt in node.body]

        self.assignment_tracker = old_versioning
        return node

    def visit_If(self, node: ast.If) -> ast.If:
        old_versioning = self.assignment_tracker

        node.test = self._rename(node.test)

        self.assignment_tracker = old_versioning.copy()
        node.body = [self.visit(el) for el in node.body]
        body_assignment_tracker = self.assignment_tracker
        body_returns = _is_guaranteed_to_return(node.body)

        self.assignment_tracker = old_versioning.copy()
        node.orelse = [self.visit(el) for el in node.orelse]
        orelse_assignment_tracker = self.assignment_tracker
        orelse_returns = _is_guaranteed_to_return(node.orelse)

        if body_returns and not orelse_returns:
            self.assignment_tracker = orelse_assignment_tracker
            return node

        if orelse_returns and not body_returns:
            self.assignment_tracker = body_assignment_tracker
            return node

        if body_returns and orelse_returns:
            self.assignment_tracker = _AssignmentTracker()
            return node

        assert not body_returns and not orelse_returns

        self.assignment_tracker = _merge_assignment_tracker(
            body_assignment_tracker, orelse_assignment_tracker
        )

        # ensure both branches conclude with the same unique names
        for name in self.assignment_tracker.names():
            assignment_count = self.assignment_tracker.count(name)
            body_assignment_count = body_assignment_tracker.count(name)
            orelse_assignment_count = orelse_assignment_tracker.count(name)

            if body_assignment_count != assignment_count:
                new_assign = _make_assign(
                    unique_name(name, assignment_count),
                    unique_name(name, body_assignment_count),
                    node,
                )
                node.body.append(new_assign)
            elif orelse_assignment_count != assignment_count:
                new_assign = _make_assign(
                    unique_name(name, assignment_count),
                    unique_name(name, orelse_assignment_count),
                    node,
                )
                node.orelse.append(new_assign)

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
            # An empty annotation always applies to the next assignment.
            # So we need to use the correct versioning, but also ensure
            # we restore the old versioning afterwards, because no assignment
            # actually happens.
            old_versioning = self.assignment_tracker.copy()
            node.target = self.visit(node.target)
            self.assignment_tracker = old_versioning
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id in TYPE_BUILTIN_NAMES:
            return node

        self.assignment_tracker.assign(node.id)
        node.id = unique_name(node.id, self.assignment_tracker.count(node.id))
        return node
