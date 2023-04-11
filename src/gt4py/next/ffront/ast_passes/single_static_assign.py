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
import typing

from gt4py.next.ffront.fbuiltins import TYPE_BUILTIN_NAMES


def _make_assign(target: str, source: str, location_node: ast.AST):
    result = ast.Assign(
        targets=[ast.Name(ctx=ast.Store(), id=target)], value=ast.Name(ctx=ast.Load(), id=source)
    )
    for node in ast.walk(result):
        ast.copy_location(node, location_node)
    return result


def is_guaranteed_to_return(node: ast.stmt | list[ast.stmt]) -> bool:
    if isinstance(node, list):
        return any(is_guaranteed_to_return(child) for child in node)
    if isinstance(node, ast.Return):
        return True
    if isinstance(node, ast.If):
        return is_guaranteed_to_return(node.body) and is_guaranteed_to_return(node.orelse)
    return False


class Versioning:
    """Helper class to keep track of whether versioning (definedness)."""

    # invariant: if a version is an `int`, it's not negative
    _versions: dict[str, None | int]

    def __init__(self):
        self._versions = {}

    def define(self, name: str) -> None:
        if name not in self._versions:
            self._versions[name] = None

    def assign(self, name: str) -> None:
        if self.is_versioned(name):
            self._versions[name] = typing.cast(int, self._versions[name]) + 1
        else:
            self._versions[name] = 0

    def is_defined(self, name: str) -> bool:
        return name in self._versions

    def is_versioned(self, name: str) -> bool:
        return self.is_defined(name) and self._versions[name] is not None

    def __getitem__(self, name: str) -> None | int:
        return self._versions[name]

    def __iter__(self) -> typing.Iterator[tuple[str, None | int]]:
        return iter(self._versions.items())

    def copy(self) -> Versioning:
        copy = Versioning()
        copy._versions = {**self._versions}
        return copy

    @staticmethod
    def merge(a: Versioning, b: Versioning) -> Versioning:
        versions_a, version_b = a._versions, b._versions
        names = set(versions_a.keys()) & set(version_b.keys())

        merged_versioning = Versioning()
        merged_versions = merged_versioning._versions

        for name in names:
            merged_versions[name] = Versioning._merge_versions(versions_a[name], version_b[name])

        return merged_versioning

    @staticmethod
    def _merge_versions(a: None | int, b: None | int) -> None | int:
        if a is None:
            return b
        elif b is None:
            return a
        return max(a, b)


class NameEncoder:
    """Helper class to encode names of versioned variables."""

    _separator: str

    def __init__(self, separator: str):
        self._separator = separator

    def encode_name(self, name: str, versions: Versioning) -> str:
        if versions.is_versioned(name):
            return f"{name}{self._separator}{versions[name]}"
        return name


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

    class RhsRenamer(ast.NodeTransformer):
        """
        Rename right hand side names.

        Only read from parent visitor state, should not modify.
        """

        @classmethod
        def apply(cls, versioning: Versioning, name_encoder: NameEncoder, node: ast.AST):
            return cls(versioning, name_encoder).visit(node)

        def __init__(self, versioning: Versioning, name_encoder: NameEncoder):
            super().__init__()
            self.versioning: Versioning = versioning
            self.name_encoder: NameEncoder = name_encoder

        def visit_Name(self, node: ast.Name) -> ast.Name:
            node.id = self.name_encoder.encode_name(node.id, self.versioning)
            return node

    @classmethod
    def apply(cls, node: ast.AST) -> ast.AST:
        return cls().visit(node)

    def __init__(self, separator="__"):
        super().__init__()
        self.versioning: Versioning = Versioning()
        self.name_encoder: NameEncoder = NameEncoder(separator)

    def _rename(self, node: ast.AST):
        return self.RhsRenamer.apply(self.versioning, self.name_encoder, node)

    def visit_FunctionDef(self, node: ast.FunctionDef):

        # For practical purposes, this is sufficient, but really not general at all.
        # However, the algorithm was never intended to be general.

        old_versioning = self.versioning.copy()

        for arg in node.args.args:
            self.versioning.define(arg.arg)

        node.body = [self.visit(stmt) for stmt in node.body]

        self.versioning = old_versioning
        return node

    def visit_If(self, node: ast.If) -> ast.If:
        old_versioning = self.versioning

        node.test = self._rename(node.test)

        self.versioning = old_versioning.copy()
        node.body = [self.visit(el) for el in node.body]
        body_versioning = self.versioning
        body_returns = is_guaranteed_to_return(node.body)

        self.versioning = old_versioning.copy()
        node.orelse = [self.visit(el) for el in node.orelse]
        orelse_versioning = self.versioning
        orelse_returns = is_guaranteed_to_return(node.orelse)

        if body_returns and not orelse_returns:
            self.versioning = orelse_versioning
            return node

        if orelse_returns and not body_returns:
            self.versioning = body_versioning
            return node

        if body_returns and orelse_returns:
            self.versioning = Versioning()
            return node

        assert not body_returns and not orelse_returns

        self.versioning = Versioning.merge(body_versioning, orelse_versioning)

        # ensure both branches conclude with the same unique names
        for name, merged_version in self.versioning:

            body_version = body_versioning[name]
            orelse_version = orelse_versioning[name]

            if body_version != merged_version:
                new_assign = _make_assign(
                    self.name_encoder.encode_name(name, self.versioning),
                    self.name_encoder.encode_name(name, body_versioning),
                    node,
                )
                node.body.append(new_assign)
            elif orelse_version != merged_version:
                new_assign = _make_assign(
                    self.name_encoder.encode_name(name, self.versioning),
                    self.name_encoder.encode_name(name, orelse_versioning),
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
            old_versioning = self.versioning.copy()
            node.target = self.visit(node.target)
            self.versioning = old_versioning
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id in TYPE_BUILTIN_NAMES:
            return node

        self.versioning.assign(node.id)
        node.id = self.name_encoder.encode_name(node.id, self.versioning)
        return node
