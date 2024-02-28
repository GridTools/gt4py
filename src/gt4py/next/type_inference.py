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

import typing
from collections import abc

import gt4py.eve as eve
from gt4py.eve import datamodels
from gt4py.eve.utils import noninstantiable


"""Customizable constraint-based inference.

Based on the classical constraint-based two-pass type consisting of the following passes:
    1. Constraint collection
    2. Type unification
"""


V = typing.TypeVar("V", bound="TypeVar")
T = typing.TypeVar("T", bound="Type")


@noninstantiable
class Type(eve.Node, unsafe_hash=True):  # type: ignore[call-arg]
    """Base class for all types.

    The initial type constraint collection pass treats all instances of Type as hashable frozen
    nodes, that is, no in-place modification is used.

    In the type unification phase however, in-place modifications are used for efficient
    renaming/node replacements and special care is taken to handle hash values that change due to
    those modifications.
    """

    def handle_constraint(
        self, other: Type, add_constraint: abc.Callable[[Type, Type], None]
    ) -> bool:
        """Implement special type-specific constraint handling for `self` ≡ `other`.

        New constraints can be added using the provided callback (`add_constraint`). Should return
        `True` if the provided constraint `self` ≡ `other` was handled, `False` otherwise. If the
        handler detects an unsatisfiable constraint, raise a `TypeError`.
        """
        return False


class TypeVar(Type):
    """Type variable."""

    idx: int

    _counter: typing.ClassVar[int] = 0

    @staticmethod
    def fresh_index() -> int:
        TypeVar._counter += 1
        return TypeVar._counter

    @classmethod
    def fresh(cls: type[V], **kwargs: typing.Any) -> V:
        """Create a type variable with a previously unused index."""
        return cls(idx=cls.fresh_index(), **kwargs)


class _TypeVarReindexer(eve.NodeTranslator):
    """Reindex type variables in a type tree."""

    def __init__(self, indexer: abc.Callable[[dict[int, int]], int]):
        super().__init__()
        self.indexer = indexer

    def visit_TypeVar(self, node: V, *, index_map: dict[int, int]) -> V:
        node = self.generic_visit(node, index_map=index_map)
        new_index = index_map.setdefault(node.idx, self.indexer(index_map))
        new_values = {
            typing.cast(str, k): (new_index if k == "idx" else v)
            for k, v in node.iter_children_items()
        }
        return node.__class__(**new_values)


@typing.overload
def freshen(dtypes: list[T]) -> list[T]: ...


@typing.overload
def freshen(dtypes: T) -> T: ...


def freshen(dtypes: list[T] | T) -> list[T] | T:
    """Re-instantiate `dtype` with fresh type variables."""
    if not isinstance(dtypes, list):
        assert isinstance(dtypes, Type)
        return freshen([dtypes])[0]

    def indexer(index_map: dict[int, int]) -> int:
        return TypeVar.fresh_index()

    index_map = dict[int, int]()
    return [_TypeVarReindexer(indexer).visit(dtype, index_map=index_map) for dtype in dtypes]


def reindex_vars(dtypes: typing.Any) -> typing.Any:
    """Reindex all type variables, to have nice indices starting at zero."""

    def indexer(index_map: dict[int, int]) -> int:
        return len(index_map)

    index_map = dict[int, int]()
    return _TypeVarReindexer(indexer).visit(dtypes, index_map=index_map)


class _FreeVariables(eve.NodeVisitor):
    """Collect type variables within a type expression."""

    def visit_TypeVar(self, node: TypeVar, *, free_variables: set[TypeVar]) -> None:
        self.generic_visit(node, free_variables=free_variables)
        free_variables.add(node)


def _free_variables(x: Type) -> set[TypeVar]:
    """Collect type variables within a type expression."""
    fv = set[TypeVar]()
    _FreeVariables().visit(x, free_variables=fv)
    return fv


class _Dedup(eve.NodeTranslator):
    """Deduplicate type nodes that have the same value but a different `id`."""

    def visit(self, node, *, memo: dict[T, T]) -> typing.Any:  # type: ignore[override]
        if isinstance(node, Type):
            node = super().visit(node, memo=memo)
            return memo.setdefault(node, node)
        return node


def _assert_constituent_types(value: typing.Any, allowed_types: tuple[type, ...]) -> None:
    if isinstance(value, tuple):
        for el in value:
            _assert_constituent_types(el, allowed_types)
    else:
        assert isinstance(value, allowed_types)


class _Renamer:
    """Efficiently rename (that is, replace) nodes in a type expression.

    Works by collecting all parent nodes of all nodes in a tree. If a node should be replaced by
    another, all referencing parent nodes can be found efficiently and modified in place.

    Note that all types have to be registered before they can be used in a `rename` call.

    Besides basic renaming, this also resolves `ValTuple` to full `Tuple` if possible after
    renaming.
    """

    def __init__(self) -> None:
        self._parents = dict[Type, list[tuple[Type, str]]]()

    def register(self, dtype: Type) -> None:
        """Register a type for possible future renaming.

        Collects the parent nodes of all nodes in the type tree.
        """

        def collect_parents(node: Type) -> None:
            for field, child in node.iter_children_items():
                if isinstance(child, Type):
                    self._parents.setdefault(child, []).append((node, typing.cast(str, field)))
                    collect_parents(child)
                else:
                    _assert_constituent_types(child, (int, str))

        collect_parents(dtype)

    def _update_node(self, node: Type, field: str, replacement: Type) -> None:
        """Replace a field of a node by some other value.

        Basically performs `setattr(node, field, replacement)`. Further, updates the mapping of node
        parents and handles the possibly changing hash value of the updated node.
        """
        # Pop the node out of the parents dict as its hash could change after modification
        popped = self._parents.pop(node, None)

        # Update the node's field
        setattr(node, field, replacement)

        # Register `node` to be the new parent of `replacement`
        self._parents.setdefault(replacement, []).append((node, field))

        # Put back possible previous entries to the parents dict after possible hash change
        if popped:
            self._parents[node] = popped

    def rename(self, node: Type, replacement: Type) -> None:
        """Rename/replace all occurrences of `node` to/by `replacement`."""
        try:
            # Find parent nodes
            nodes = self._parents.pop(node)
        except KeyError:
            return

        for node, field in nodes:
            # Default case: just update a field value of the node
            self._update_node(node, field, replacement)


class _Box(Type):
    """Simple value holder, used for wrapping root nodes of a type tree.

    This makes sure that all root nodes have a parent node which can be updated by the `_Renamer`.
    """

    value: Type


class _Unifier:
    """A classical type unifier (Robinson, 1971).

    Computes the most general type satisfying all given constraints. Uses a `_Renamer` for efficient
    type variable renaming.
    """

    def __init__(self, dtypes: list[Type], constraints: set[tuple[Type, Type]]) -> None:
        # Wrap the original `dtype` and all `constraints` to make sure they have a parent node and
        # thus the root nodes are correctly handled by the renamer
        self._dtypes = [_Box(value=dtype) for dtype in dtypes]
        self._constraints = [(_Box(value=s), _Box(value=t)) for s, t in constraints]

        # Create a renamer and register `dtype` and all `constraints` types
        self._renamer = _Renamer()
        for dtype in self._dtypes:
            self._renamer.register(dtype)
        for s, t in self._constraints:
            self._renamer.register(s)
            self._renamer.register(t)

    def unify(self) -> tuple[list[Type] | Type, list[tuple[Type, Type]]]:
        """Run the unification."""
        unsatisfiable_constraints = []
        while self._constraints:
            constraint = self._constraints.pop()
            try:
                handled = self._handle_constraint(constraint)
                if not handled:
                    # Try with swapped LHS and RHS
                    handled = self._handle_constraint(constraint[::-1])
            except TypeError:
                # custom constraint handler raised an error as constraint is not satisfiable
                # (contrary to just not handled)
                handled = False

            if not handled:
                unsatisfiable_constraints.append((constraint[0].value, constraint[1].value))

        unboxed_dtypes = [dtype.value for dtype in self._dtypes]

        return unboxed_dtypes, unsatisfiable_constraints

    def _rename(self, x: Type, y: Type) -> None:
        """Type renaming/replacement."""
        self._renamer.register(x)
        self._renamer.register(y)
        self._renamer.rename(x, y)

    def _add_constraint(self, x: Type, y: Type) -> None:
        """Register a new constraint."""
        x = _Box(value=x)
        y = _Box(value=y)
        self._renamer.register(x)
        self._renamer.register(y)
        self._constraints.append((x, y))

    def _handle_constraint(self, constraint: tuple[_Box, _Box]) -> bool:
        """Handle a single constraint."""
        s, t = (c.value for c in constraint)
        if s == t:
            # Constraint is satisfied if LHS equals RHS
            return True

        if type(s) is TypeVar:
            assert s not in _free_variables(t)
            # Just replace LHS by RHS if LHS is a type variable
            self._rename(s, t)
            return True

        if s.handle_constraint(t, self._add_constraint):
            # Use a custom constraint handler if available
            return True

        if type(s) is type(t):
            assert s not in _free_variables(t) and t not in _free_variables(s)
            assert datamodels.fields(s).keys() == datamodels.fields(t).keys()
            for k in datamodels.fields(s).keys():
                sv = getattr(s, k)
                tv = getattr(t, k)
                if isinstance(sv, Type):
                    assert isinstance(tv, Type)
                    self._add_constraint(sv, tv)
                else:
                    assert sv == tv
            return True

        # Constraint handling failed
        return False


@typing.overload
def unify(
    dtypes: list[Type], constraints: set[tuple[Type, Type]]
) -> tuple[list[Type], list[tuple[Type, Type]]]: ...


@typing.overload
def unify(
    dtypes: Type, constraints: set[tuple[Type, Type]]
) -> tuple[Type, list[tuple[Type, Type]]]: ...


def unify(
    dtypes: list[Type] | Type, constraints: set[tuple[Type, Type]]
) -> tuple[list[Type] | Type, list[tuple[Type, Type]]]:
    """
    Unify all given constraints.

    Returns the unified types and a list of unsatisfiable constraints.
    """
    if isinstance(dtypes, Type):
        result_types, unsatisfiable_constraints = unify([dtypes], constraints)
        return result_types[0], unsatisfiable_constraints

    # Deduplicate type nodes, this can speed up later things a bit
    memo = dict[Type, Type]()
    dtypes = [_Dedup().visit(dtype, memo=memo) for dtype in dtypes]
    constraints = {_Dedup().visit(c, memo=memo) for c in constraints}
    del memo

    unifier = _Unifier(dtypes, constraints)
    return unifier.unify()
