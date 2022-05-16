# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
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

"""Definitions of Trait classes."""


from __future__ import annotations

import collections

import pydantic

from . import concepts, visitors
from .extended_typing import Any, Dict, Set, Type
from .type_definitions import SymbolName, SymbolRef


# ---  Node Traits ---
class SymbolTableTrait(concepts.Model):
    """Node trait adding an automatically created symbol table to the parent node.

    The actual symbol table dict will be stored in the `symtable_` attribute.
    To inject extra symbol definitions, add the nodes to a class attribute
    called ``_NODE_SYMBOLS_``.
    """

    symtable_: Dict[str, Any] = pydantic.Field(default_factory=dict)

    @pydantic.root_validator(skip_on_failure=True)
    def __collect_symbols(  # type: ignore  # validators are classmethods
        cls: Type[SymbolTableTrait], values: Dict[str, Any]
    ) -> Dict[str, Any]:
        values.pop("symtable_", None)
        values["symtable_"] = cls._CollectSymbols.apply(
            {**values, "_NODE_SYMBOLS_": getattr(cls, "_NODE_SYMBOLS_", {})}
        )
        return values

    class _CollectSymbols(visitors.NodeVisitor):
        def __init__(self) -> None:
            self.collected: Dict[str, Any] = {}

        def visit_Node(self, node: concepts.Node) -> None:
            for name, metadata in node.__node_children__.items():
                if isinstance(metadata["definition"].type_, type) and issubclass(
                    metadata["definition"].type_, SymbolName
                ):
                    symbol_name = getattr(node, name)
                    if symbol_name in self.collected:
                        raise ValueError(f"Multiple definitions of symbol '{symbol_name}'")
                    self.collected[symbol_name] = node
            if not isinstance(node, SymbolTableTrait):
                # don't recurse into a new scope (i.e. node with SymbolTableTrait)
                self.generic_visit(node)

        @classmethod
        def apply(cls, node: concepts.TreeNode) -> Dict[str, Any]:
            instance = cls()
            instance.generic_visit(node)
            return instance.collected


class SymbolRefsValidatorTrait(concepts.Model):
    """Node trait adding automatic validation of symbol references appearing the node tree.

    It assumes that the symbol table with the actual definitions is stored as
    a dict in the `symtable_` attribute (like :class:`SymbolTableTrait` does).
    """

    @pydantic.root_validator(skip_on_failure=True)
    def __validate_refs(  # type: ignore  # validators are classmethods
        cls: Type[SymbolRefsValidatorTrait], values: Dict[str, Any]
    ) -> Dict[str, Any]:
        validator = cls._SymbolRefsValidator()
        for v in values.values():
            validator.visit(v, symtable=values["symtable_"])

        if len(validator.missing_symbols) > 0:
            raise ValueError("Symbols {} not found.".format(validator.missing_symbols))

        return values

    class _SymbolRefsValidator(visitors.NodeVisitor):
        def __init__(self) -> None:
            self.missing_symbols: Set[str] = set()

        def visit_Node(
            self, node: concepts.Node, *, symtable: Dict[str, Any], **kwargs: Any
        ) -> None:
            for name, metadata in node.__node_children__.items():
                if isinstance(metadata["definition"].type_, type) and issubclass(
                    metadata["definition"].type_, SymbolRef
                ):
                    if getattr(node, name) and getattr(node, name) not in symtable:
                        self.missing_symbols.add(getattr(node, name))

            if isinstance(node, SymbolTableTrait):
                symtable = {**symtable, **node.symtable_}
            self.generic_visit(node, symtable=symtable, **kwargs)


class ValidatedSymbolTableTrait(SymbolRefsValidatorTrait, SymbolTableTrait):
    """Node trait adding an automatically created and validated symbol table.

    It is just the combination of the :class:`SymbolTableTrait` and
    :class:`SymbolRefsValidatorTrait` traits.
    """

    pass


# --- Visitor Traits ---
class VisitorWithSymbolTableTrait(visitors.NodeVisitor):
    """Visitor trait to update or add the symtable to kwargs in the visitor calls.

    Visitors inhering from this trait will automatically pass the active
    symbol table to visitor methods as the 'symtable' keyword argument.
    """

    def visit(self, node: concepts.TreeNode, **kwargs: Any) -> Any:
        kwargs.setdefault("symtable", collections.ChainMap())
        if has_table := isinstance(node, SymbolTableTrait):
            kwargs["symtable"] = kwargs["symtable"].new_child(node.symtable_)

        result = super().visit(node, **kwargs)

        if has_table:
            kwargs["symtable"] = kwargs["symtable"].parents

        return result
