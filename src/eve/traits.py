# Eve Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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
import contextlib

import pydantic

from . import concepts, visitors
from .type_definitions import SymbolName
from .typingx import Any, Dict, Iterator, Type


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


class SymbolTableTrait(concepts.Model):
    symtable_: Dict[str, Any] = pydantic.Field(default_factory=dict)

    @staticmethod
    def _collect_symbols(root_node: concepts.TreeNode) -> Dict[str, Any]:
        return _CollectSymbols.apply(root_node)

    @pydantic.root_validator(skip_on_failure=True)
    def _collect_symbols_validator(  # type: ignore  # validators are classmethods
        cls: Type[SymbolTableTrait], values: Dict[str, Any]
    ) -> Dict[str, Any]:
        values.pop("symtable_", None)
        values["symtable_"] = cls._collect_symbols(values)
        return values

    @staticmethod
    @contextlib.contextmanager
    def symtable_merger(
        node_visitor: visitors.NodeVisitor, node: concepts.Node, kwargs: Dict[str, Any]
    ) -> Iterator[None]:
        """Update or add the symtable to kwargs in the visitor calls.

        This is a context manager that, when included to the contexts classvar, will
        automatically pass 'symtable' as a keyword argument to visitor methods.
        """
        kwargs.setdefault("symtable", collections.ChainMap())
        if has_table := isinstance(node, SymbolTableTrait):
            kwargs["symtable"] = kwargs["symtable"].new_child(node.symtable_)

        yield

        if has_table:
            kwargs["symtable"] = kwargs["symtable"].parents
