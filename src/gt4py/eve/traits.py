# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Definitions of node and visitor trait classes."""

from __future__ import annotations

import collections

from . import concepts, datamodels, exceptions, visitors
from .extended_typing import Any, Dict, Set, Type, no_type_check


# ---  Node Traits ---
@concepts.register_annex_user("symtable", Dict[str, concepts.Node], shared=True)
@datamodels.datamodel
class SymbolTableTrait:
    """
    Node trait adding an automatically created symbol table to the parent node.

    The actual symbol table dict will be stored in the `annex.symtable` attribute.
    To inject extra symbol definitions, add the nodes to a class attribute
    called ``_NODE_SYMBOLS_``.
    """

    __slots__ = ()

    @no_type_check
    @datamodels.root_validator
    @classmethod
    def _collect_symbol_names(cls: Type[SymbolTableTrait], instance: concepts.Node) -> None:
        collected_symbols = cls.SymbolsCollector.apply(instance)
        instance.annex.symtable = collected_symbols

    class SymbolsCollector(visitors.NodeVisitor):
        def __init__(self) -> None:
            self.collected_symbols: Dict[str, concepts.Node] = {}

        def visit_Node(self, node: concepts.Node, /) -> None:
            for field_name, attribute in node.__datamodel_fields__.items():
                if isinstance(attribute.type, type) and issubclass(
                    attribute.type, concepts.SymbolName
                ):
                    symbol_name = getattr(node, field_name)
                    if symbol_name in self.collected_symbols:
                        raise exceptions.EveValueError(
                            f"Multiple definitions of symbol '{symbol_name}'"
                        )
                    self.collected_symbols[symbol_name] = node

            # TODO(egparedes): revisit and generalize mechanism to resolve name collisions.
            if hasattr(node.annex, "propagated_symbols"):
                # ensure we have no collisions
                assert not set(self.collected_symbols.keys()) & set(
                    node.annex.propagated_symbols.keys()
                )
                self.collected_symbols = {**self.collected_symbols, **node.annex.propagated_symbols}
            elif not isinstance(node, SymbolTableTrait):
                # Stop recursion if the node opens a new scope (i.e. node with SymbolTableTrait)
                self.generic_visit(node)

        def visit(self, node: concepts.RootNode, **kwargs: Any) -> Any:
            if hasattr(node.__class__, "_NODE_SYMBOLS_"):
                self.visit(node.__class__._NODE_SYMBOLS_)
            return super().visit(node, **kwargs)

        @classmethod
        def apply(cls, node: concepts.Node) -> Dict[str, concepts.Node]:
            collector = cls()
            # If the passed root node already contains a symbol table, the check in `visit_Node()`
            # will automatically stop the traversal. To avoid this premature stop, we start the
            # traversal here calling `generic_visit()` to directly inspect the children (after
            # adding any extra node symbols defined in the node class).
            if hasattr(node.__class__, "_NODE_SYMBOLS_"):
                collector.visit(node.__class__._NODE_SYMBOLS_)
            collector.generic_visit(node)
            return collector.collected_symbols


@concepts.register_annex_user("symtable", Dict[str, concepts.Node], shared=True)
@datamodels.datamodel
class SymbolRefsValidatorTrait:
    """Node trait adding automatic validation of symbol references appearing the node tree.

    It assumes that the symbol table with the actual definitions is stored as
    a dict in the `annex.symtable` attribute (like :class:`SymbolTableTrait` does).
    """

    __slots__ = ()

    @no_type_check
    @datamodels.root_validator
    @classmethod
    def _validate_symbol_refs(cls: Type[SymbolRefsValidatorTrait], instance: concepts.Node) -> None:
        validator = cls.SymbolRefsValidator()
        symtable = instance.annex.symtable
        for child_node in instance.iter_children_values():
            validator.visit(child_node, symtable=symtable)

        if validator.missing_symbols:
            raise exceptions.EveValueError(
                "Symbols {} not found.".format(validator.missing_symbols)
            )

    class SymbolRefsValidator(visitors.NodeVisitor):
        def __init__(self) -> None:
            self.missing_symbols: Set[str] = set()

        def visit_Node(
            self, node: concepts.Node, *, symtable: Dict[str, Any], **kwargs: Any
        ) -> None:
            for field_name, attribute in node.__datamodel_fields__.items():
                if isinstance(attribute.type, type) and issubclass(
                    attribute.type, concepts.SymbolRef
                ):
                    symbol_name = getattr(node, field_name)
                    if symbol_name not in symtable:
                        self.missing_symbols.add(symbol_name)

            if isinstance(node, SymbolTableTrait):
                # Append symbols from nested scope for nested nodes
                symtable = {**symtable, **node.annex.symtable}
            self.generic_visit(node, symtable=symtable, **kwargs)

        @classmethod
        def apply(cls, node: concepts.Node, *, symtable: Dict[str, Any]) -> Set[str]:
            validator = cls()
            validator.visit(node, symtable=symtable)
            return validator.missing_symbols


class ValidatedSymbolTableTrait(SymbolRefsValidatorTrait, SymbolTableTrait):
    """Node trait adding an automatically created and validated symbol table.

    It is just the combination of the :class:`SymbolTableTrait` and
    :class:`SymbolRefsValidatorTrait` traits.
    """

    __slots__ = ()
    pass


# --- Visitor Traits ---
class VisitorWithSymbolTableTrait(visitors.NodeVisitor):
    """Visitor trait to update or add the symtable to kwargs in the visitor calls.

    Visitors inhering from this trait will automatically pass the active
    symbol table to visitor methods as the 'symtable' keyword argument.
    """

    def visit(self, node: concepts.RootNode, **kwargs: Any) -> Any:
        kwargs.setdefault("symtable", collections.ChainMap())
        new_scope = False
        if isinstance(node, concepts.Node):
            if new_scope := ("symtable" in node.annex):
                kwargs["symtable"] = kwargs["symtable"].new_child(node.annex.symtable)

        result = super().visit(node, **kwargs)

        if new_scope:
            kwargs["symtable"] = kwargs["symtable"].parents

        return result


class PreserveLocationVisitor(visitors.NodeVisitor):
    def visit(self, node: concepts.RootNode, **kwargs: Any) -> Any:
        result = super().visit(node, **kwargs)
        if hasattr(node, "location") and hasattr(result, "location"):
            result.location = node.location
        return result
