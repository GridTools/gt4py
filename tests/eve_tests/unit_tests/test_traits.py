# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2020, CSCS - Swiss National Supercomputing Center, ETH Zurich
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

import pytest

import eve
from eve.extended_typing import Any, ClassVar, List

from .. import definitions


@pytest.fixture
def symtable_node_and_expected_symbols():
    node = definitions.make_node_with_symbol_table()
    symbols = {
        node.node_with_name.name: node.node_with_name,
        node.node_with_default_name.name: node.node_with_default_name,
        node.compound_with_name.node_with_name.name: node.compound_with_name.node_with_name,
    }
    symbols.update({n.name: n for n in node.list_with_name})

    yield node, symbols


class _NodeWithSymbolName(eve.Node):
    name: eve.Coerced[eve.SymbolName]


class _NodeWithSymbolTable(eve.Node, eve.SymbolTableTrait):
    symbols: List[_NodeWithSymbolName]


@pytest.fixture
def node_with_duplicated_names_maker():
    def _maker():
        return _NodeWithSymbolTable(
            symbols=[_NodeWithSymbolName(name="repeated"), _NodeWithSymbolName(name="repeated")]
        )

    yield _maker


class TestSymbolTable:
    def test_no_duplicated_names(self, node_with_duplicated_names_maker):
        with pytest.raises(ValueError, match="Multiple definitions of symbol"):
            node_with_duplicated_names_maker()

    def test_symbol_table_creation(self, symtable_node_and_expected_symbols):
        node, expected_symbols = symtable_node_and_expected_symbols
        collected_symtable = node.annex.symtable
        assert isinstance(node.annex.symtable, dict)
        assert all(isinstance(key, str) for key in collected_symtable)

    def test_symbol_table_collection(self, symtable_node_and_expected_symbols):
        node, expected_symbols = symtable_node_and_expected_symbols
        collected_symtable = node.annex.symtable
        assert collected_symtable == expected_symbols
        assert all(
            collected_symtable[symbol_name] is symbol_node
            for symbol_name, symbol_node in expected_symbols.items()
        )

    def test_extra_node_symbols(self):
        class NodeWithName(eve.Node):
            name: eve.Coerced[eve.SymbolName]

        class NodeWithRef(eve.Node):
            ref_name: eve.Coerced[eve.SymbolRef]

        class NodeWithSymbolTable(eve.Node, eve.traits.ValidatedSymbolTableTrait):
            symbols: List[NodeWithRef]

            _NODE_SYMBOLS_: ClassVar[List] = []

        NodeWithSymbolTable.update_forward_refs(locals())

        with pytest.raises(ValueError, match="'foo'"):
            NodeWithSymbolTable(symbols=[NodeWithRef(ref_name="foo")])

        NodeWithSymbolTable._NODE_SYMBOLS_.append(NodeWithName(name="foo"))
        NodeWithSymbolTable(symbols=[NodeWithRef(ref_name=eve.SymbolRef("foo"))])


def test_visitor_with_symbol_table_trait(node_with_symbol_table):
    class BareVisitor(eve.visitors.NodeVisitor):
        def visit_Node(self, node: eve.concepts.RootNode, **kwargs: Any) -> Any:
            assert "symtable" in kwargs

    with pytest.raises(AssertionError, match="'symtable'"):
        BareVisitor().visit(node_with_symbol_table)

    class ExtendedVisitor(eve.traits.VisitorWithSymbolTableTrait):
        def visit_Node(self, node: eve.concepts.RootNode, **kwargs: Any) -> Any:
            assert "symtable" in kwargs
            assert "symbol_name" in kwargs["symtable"]

    ExtendedVisitor().visit(node_with_symbol_table)
