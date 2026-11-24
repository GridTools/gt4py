# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from gt4py import eve

from .. import definitions


def test_annex_preservation(compound_node: eve.Node):
    compound_node.annex.foo = 1
    compound_node.annex.bar = None  # None is easily forgotten so test seperately
    compound_node.annex.baz = 2

    class SampleTranslator(eve.NodeTranslator):
        PRESERVED_ANNEX_ATTRS = ("foo", "bar")

    translated_node = SampleTranslator().visit(compound_node)

    assert translated_node.annex.foo == 1
    assert translated_node.annex.bar is None
    assert not hasattr(translated_node.annex, "baz")


def test_annex_preservation_translated_node(compound_node: eve.Node):
    compound_node.annex.foo = 1
    compound_node.annex.baz = 2

    class SampleTranslator(eve.NodeTranslator):
        PRESERVED_ANNEX_ATTRS = ("foo",)

        def visit_Node(self, node: eve.Node):
            # just return an empty node, we care about the annex only anyway
            return eve.Node()

    translated_node = SampleTranslator().visit(compound_node)

    assert translated_node.annex.foo == 1
    assert not hasattr(translated_node.annex, "baz")


def test_annex_preservation_translated_node_overwritten(compound_node: eve.Node):
    compound_node.annex.foo = "1+1"

    class SampleTranslator(eve.NodeTranslator):
        PRESERVED_ANNEX_ATTRS = ("foo",)

        def visit_Node(self, node: eve.Node):
            # just return an empty node, we care about the annex only anyway
            new_node = eve.Node()
            # the annex value is different, but considered equivalent by this pass
            new_node.annex.foo = "2"
            return new_node

    translated_node = SampleTranslator().visit(compound_node)

    assert translated_node.annex.foo == "2"


def test_immutable_leaves_are_shared(simple_node_with_collections: eve.Node):
    """`NodeTranslator` copies what a pass may mutate; immutable leaves are passed through."""
    translated_node = eve.NodeTranslator().visit(simple_node_with_collections)

    assert translated_node is not simple_node_with_collections
    assert translated_node == simple_node_with_collections
    assert translated_node.loc is simple_node_with_collections.loc
    assert translated_node.int_list is not simple_node_with_collections.int_list
    assert translated_node.str_set is not simple_node_with_collections.str_set
    assert translated_node.str_to_int_dict is not simple_node_with_collections.str_to_int_dict


class Leaves(eve.Node):
    string: str
    symbol: eve.SymbolName
    enum_value: definitions.StrKind
    location: eve.SourceLocation
    locations: eve.SourceLocationGroup


def test_immutable_leaf_types_are_shared():

    node = Leaves(
        string="value",
        symbol=eve.SymbolName("name"),
        enum_value=definitions.StrKind.FOO,
        location=(location := eve.SourceLocation(line=1, column=1, filename="source.py")),
        locations=eve.SourceLocationGroup(location),
    )

    translated_node = eve.NodeTranslator().visit(node)

    assert translated_node == node
    for name, value in node.iter_children_items():
        assert getattr(translated_node, str(name)) is value
