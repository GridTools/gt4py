# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from gt4py import eve


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
