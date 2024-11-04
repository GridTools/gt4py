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
