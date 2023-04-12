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

from gt4py import eve


def test_annex_preservation(compound_node: eve.Node):
    compound_node.annex.foo = 1
    compound_node.annex.bar = 2

    class SampleTranslator(eve.NodeTranslator):
        PRESERVED_ANNEX_ATTRS = ("foo",)

    translated_node = SampleTranslator().visit(compound_node)

    assert translated_node.annex.foo == 1
    assert not hasattr(translated_node.annex, "bar")
