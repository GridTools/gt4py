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

from dataclasses import dataclass
from typing import Optional

from gt4py import eve
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.transforms.symbol_ref_utils import (
    collect_symbol_refs,
    get_user_defined_symbols,
)


def test_get_user_defined_symbols():
    ir = itir.FencilDefinition(
        id="foo",
        function_definitions=[],
        params=[itir.Sym(id="target_symbol")],
        closures=[
            itir.StencilClosure(
                domain=itir.FunCall(fun=itir.SymRef(id="cartesian_domain"), args=[]),
                stencil=itir.SymRef(id="deref"),
                output=itir.SymRef(id="target_symbol"),
                inputs=[],
            )
        ],
    )
    testee = ir.annex.symtable
    actual = get_user_defined_symbols(testee)
    assert actual == {"target_symbol"}


def test_collect_symbol_refs():
    ...
    # TODO: Test collect_symbol_refs
