# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

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
