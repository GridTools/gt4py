# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.transforms.symbol_ref_utils import get_user_defined_symbols


def test_get_user_defined_symbols():
    domain = itir.FunCall(fun=itir.SymRef(id="cartesian_domain"), args=[])
    ir = itir.Program(
        id="foo",
        function_definitions=[],
        params=[itir.Sym(id="target_symbol")],
        declarations=[],
        body=[
            itir.SetAt(
                expr=itir.Lambda(params=[itir.Sym(id="foo")], expr=itir.SymRef(id="foo")),
                domain=domain,
                target=itir.SymRef(id="target_symbol"),
            )
        ],
    )
    testee = ir.annex.symtable
    actual = get_user_defined_symbols(testee)
    assert actual == {"target_symbol"}


def test_collect_symbol_refs():
    ...
    # TODO: Test collect_symbol_refs
