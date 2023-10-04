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

import ast
import typing
from typing import TypeAlias

import pytest

import gt4py.next as gtx
from gt4py.eve import SymbolRef
from gt4py.next import float32, float64
from gt4py.next.ffront.fbuiltins import astype
from gt4py.next.ffront.foast_to_itir import FieldOperatorLowering
from gt4py.next.ffront.func_to_foast import FieldOperatorParser
from gt4py.next.iterator import ir as itir, ir_makers as im
from gt4py.next.type_system import type_specifications as ts


TDim = gtx.Dimension("TDim")  # Meaningless dimension, used for tests.
vpfloat: TypeAlias = float32
wpfloat: TypeAlias = float64


@pytest.mark.parametrize("test_input,expected", [(vpfloat, "float32"), (wpfloat, "float64")])
def test_type_alias_replacement(test_input, expected):
    def fieldop_with_typealias(
        a: gtx.Field[[TDim], test_input], b: gtx.Field[[TDim], float32]
    ) -> gtx.Field[[TDim], test_input]:
        return test_input("3.1418") + astype(a, test_input)

    foast_tree = FieldOperatorParser.apply_to_function(fieldop_with_typealias)

    assert (
        foast_tree.body.stmts[0].value.left.func.id == expected
        and foast_tree.body.stmts[0].value.right.args[1].id == expected
    )


def test_type_alias_replacement_astype_with_tuples():
    def fieldop_with_typealias_with_tuples(
        a: gtx.Field[[TDim], vpfloat], b: gtx.Field[[TDim], vpfloat]
    ) -> tuple[gtx.Field[[TDim], wpfloat], gtx.Field[[TDim], wpfloat]]:
        return astype((a, b), wpfloat)

    parsed = FieldOperatorParser.apply_to_function(fieldop_with_typealias_with_tuples)
    lowered = FieldOperatorLowering.apply(parsed)

    # Check that the type of the first arg of "astype" is a tuple
    assert isinstance(parsed.body.stmts[0].value.args[0].type, ts.TupleType)
    # Check that the return type of "astype" is a tuple
    assert isinstance(parsed.body.stmts[0].value.type, ts.TupleType)
    # Check inside the lift function that make_tuple is applied to return a tuple
    assert lowered.expr.fun.args[0].expr.fun == itir.SymRef(id=SymbolRef("make_tuple"))
    # Check that the elements that form the tuple called the cast_ function individually
    assert lowered.expr.args[0].fun.args[0].expr.fun.expr.fun == itir.SymRef(id=SymbolRef("cast_"))
