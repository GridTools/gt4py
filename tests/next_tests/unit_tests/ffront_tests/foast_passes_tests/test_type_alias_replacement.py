# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import ast
import typing
from typing import TypeAlias

import pytest

import gt4py.next as gtx
from gt4py.next import float32, float64
from gt4py.next.ffront.fbuiltins import astype
from gt4py.next.ffront.func_to_foast import FieldOperatorParser


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
