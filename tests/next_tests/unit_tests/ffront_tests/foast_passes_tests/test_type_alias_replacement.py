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
