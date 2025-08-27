# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py import next as gtx
from gt4py.next.ffront import func_to_foast

TDim = gtx.Dimension("TDim")


def test_lowering_stability_regression():
    def foo(
        a: gtx.Field[[TDim], float],
        b: gtx.Field[[TDim], float],
        flag0: bool,
        flag1: bool,
        c: gtx.Field[[TDim], float],
    ):
        if flag0:
            a, b = (2.0 * c, 2.0 * c) if flag1 else (3.0 * c, 3.0 * c)

        return a, b

    res = func_to_foast.FieldOperatorParser.apply_to_function(foo)
    for _ in range(100):
        assert res == func_to_foast.FieldOperatorParser.apply_to_function(foo)
