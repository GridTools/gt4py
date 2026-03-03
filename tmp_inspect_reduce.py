# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next.ffront.fbuiltins import float64, neighbor_sum
from gt4py.next.ffront.foast_to_gtir import FieldOperatorLowering
from gt4py.next.ffront.func_to_foast import FieldOperatorParser


IDim = gtx.Dimension("IDim")
JDim = gtx.Dimension("JDim")
Kolor = gtx.Dimension("Kolor")


def foo(inp: gtx.Field[[IDim, JDim, Kolor], float64]):
    return neighbor_sum(inp, axis=Kolor)


parsed = FieldOperatorParser.apply_to_function(foo)
lowered = FieldOperatorLowering.apply(parsed)
print(repr(lowered.expr))
print(type(lowered.expr))
