# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next import common as gtx_common
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.type_system import type_specifications as ts
from gt4py.next.iterator.transforms.prune_casts import PruneCasts


def test_prune_casts_simple():
    testee = im.call("plus")(im.call("cast_")("x", "float64"), im.call("cast_")("y", "float64"))
    testee.args[0].args[0].type = ts.ScalarType(kind=ts.ScalarKind.FLOAT32)  # x
    testee.args[1].args[0].type = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)  # y

    expected = im.call("plus")(im.call("cast_")("x", "float64"), "y")
    actual = PruneCasts().visit(testee)
    assert actual == expected
