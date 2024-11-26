# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py import next as gtx
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.prune_casts import PruneCasts
from gt4py.next.iterator.type_system import inference as type_inference
from gt4py.next.type_system import type_specifications as ts


def test_prune_casts_simple():
    x_ref = im.ref("x", ts.ScalarType(kind=ts.ScalarKind.FLOAT32))
    y_ref = im.ref("y", ts.ScalarType(kind=ts.ScalarKind.FLOAT64))
    testee = im.call("plus")(im.call("cast_")(x_ref, "float64"), im.call("cast_")(y_ref, "float64"))
    testee = type_inference.infer(testee, offset_provider_type={}, allow_undeclared_symbols=True)

    expected = im.call("plus")(im.call("cast_")(x_ref, "float64"), y_ref)
    actual = PruneCasts.apply(testee)
    assert actual == expected


def test_prune_casts_fieldop():
    IDim = gtx.Dimension("IDim")
    x_ref = im.ref("x", ts.FieldType(dims=[IDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT32)))
    y_ref = im.ref("y", ts.FieldType(dims=[IDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64)))
    testee = im.op_as_fieldop("plus")(
        im.cast_as_fieldop("float64")(x_ref),
        im.cast_as_fieldop("float64")(y_ref),
    )
    testee = type_inference.infer(testee, offset_provider_type={}, allow_undeclared_symbols=True)

    expected = im.op_as_fieldop("plus")(
        im.cast_as_fieldop("float64")(x_ref),
        y_ref,
    )
    actual = PruneCasts.apply(testee)
    assert actual == expected
