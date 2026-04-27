# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.unroll_tree_map import _unroll
from gt4py.next.type_system import type_specifications as ts


T = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
TT = ts.TupleType(types=[T, T])


def test_single_arg():
    result = _unroll(im.ref("f"), [TT], [im.ref("t")])
    expected = im.make_tuple(im.call("f")(im.tuple_get(0, "t")), im.call("f")(im.tuple_get(1, "t")))
    assert result == expected


def test_multi_arg():
    result = _unroll(im.ref("f"), [TT, TT], [im.ref("a"), im.ref("b")])
    expected = im.make_tuple(
        im.call("f")(im.tuple_get(0, "a"), im.tuple_get(0, "b")),
        im.call("f")(im.tuple_get(1, "a"), im.tuple_get(1, "b")),
    )
    assert result == expected


def test_nested():
    outer = ts.TupleType(types=[TT, T])
    result = _unroll(im.ref("f"), [outer], [im.ref("t")])
    expected = im.make_tuple(
        im.make_tuple(
            im.call("f")(im.tuple_get(0, im.tuple_get(0, "t"))),
            im.call("f")(im.tuple_get(1, im.tuple_get(0, "t"))),
        ),
        im.call("f")(im.tuple_get(1, "t")),
    )
    assert result == expected
