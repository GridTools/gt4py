# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Callable, Optional

from gt4py import next as gtx
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms import fuse_as_fieldop
from gt4py.next.type_system import type_specifications as ts


IDim = gtx.Dimension("IDim")
field_type = ts.FieldType(dims=[IDim], dtype=ts.ScalarType(kind=ts.ScalarKind.INT32))


def test_trivial():
    d = im.domain("cartesian_domain", {IDim: (0, 1)})
    testee = im.op_as_fieldop("plus", d)(
        im.op_as_fieldop("multiplies", d)(im.ref("inp1", field_type), im.ref("inp2", field_type)),
        im.ref("inp3", field_type),
    )
    expected = im.as_fieldop(
        im.lambda_("inp1", "inp2", "inp3")(
            im.plus(im.multiplies_(im.deref("inp1"), im.deref("inp2")), im.deref("inp3"))
        ),
        d,
    )(im.ref("inp1", field_type), im.ref("inp2", field_type), im.ref("inp3", field_type))
    actual = fuse_as_fieldop.FuseAsFieldOp.apply(
        testee, offset_provider_type={}, allow_undeclared_symbols=True
    )
    assert actual == expected


def test_trivial_literal():
    d = im.domain("cartesian_domain", {})
    testee = im.op_as_fieldop("plus", d)(im.op_as_fieldop("multiplies", d)(1, 2), 3)
    expected = im.as_fieldop(im.lambda_()(im.plus(im.multiplies_(1, 2), 3)), d)()
    actual = fuse_as_fieldop.FuseAsFieldOp.apply(
        testee, offset_provider_type={}, allow_undeclared_symbols=True
    )
    assert actual == expected


def test_tuple_arg():
    d = im.domain("cartesian_domain", {})
    testee = im.op_as_fieldop("plus", d)(
        im.op_as_fieldop(im.lambda_("t")(im.plus(im.tuple_get(0, "t"), im.tuple_get(1, "t"))), d)(
            im.make_tuple(1, 2)
        ),
        3,
    )
    expected = im.as_fieldop(
        im.lambda_()(
            im.plus(
                im.let("t", im.make_tuple(1, 2))(
                    im.plus(im.tuple_get(0, "t"), im.tuple_get(1, "t"))
                ),
                3,
            )
        ),
        d,
    )()
    actual = fuse_as_fieldop.FuseAsFieldOp.apply(
        testee, offset_provider_type={}, allow_undeclared_symbols=True
    )
    assert actual == expected


def test_symref_used_twice():
    d = im.domain("cartesian_domain", {IDim: (0, 1)})
    testee = im.as_fieldop(im.lambda_("a", "b")(im.plus(im.deref("a"), im.deref("b"))), d)(
        im.as_fieldop(im.lambda_("c", "d")(im.multiplies_(im.deref("c"), im.deref("d"))), d)(
            im.ref("inp1", field_type), im.ref("inp2", field_type)
        ),
        im.ref("inp1", field_type),
    )
    expected = im.as_fieldop(
        im.lambda_("inp1", "inp2")(
            im.plus(im.multiplies_(im.deref("inp1"), im.deref("inp2")), im.deref("inp1"))
        ),
        d,
    )("inp1", "inp2")
    actual = fuse_as_fieldop.FuseAsFieldOp.apply(
        testee, offset_provider_type={}, allow_undeclared_symbols=True
    )
    assert actual == expected


def test_no_inline():
    d1 = im.domain("cartesian_domain", {IDim: (1, 2)})
    d2 = im.domain("cartesian_domain", {IDim: (0, 3)})
    testee = im.as_fieldop(
        im.lambda_("a")(
            im.plus(im.deref(im.shift("IOff", 1)("a")), im.deref(im.shift("IOff", -1)("a")))
        ),
        d1,
    )(im.as_fieldop(im.lambda_("inp1")(im.deref("inp1")), d2)(im.ref("inp1", field_type)))
    actual = fuse_as_fieldop.FuseAsFieldOp.apply(
        testee, offset_provider_type={"IOff": IDim}, allow_undeclared_symbols=True
    )
    assert actual == testee


def test_staged_inlining():
    d = im.domain("cartesian_domain", {IDim: (0, 1)})
    testee = im.let(
        "tmp", im.op_as_fieldop("plus", d)(im.ref("a", field_type), im.ref("b", field_type))
    )(
        im.op_as_fieldop("plus", d)(
            im.op_as_fieldop(im.lambda_("a")(im.plus("a", 1)), d)("tmp"),
            im.op_as_fieldop(im.lambda_("a")(im.plus("a", 2)), d)("tmp")
        )
    )
    expected = im.as_fieldop(
        im.lambda_("a", "b")(
            im.let("_icdlv_1", im.plus(im.deref("a"), im.deref("b")))(im.plus(im.plus("_icdlv_1", 1), im.plus("_icdlv_1", 2)))
        ),
        d,
    )(im.ref("a", field_type), im.ref("b", field_type))
    actual = fuse_as_fieldop.FuseAsFieldOp.apply(
        testee, offset_provider_type={}, allow_undeclared_symbols=True
    )
    assert actual == expected

def test_make_tuple_fusion():
    d = im.domain("cartesian_domain", {IDim: (0, 1)})
    testee = im.make_tuple(
        im.as_fieldop("deref", d)(im.ref("a", field_type)),
        im.as_fieldop("deref", d)(im.ref("a", field_type))
    )
    expected = im.as_fieldop(
        im.lambda_("a")(im.make_tuple("a", "a")),
        d,
    )(im.ref("a", field_type))
    actual = fuse_as_fieldop.FuseAsFieldOp.apply(
        testee, offset_provider_type={}, allow_undeclared_symbols=True
    )
    assert actual == expected

def test_partial_inline():
    d1 = im.domain("cartesian_domain", {IDim: (1, 2)})
    d2 = im.domain("cartesian_domain", {IDim: (0, 3)})
    testee = im.as_fieldop(
        # first argument read at multiple locations -> not inlined
        # second argument only reat at a single location -> inlined
        im.lambda_("a", "b")(
            im.plus(
                im.plus(im.deref(im.shift("IOff", 1)("a")), im.deref(im.shift("IOff", -1)("a"))),
                im.deref("b"),
            )
        ),
        d1,
    )(
        im.as_fieldop(im.lambda_("inp1")(im.deref("inp1")), d2)(im.ref("inp1", field_type)),
        im.as_fieldop(im.lambda_("inp1")(im.deref("inp1")), d2)(im.ref("inp1", field_type)),
    )
    expected = im.as_fieldop(
        im.lambda_("a", "inp1")(
            im.plus(
                im.plus(im.deref(im.shift("IOff", 1)("a")), im.deref(im.shift("IOff", -1)("a"))),
                im.deref("inp1"),
            )
        ),
        d1,
    )(im.as_fieldop(im.lambda_("inp1")(im.deref("inp1")), d2)(im.ref("inp1", field_type)), "inp1")
    actual = fuse_as_fieldop.FuseAsFieldOp.apply(
        testee, offset_provider_type={"IOff": IDim}, allow_undeclared_symbols=True
    )
    assert actual == expected
