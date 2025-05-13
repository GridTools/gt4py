# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import copy
from typing import Callable, Optional

from gt4py import next as gtx
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im, domain_utils
from gt4py.next.iterator.transforms import fuse_as_fieldop, collapse_tuple
from gt4py.next.type_system import type_specifications as ts


IDim = gtx.Dimension("IDim")
JDim = gtx.Dimension("JDim")
field_type = ts.FieldType(dims=[IDim], dtype=ts.ScalarType(kind=ts.ScalarKind.INT32))


def _with_domain_annex(node: itir.Expr, domain: itir.Expr):
    node = copy.deepcopy(node)
    node.annex.domain = domain_utils.SymbolicDomain.from_expr(domain)
    return node


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


def test_trivial_same_arg_twice():
    d = im.domain("cartesian_domain", {IDim: (0, 1)})
    testee = im.op_as_fieldop("plus", d)(
        # note: inp1 occurs twice here
        im.op_as_fieldop("multiplies", d)(im.ref("inp1", field_type), im.ref("inp1", field_type)),
        im.ref("inp2", field_type),
    )
    expected = im.as_fieldop(
        im.lambda_("inp1", "inp2")(
            im.plus(im.multiplies_(im.deref("inp1"), im.deref("inp1")), im.deref("inp2"))
        ),
        d,
    )(im.ref("inp1", field_type), im.ref("inp2", field_type))
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
    )(im.op_as_fieldop("plus", d2)(im.ref("inp1", field_type), im.ref("inp2", field_type)))
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
            im.op_as_fieldop(im.lambda_("a")(im.plus("a", 2)), d)("tmp"),
        )
    )
    expected = im.as_fieldop(
        im.lambda_("a", "b")(
            im.let("_icdlv_5", im.lambda_()(im.plus(im.deref("a"), im.deref("b"))))(
                im.plus(im.plus(im.call("_icdlv_5")(), 1), im.plus(im.call("_icdlv_5")(), 2))
            )
        ),
        d,
    )(im.ref("a", field_type), im.ref("b", field_type))
    actual = fuse_as_fieldop.FuseAsFieldOp.apply(
        testee, offset_provider_type={}, allow_undeclared_symbols=True
    )
    assert actual == expected


def test_make_tuple_fusion_trivial():
    d = im.domain("cartesian_domain", {IDim: (0, 1)})
    testee = im.make_tuple(
        im.as_fieldop("deref", d)(im.ref("a", field_type)),
        im.as_fieldop("deref", d)(im.ref("a", field_type)),
    )
    expected = im.as_fieldop(
        im.lambda_("a")(im.make_tuple(im.deref("a"), im.deref("a"))),
        d,
    )(im.ref("a", field_type))
    actual = fuse_as_fieldop.FuseAsFieldOp.apply(
        testee, offset_provider_type={}, allow_undeclared_symbols=True
    )
    # simplify to remove unnecessary make_tuple call `{v[0], v[1]}(actual)`
    actual_simplified = collapse_tuple.CollapseTuple.apply(
        actual, within_stencil=False, allow_undeclared_symbols=True
    )
    assert actual_simplified == expected


def test_make_tuple_fusion_symref():
    d = im.domain("cartesian_domain", {IDim: (0, 1)})
    testee = im.make_tuple(
        im.as_fieldop("deref", d)(im.ref("a", field_type)),
        _with_domain_annex(im.ref("b", field_type), d),
    )
    expected = im.as_fieldop(
        im.lambda_("a", "b")(im.make_tuple(im.deref("a"), im.deref("b"))),
        d,
    )(im.ref("a", field_type), im.ref("b", field_type))
    actual = fuse_as_fieldop.FuseAsFieldOp.apply(
        testee, offset_provider_type={}, allow_undeclared_symbols=True
    )
    # simplify to remove unnecessary make_tuple call
    actual_simplified = collapse_tuple.CollapseTuple.apply(
        actual, within_stencil=False, allow_undeclared_symbols=True
    )
    assert actual_simplified == expected


def test_make_tuple_fusion_symref_same_ref():
    d = im.domain("cartesian_domain", {IDim: (0, 1)})
    testee = im.make_tuple(
        im.as_fieldop("deref", d)(im.ref("a", field_type)),
        _with_domain_annex(im.ref("a", field_type), d),
    )
    expected = im.as_fieldop(
        im.lambda_("a")(im.make_tuple(im.deref("a"), im.deref("a"))),
        d,
    )(im.ref("a", field_type))
    actual = fuse_as_fieldop.FuseAsFieldOp.apply(
        testee, offset_provider_type={}, allow_undeclared_symbols=True
    )
    # simplify to remove unnecessary make_tuple call
    actual_simplified = collapse_tuple.CollapseTuple.apply(
        actual, within_stencil=False, allow_undeclared_symbols=True
    )
    assert actual_simplified == expected


def test_make_tuple_nested():
    d = im.domain("cartesian_domain", {IDim: (0, 1)})
    testee = im.make_tuple(
        _with_domain_annex(im.ref("a", field_type), d),
        im.make_tuple(
            _with_domain_annex(im.ref("b", field_type), d),
            _with_domain_annex(im.ref("c", field_type), d),
        ),
    )
    expected = im.as_fieldop(
        im.lambda_("a", "b", "c")(
            im.make_tuple(im.deref("a"), im.make_tuple(im.deref("b"), im.deref("c")))
        ),
        d,
    )(im.ref("a", field_type), im.ref("b", field_type), im.ref("c", field_type))
    actual = fuse_as_fieldop.FuseAsFieldOp.apply(
        testee, offset_provider_type={}, allow_undeclared_symbols=True
    )
    # simplify to remove unnecessary make_tuple call
    actual_simplified = collapse_tuple.CollapseTuple.apply(
        actual, within_stencil=False, allow_undeclared_symbols=True
    )
    assert actual_simplified == expected


def test_make_tuple_fusion_different_domains():
    d1 = im.domain("cartesian_domain", {IDim: (0, 1)})
    d2 = im.domain("cartesian_domain", {JDim: (0, 1)})
    field_i_type = ts.FieldType(dims=[IDim], dtype=ts.ScalarType(kind=ts.ScalarKind.INT32))
    field_j_type = ts.FieldType(dims=[JDim], dtype=ts.ScalarType(kind=ts.ScalarKind.INT32))
    testee = im.make_tuple(
        im.as_fieldop("deref", d1)(im.ref("a", field_i_type)),
        im.as_fieldop("deref", d2)(im.ref("b", field_j_type)),
        im.as_fieldop("deref", d1)(im.ref("c", field_i_type)),
        im.as_fieldop("deref", d2)(im.ref("d", field_j_type)),
    )
    expected = im.let(
        (
            "__fasfop_1",
            im.as_fieldop(im.lambda_("a", "c")(im.make_tuple(im.deref("a"), im.deref("c"))), d1)(
                "a", "c"
            ),
        ),
        (
            "__fasfop_4",
            im.as_fieldop(im.lambda_("b", "d")(im.make_tuple(im.deref("b"), im.deref("d"))), d2)(
                "b", "d"
            ),
        ),
    )(
        im.make_tuple(
            im.tuple_get(0, "__fasfop_1"),
            im.tuple_get(0, "__fasfop_4"),
            im.tuple_get(1, "__fasfop_1"),
            im.tuple_get(1, "__fasfop_4"),
        )
    )
    actual = fuse_as_fieldop.FuseAsFieldOp.apply(
        testee, offset_provider_type={}, allow_undeclared_symbols=True
    )
    assert actual == expected


def test_partial_inline():
    d1 = im.domain("cartesian_domain", {IDim: (1, 2)})
    d2 = im.domain("cartesian_domain", {IDim: (0, 3)})
    testee = im.as_fieldop(
        # first argument read at multiple locations -> not inlined
        # second argument only read at a single location -> inlined
        im.lambda_("a", "b")(
            im.plus(
                im.plus(im.deref(im.shift("IOff", 1)("a")), im.deref(im.shift("IOff", -1)("a"))),
                im.deref("b"),
            )
        ),
        d1,
    )(
        im.op_as_fieldop("plus", d2)(im.ref("inp1", field_type), im.ref("inp2", field_type)),
        im.op_as_fieldop("plus", d2)(im.ref("inp1", field_type), im.ref("inp2", field_type)),
    )
    expected = im.as_fieldop(
        im.lambda_("a", "inp1", "inp2")(
            im.plus(
                im.plus(im.deref(im.shift("IOff", 1)("a")), im.deref(im.shift("IOff", -1)("a"))),
                im.plus(im.deref("inp1"), im.deref("inp2")),
            )
        ),
        d1,
    )(
        im.op_as_fieldop("plus", d2)(im.ref("inp1", field_type), im.ref("inp2", field_type)),
        "inp1",
        "inp2",
    )
    actual = fuse_as_fieldop.FuseAsFieldOp.apply(
        testee, offset_provider_type={"IOff": IDim}, allow_undeclared_symbols=True
    )
    assert actual == expected


def test_chained_fusion():
    d = im.domain("cartesian_domain", {IDim: (0, 1)})
    testee = im.let(
        "a", im.op_as_fieldop("plus", d)(im.ref("inp1", field_type), im.ref("inp2", field_type))
    )(
        im.op_as_fieldop("plus", d)(
            im.as_fieldop("deref", d)(im.ref("a", field_type)),
            im.as_fieldop("deref", d)(im.ref("a", field_type)),
        )
    )
    expected = im.as_fieldop(
        im.lambda_("inp1", "inp2")(
            im.let("_icdlv_5", im.lambda_()(im.plus(im.deref("inp1"), im.deref("inp2"))))(
                im.plus(im.call("_icdlv_5")(), im.call("_icdlv_5")())
            )
        ),
        d,
    )(im.ref("inp1", field_type), im.ref("inp2", field_type))
    actual = fuse_as_fieldop.FuseAsFieldOp.apply(
        testee, offset_provider_type={}, allow_undeclared_symbols=True
    )
    assert actual == expected


def test_inline_as_fieldop_with_list_dtype():
    list_field_type = ts.FieldType(
        dims=[IDim], dtype=ts.ListType(element_type=ts.ScalarType(kind=ts.ScalarKind.INT32))
    )
    d = im.domain("cartesian_domain", {IDim: (0, 1)})
    testee = im.as_fieldop(
        im.lambda_("inp")(im.call(im.call("reduce")("plus", 0))(im.deref("inp"))), d
    )(im.as_fieldop("deref")(im.ref("inp", list_field_type)))
    expected = im.as_fieldop(
        im.lambda_("inp")(im.call(im.call("reduce")("plus", 0))(im.deref("inp"))), d
    )(im.ref("inp", list_field_type))
    actual = fuse_as_fieldop.FuseAsFieldOp.apply(
        testee, offset_provider_type={}, allow_undeclared_symbols=True
    )
    assert actual == expected


def test_inline_into_scan():
    d = im.domain("cartesian_domain", {IDim: (0, 1)})
    scan = im.call("scan")(im.lambda_("state", "a")(im.plus("state", im.deref("a"))), True, 0)
    testee = im.as_fieldop(scan, d)(im.as_fieldop("deref")(im.ref("a", field_type)))
    expected = im.as_fieldop(scan, d)(im.ref("a", field_type))
    actual = fuse_as_fieldop.FuseAsFieldOp.apply(
        testee, offset_provider_type={}, allow_undeclared_symbols=True
    )
    assert actual == expected


def test_no_inline_into_scan():
    d = im.domain("cartesian_domain", {IDim: (0, 1)})
    scan_stencil = im.call("scan")(
        im.lambda_("state", "a")(im.plus("state", im.deref("a"))), True, 0
    )
    scan = im.as_fieldop(scan_stencil, d)(im.ref("a", field_type))
    testee = im.as_fieldop(im.lambda_("arg")(im.deref("arg")), d)(scan)
    actual = fuse_as_fieldop.FuseAsFieldOp.apply(
        testee, offset_provider_type={}, allow_undeclared_symbols=True
    )
    assert actual == testee


def test_opage_arg_deduplication():
    d = im.domain("cartesian_domain", {IDim: (0, 1)})
    testee = im.op_as_fieldop("plus", d)(im.as_fieldop("deref", d)(im.index(IDim)), im.index(IDim))
    expected = im.as_fieldop(
        im.lambda_("__arg1")(im.plus(im.deref("__arg1"), im.deref("__arg1"))),
        d,
    )(im.index(IDim))
    actual = fuse_as_fieldop.FuseAsFieldOp.apply(
        testee, offset_provider_type={}, allow_undeclared_symbols=True
    )
    assert actual == expected
