# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import copy

import pytest

from gt4py.next import utils, common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import (
    ir_makers as im,
    domain_utils,
    common_pattern_matcher as cpm,
)
from gt4py.next.iterator.type_system import inference
from gt4py.next.iterator.transforms import (
    fuse_as_fieldop,
    collapse_tuple as ct,
)
from gt4py.next.type_system import type_specifications as ts


IDim = common.Dimension("IDim")
JDim = common.Dimension("JDim")
field_type = ts.FieldType(dims=[IDim], dtype=ts.ScalarType(kind=ts.ScalarKind.INT32))
IOff = im.cartesian_offset(IDim, IDim)


def _with_domain_annex(node: itir.Expr, domain: itir.Expr):
    node = copy.deepcopy(node)
    node.annex.domain = domain_utils.SymbolicDomain.from_expr(domain)
    return node


def test_trivial(uids: utils.IDGeneratorPool):
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
        testee, offset_provider_type={}, allow_undeclared_symbols=True, uids=uids
    )
    assert actual == expected


def test_trivial_literal(uids: utils.IDGeneratorPool):
    d = im.domain("cartesian_domain", {})
    testee = im.op_as_fieldop("plus", d)(im.op_as_fieldop("multiplies", d)(1, 2), 3)
    expected = im.as_fieldop(im.lambda_()(im.plus(im.multiplies_(1, 2), 3)), d)()
    actual = fuse_as_fieldop.FuseAsFieldOp.apply(
        testee, offset_provider_type={}, allow_undeclared_symbols=True, uids=uids
    )
    assert actual == expected


def test_trivial_same_arg_twice(uids: utils.IDGeneratorPool):
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
        testee, offset_provider_type={}, allow_undeclared_symbols=True, enable_cse=False, uids=uids
    )
    assert actual == expected


def test_tuple_arg(uids: utils.IDGeneratorPool):
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
        testee, offset_provider_type={}, allow_undeclared_symbols=True, enable_cse=False, uids=uids
    )
    assert actual == expected


def test_symref_used_twice(uids: utils.IDGeneratorPool):
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
        testee, offset_provider_type={}, allow_undeclared_symbols=True, enable_cse=False, uids=uids
    )
    assert actual == expected


def test_no_inline(uids: utils.IDGeneratorPool):
    d1 = im.domain("cartesian_domain", {IDim: (1, 2)})
    d2 = im.domain("cartesian_domain", {IDim: (0, 3)})
    testee = im.as_fieldop(
        im.lambda_("a")(
            im.plus(im.deref(im.shift(IOff, 1)("a")), im.deref(im.shift(IOff, -1)("a")))
        ),
        d1,
    )(im.op_as_fieldop("plus", d2)(im.ref("inp1", field_type), im.ref("inp2", field_type)))
    actual = fuse_as_fieldop.FuseAsFieldOp.apply(
        testee,
        offset_provider_type={},
        allow_undeclared_symbols=True,
        enable_cse=False,
        uids=uids,
    )
    assert actual == testee


def test_staged_inlining(uids: utils.IDGeneratorPool):
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
            im.let("_icdlv_0", im.lambda_()(im.plus(im.deref("a"), im.deref("b"))))(
                im.plus(im.plus(im.call("_icdlv_0")(), 1), im.plus(im.call("_icdlv_0")(), 2))
            )
        ),
        d,
    )(im.ref("a", field_type), im.ref("b", field_type))
    actual = fuse_as_fieldop.FuseAsFieldOp.apply(
        testee, offset_provider_type={}, allow_undeclared_symbols=True, enable_cse=False, uids=uids
    )
    assert actual == expected


def test_make_tuple_fusion_trivial(uids: utils.IDGeneratorPool):
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
        testee, offset_provider_type={}, allow_undeclared_symbols=True, enable_cse=False, uids=uids
    )
    # simplify to remove unnecessary make_tuple call `{v[0], v[1]}(actual)`
    actual_simplified = ct.CollapseTuple.apply(
        actual, uids=uids, within_stencil=False, allow_undeclared_symbols=True
    )
    assert actual_simplified == expected


def test_make_tuple_fusion_symref(uids: utils.IDGeneratorPool):
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
        testee, offset_provider_type={}, allow_undeclared_symbols=True, enable_cse=False, uids=uids
    )
    # simplify to remove unnecessary make_tuple call
    actual_simplified = ct.CollapseTuple.apply(
        actual, uids=uids, within_stencil=False, allow_undeclared_symbols=True
    )
    assert actual_simplified == expected


def test_make_tuple_fusion_symref_same_ref(uids: utils.IDGeneratorPool):
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
        testee, offset_provider_type={}, allow_undeclared_symbols=True, enable_cse=False, uids=uids
    )
    # simplify to remove unnecessary make_tuple call
    actual_simplified = ct.CollapseTuple.apply(
        actual, uids=uids, within_stencil=False, allow_undeclared_symbols=True
    )
    assert actual_simplified == expected


def test_make_tuple_nested(uids: utils.IDGeneratorPool):
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
        testee, offset_provider_type={}, allow_undeclared_symbols=True, enable_cse=False, uids=uids
    )
    # simplify to remove unnecessary make_tuple call
    actual_simplified = ct.CollapseTuple.apply(
        actual, uids=uids, within_stencil=False, allow_undeclared_symbols=True
    )
    assert actual_simplified == expected


def test_make_tuple_fusion_different_domains(uids: utils.IDGeneratorPool):
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
            "__fasfop_0",
            im.as_fieldop(im.lambda_("a", "c")(im.make_tuple(im.deref("a"), im.deref("c"))), d1)(
                "a", "c"
            ),
        ),
        (
            "__fasfop_1",
            im.as_fieldop(im.lambda_("b", "d")(im.make_tuple(im.deref("b"), im.deref("d"))), d2)(
                "b", "d"
            ),
        ),
    )(
        im.make_tuple(
            im.tuple_get(0, "__fasfop_0"),
            im.tuple_get(0, "__fasfop_1"),
            im.tuple_get(1, "__fasfop_0"),
            im.tuple_get(1, "__fasfop_1"),
        )
    )
    actual = fuse_as_fieldop.FuseAsFieldOp.apply(
        testee, offset_provider_type={}, allow_undeclared_symbols=True, enable_cse=False, uids=uids
    )
    assert actual == expected


def test_partial_inline(uids: utils.IDGeneratorPool):
    d1 = im.domain("cartesian_domain", {IDim: (1, 2)})
    d2 = im.domain("cartesian_domain", {IDim: (0, 3)})
    testee = im.as_fieldop(
        # first argument read at multiple locations -> not inlined
        # second argument only read at a single location -> inlined
        im.lambda_("a", "b")(
            im.plus(
                im.plus(im.deref(im.shift(IOff, 1)("a")), im.deref(im.shift(IOff, -1)("a"))),
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
                im.plus(im.deref(im.shift(IOff, 1)("a")), im.deref(im.shift(IOff, -1)("a"))),
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
        testee,
        offset_provider_type={},
        allow_undeclared_symbols=True,
        enable_cse=False,
        uids=uids,
    )
    assert actual == expected


def test_chained_fusion(uids: utils.IDGeneratorPool):
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
            im.let("_icdlv_0", im.lambda_()(im.plus(im.deref("inp1"), im.deref("inp2"))))(
                im.plus(im.call("_icdlv_0")(), im.call("_icdlv_0")())
            )
        ),
        d,
    )(im.ref("inp1", field_type), im.ref("inp2", field_type))
    actual = fuse_as_fieldop.FuseAsFieldOp.apply(
        testee, offset_provider_type={}, allow_undeclared_symbols=True, enable_cse=False, uids=uids
    )
    assert actual == expected


def test_inline_as_fieldop_with_list_dtype(uids: utils.IDGeneratorPool):
    list_field_type = ts.FieldType(
        dims=[IDim],
        dtype=ts.ListType(
            element_type=ts.ScalarType(kind=ts.ScalarKind.INT32),
            offset_type=common.Dimension("Neighbor", kind=common.DimensionKind.LOCAL),
        ),
    )
    d = im.domain("cartesian_domain", {IDim: (0, 1)})
    testee = im.as_fieldop(
        im.lambda_("inp")(im.call(im.call("reduce")("plus", 0))(im.deref("inp"))), d
    )(im.as_fieldop("deref")(im.ref("inp", list_field_type)))
    expected = im.as_fieldop(
        im.lambda_("inp")(im.call(im.call("reduce")("plus", 0))(im.deref("inp"))), d
    )(im.ref("inp", list_field_type))
    actual = fuse_as_fieldop.FuseAsFieldOp.apply(
        testee, offset_provider_type={}, allow_undeclared_symbols=True, enable_cse=False, uids=uids
    )
    assert actual == expected


def test_inline_into_scan(uids: utils.IDGeneratorPool):
    d = im.domain("cartesian_domain", {IDim: (0, 1)})
    scan = im.call("scan")(im.lambda_("state", "a")(im.plus("state", im.deref("a"))), True, 0)
    testee = im.as_fieldop(scan, d)(im.as_fieldop("deref")(im.ref("a", field_type)))
    expected = im.as_fieldop(scan, d)(im.ref("a", field_type))
    actual = fuse_as_fieldop.FuseAsFieldOp.apply(
        testee, offset_provider_type={}, allow_undeclared_symbols=True, enable_cse=False, uids=uids
    )
    assert actual == expected


def test_no_inline_into_scan(uids: utils.IDGeneratorPool):
    d = im.domain("cartesian_domain", {IDim: (0, 1)})
    scan_stencil = im.call("scan")(
        im.lambda_("state", "a")(im.plus("state", im.deref("a"))), True, 0
    )
    scan = im.as_fieldop(scan_stencil, d)(im.ref("a", field_type))
    testee = im.as_fieldop(im.lambda_("arg")(im.deref("arg")), d)(scan)
    actual = fuse_as_fieldop.FuseAsFieldOp.apply(
        testee, offset_provider_type={}, allow_undeclared_symbols=True, enable_cse=False, uids=uids
    )
    assert actual == testee


def test_opage_arg_deduplication(uids: utils.IDGeneratorPool):
    d = im.domain("cartesian_domain", {IDim: (0, 1)})
    testee = im.op_as_fieldop("plus", d)(im.as_fieldop("deref", d)(im.index(IDim)), im.index(IDim))
    expected = im.as_fieldop(
        im.lambda_("__arg1")(im.plus(im.deref("__arg1"), im.deref("__arg1"))),
        d,
    )(im.index(IDim))
    actual = fuse_as_fieldop.FuseAsFieldOp.apply(
        testee, offset_provider_type={}, allow_undeclared_symbols=True, enable_cse=False, uids=uids
    )
    assert actual == expected


def test_single_step_does_not_absorb_newly_exposed_producer(uids):
    domain = im.domain("cartesian_domain", {IDim: (0, 4)})
    upstream = im.op_as_fieldop("multiplies", domain)(
        im.ref("x", field_type), im.ref("y", field_type)
    )
    immediate = im.op_as_fieldop("plus", domain)(upstream, im.ref("z", field_type))
    outer = inference.infer(
        im.op_as_fieldop("minus", domain)(immediate, im.ref("w", field_type)),
        offset_provider_type={},
        allow_undeclared_symbols=True,
    )
    fused = fuse_as_fieldop.FuseAsFieldOp.fuse_once(
        outer,
        offset_provider_type={},
        uids=uids,
        enable_cse=False,
    )
    assert any(cpm.is_applied_as_fieldop(arg) and arg == upstream for arg in fused.args)
    fully_fused = fuse_as_fieldop.FuseAsFieldOp.apply(
        outer,
        offset_provider_type={},
        uids=uids,
        enable_cse=False,
        allow_undeclared_symbols=True,
    )
    assert not any(cpm.is_applied_as_fieldop(arg) for arg in fully_fused.args)


@pytest.mark.parametrize("allowed", [False, True])
def test_single_step_predicate_cannot_override_eligibility(uids, allowed):
    inner_domain = im.domain("cartesian_domain", {IDim: (0, 3)})
    outer_domain = im.domain("cartesian_domain", {IDim: (1, 2)})
    node = inference.infer(
        im.as_fieldop(
            im.lambda_("a")(
                im.plus(im.deref(im.shift(IOff, 1)("a")), im.deref(im.shift(IOff, -1)("a")))
            ),
            outer_domain,
        )(im.op_as_fieldop("plus", inner_domain)(im.ref("x", field_type), im.ref("y", field_type))),
        offset_provider_type={},
        allow_undeclared_symbols=True,
    )
    assert (
        fuse_as_fieldop.FuseAsFieldOp.fuse_once(
            node,
            offset_provider_type={},
            uids=uids,
            arg_predicate=lambda *_: allowed,
        )
        is node
    )


def test_single_step_predicate_selects_only_requested_input(uids):
    domain = im.domain("cartesian_domain", {IDim: (0, 4)})
    inputs = [im.op_as_fieldop("plus", domain)(im.ref(name, field_type), 1) for name in ("x", "y")]
    node = inference.infer(
        im.op_as_fieldop("plus", domain)(*inputs),
        offset_provider_type={},
        allow_undeclared_symbols=True,
    )
    fused = fuse_as_fieldop.FuseAsFieldOp.fuse_once(
        node,
        offset_provider_type={},
        uids=uids,
        enable_cse=False,
        arg_predicate=lambda arg, _: arg == inputs[0],
    )
    assert inputs[0] not in fused.args
    assert inputs[1] in fused.args
