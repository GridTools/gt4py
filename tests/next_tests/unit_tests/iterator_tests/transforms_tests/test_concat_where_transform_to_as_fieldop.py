# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from gt4py.next import common

from gt4py.next.iterator.ir_utils import ir_makers as im, domain_utils
from gt4py.next.iterator.transforms import concat_where, inline_lambdas
from gt4py.next.iterator.transforms.concat_where import transform_to_as_fieldop
from gt4py.next.iterator.transforms.concat_where.transform_to_as_fieldop import _in
from gt4py.next.type_system import type_specifications as ts

int_type = ts.ScalarType(kind=ts.ScalarKind.INT32)
IDim = common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL)
JDim = common.Dimension(value="JDim", kind=common.DimensionKind.HORIZONTAL)


def test_in_helper():
    pos = im.make_tuple(0, 1)
    bounds = {
        IDim: (3, 4),
        JDim: (5, 6),
    }
    expected = im.and_(
        im.and_(
            im.less_equal(bounds[IDim][0], im.tuple_get(0, pos)),
            im.less(im.tuple_get(0, pos), bounds[IDim][1]),
        ),
        im.and_(
            im.less_equal(bounds[JDim][0], im.tuple_get(1, pos)),
            im.less(im.tuple_get(1, pos), bounds[JDim][1]),
        ),
    )
    actual = _in(pos, im.domain(common.GridType.CARTESIAN, bounds))
    assert actual == expected


def test_trivial():
    domain = im.domain(common.GridType.CARTESIAN, {IDim: (0, 2)})

    cond = im.domain(common.GridType.CARTESIAN, {IDim: (0, 1)})
    testee = im.concat_where(cond, "true_branch", "false_branch")
    testee.annex.domain = domain_utils.SymbolicDomain.from_expr(domain)
    expected = im.as_fieldop(
        im.lambda_("__tcw_pos", "__tcw_arg0", "__tcw_arg1")(
            im.if_(
                im.call("in_")(im.deref("__tcw_pos"), cond),
                im.deref("__tcw_arg0"),
                im.deref("__tcw_arg1"),
            )
        ),
        domain,
    )(im.make_tuple(im.index(IDim)), "true_branch", "false_branch")

    actual = concat_where.transform_to_as_fieldop(testee)
    actual = inline_lambdas.InlineLambdas.apply(actual)  # simplify

    assert actual == expected


def test_capturing_cond():
    domain = im.domain(common.GridType.CARTESIAN, {IDim: (0, 1)})

    cond = im.domain(common.GridType.CARTESIAN, {IDim: ("start", "stop")})
    testee = im.concat_where(cond, "true_branch", "false_branch")
    testee.annex.domain = domain_utils.SymbolicDomain.from_expr(domain)
    expected = im.as_fieldop(
        im.lambda_("__tcw_pos", "__tcw_arg0", "__tcw_arg1", "start", "stop")(
            im.if_(
                im.call("in_")(
                    im.deref("__tcw_pos"),
                    im.domain(
                        common.GridType.CARTESIAN, {IDim: (im.deref("start"), im.deref("stop"))}
                    ),
                ),
                im.deref("__tcw_arg0"),
                im.deref("__tcw_arg1"),
            )
        ),
        domain,
    )(im.make_tuple(im.index(IDim)), "true_branch", "false_branch", "start", "stop")

    actual = concat_where.transform_to_as_fieldop(testee)
    actual = inline_lambdas.InlineLambdas.apply(actual)  # simplify

    assert actual == expected
