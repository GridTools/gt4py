# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from gt4py.next import common
import pytest
from gt4py.next.iterator.ir_utils import ir_makers as im, domain_utils
from gt4py.next.iterator.transforms import concat_where_transforms
from gt4py.next.iterator.transforms import inline_lambdas
from gt4py.next.type_system import type_specifications as ts
from gt4py.next.iterator.type_system import type_specifications as it_ts

int_type = ts.ScalarType(kind=ts.ScalarKind.INT32)
IDim = common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL)


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

    actual = concat_where_transforms.expand(testee)
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

    actual = concat_where_transforms.expand(testee)
    actual = inline_lambdas.InlineLambdas.apply(actual)  # simplify

    assert actual == expected
