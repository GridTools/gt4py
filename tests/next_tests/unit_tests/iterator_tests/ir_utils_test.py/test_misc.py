# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from gt4py.next.iterator.ir_utils import ir_makers as im, misc
from gt4py.next.iterator.transforms import inline_lambdas


@pytest.mark.parametrize(
    "expr,  expected_expr",
    [
        (
            im.let("a", "b")(im.make_tuple(im.tuple_get(0, "a"), im.tuple_get(42, "a"))),
            im.ref("b"),
        ),
        (
            im.tuple_get(1, im.ref("a")),
            im.ref("a"),
        ),
        (
            im.make_tuple(im.tuple_get(0, im.ref("a")), im.tuple_get(42, im.ref("a"))),
            im.ref("a"),
        ),
        (
            im.make_tuple(
                im.tuple_get(1, im.ref("a")),
                im.make_tuple(im.tuple_get(2, im.ref("a")), im.tuple_get(0, im.ref("a"))),
            ),
            im.ref("a"),
        ),
        (
            im.tuple_get(3, im.tuple_get(0, im.ref("a"))),
            im.ref("a"),
        ),
        (
            im.tuple_get(
                1,
                im.as_fieldop("scan")(
                    im.lambda_("state", "val")(
                        im.make_tuple("val", im.plus(im.tuple_get(0, "state"), "val"))
                    )
                ),
            ),
            im.as_fieldop("scan")(
                im.lambda_("state", "val")(
                    im.make_tuple("val", im.plus(im.tuple_get(0, "state"), "val"))
                )
            ),
        ),
        (
            im.plus(im.ref("a"), im.ref("b")),
            im.plus(im.ref("a"), im.ref("b")),
        ),
        (
            im.call("as_fieldop")(im.ref("a")),
            im.call("as_fieldop")(im.ref("a")),
        ),
    ],
)
def test_extract_projector(expr, expected_expr):
    actual_projector, actual_expr = misc.extract_projector(expr)
    assert actual_expr == expected_expr

    if expr == expected_expr:
        assert actual_projector is None

    if actual_projector is not None:
        applied_projector = im.call(actual_projector)(actual_expr)

        # simplify original expression and applied projector for comparison
        applied_projector = inline_lambdas.InlineLambdas.apply(applied_projector)
        inlined_expr = inline_lambdas.InlineLambdas.apply(expr)
        assert applied_projector == inlined_expr
