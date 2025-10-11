# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from gt4py.next.type_system import type_specifications as ts
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.inline_lambdas import InlineLambdas


test_data = [
    ("identity1", im.call(im.lambda_("x")("x"))("y"), im.ref("y")),
    ("identity2", im.call(im.lambda_("x")("x"))(im.plus("y", "y")), im.plus("y", "y")),
    ("unused_param", im.call(im.lambda_("x", "y")("x"))("x", "y"), im.ref("x")),
    (
        "composed_addition",
        im.call(im.lambda_("x")(im.plus("x", "x")))(im.plus("y", "y")),
        {
            True: im.call(im.lambda_("x")(im.plus("x", "x")))(im.plus("y", "y")),  # stays as is
            False: im.plus(im.plus("y", "y"), im.plus("y", "y")),
        },
    ),
    (
        "name_collision",
        im.call(im.lambda_("x")(im.plus("x", "x")))(im.plus("x", "y")),
        {
            True: im.call(im.lambda_("x")(im.plus("x", "x")))(im.plus("x", "y")),  # stays as is
            False: im.plus(im.plus("x", "y"), im.plus("x", "y")),
        },
    ),
    (
        "name_shadowing",
        im.call(im.lambda_("x")(im.multiplies_(im.call(im.lambda_("x")(im.plus("x", 1)))(2), "x")))(
            im.plus("x", "x")
        ),
        im.multiplies_(im.plus(2, 1), im.plus("x", "x")),
    ),
    (
        "name_shadowing_external",
        im.call(im.lambda_("x")(im.lambda_("y")(im.plus("x", "y"))))(im.plus("x", "y")),
        im.lambda_("y_")(im.plus(im.plus("x", "y"), "y_")),
    ),
    (
        "renaming_collision",
        # the `y` param of the lambda may not be renamed to `y_` as this name is already referenced
        im.call(im.lambda_("x")(im.lambda_("y")(im.plus(im.plus("x", "y"), "y_"))))(
            im.plus("x", "y")
        ),
        im.lambda_("y__")(im.plus(im.plus(im.plus("x", "y"), "y__"), "y_")),
    ),
    (
        # ensure opcount preserving option works whether `itir.SymRef` has a type or not
        "typed_ref",
        im.let("a", im.call("opaque")())(
            im.plus(im.ref("a", ts.ScalarType(kind=ts.ScalarKind.FLOAT32)), im.ref("a", None))
        ),
        {
            True: im.let("a", im.call("opaque")())(
                im.plus(  # stays as is
                    im.ref("a", ts.ScalarType(kind=ts.ScalarKind.FLOAT32)), im.ref("a", None)
                )
            ),
            False: im.plus(im.call("opaque")(), im.call("opaque")()),
        },
    ),
    (
        # symbol clash when partially inlining (opcount preserving)
        "symbol_clash",
        # (λ(x, y) → f(x, x + y))(y + y, x)
        im.call(im.lambda_("x", "y")(im.call("f")("x", im.plus("x", "y"))))(im.plus("y", "y"), "x"),
        {
            # (λ(x_) → f(x_, x_ + x))(y + y)
            True: im.call(im.lambda_("x_")(im.call("f")("x_", im.plus("x_", "x"))))(
                im.plus("y", "y")
            ),
            # f(y + y, (y + y) + x)  # noqa: ERA001
            False: im.call("f")(im.plus("y", "y"), im.plus(im.plus("y", "y"), "x")),
        },
    ),
]


@pytest.mark.parametrize("opcount_preserving", [True, False])
@pytest.mark.parametrize("name,testee,expected", test_data)
def test(name, opcount_preserving, testee, expected):
    if isinstance(expected, dict):
        expected = expected[opcount_preserving]

    inlined = InlineLambdas.apply(testee, opcount_preserving=opcount_preserving)
    assert inlined == expected


def test_inline_lambda_args():
    testee = im.let("reduce_step", im.lambda_("x", "y")(im.plus("x", "y")))(
        im.lambda_("a")(
            im.call("reduce_step")(im.call("reduce_step")(im.call("reduce_step")("a", 1), 2), 3)
        )
    )
    expected = im.lambda_("a")(
        im.call(im.lambda_("x", "y")(im.plus("x", "y")))(
            im.call(im.lambda_("x", "y")(im.plus("x", "y")))(
                im.call(im.lambda_("x", "y")(im.plus("x", "y")))("a", 1), 2
            ),
            3,
        )
    )
    inlined = InlineLambdas.apply(testee, opcount_preserving=True, force_inline_lambda_args=True)
    assert inlined == expected


def test_type_preservation():
    testee = im.let("a", "b")("a")
    testee.type = testee.annex.type = ts.ScalarType(kind=ts.ScalarKind.FLOAT32)
    inlined = InlineLambdas.apply(testee)
    assert inlined.type == inlined.annex.type == ts.ScalarType(kind=ts.ScalarKind.FLOAT32)
