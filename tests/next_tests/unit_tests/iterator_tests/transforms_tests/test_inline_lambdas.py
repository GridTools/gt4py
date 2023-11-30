# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest

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
    inlined = InlineLambdas.apply(
        testee,
        opcount_preserving=True,
        force_inline_lambda_args=True,
    )
    assert inlined == expected
