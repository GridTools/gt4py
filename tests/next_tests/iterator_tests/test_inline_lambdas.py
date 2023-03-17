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

from gt4py.next.ffront import itir_makers as im
from gt4py.next.iterator.transforms.inline_lambdas import InlineLambdas


test_data = [
    ("identity1", im.call_(im.lambda__("x")("x"))("y"), im.ref("y")),
    ("identity2", im.call_(im.lambda__("x")("x"))(im.plus_("y", "y")), im.plus_("y", "y")),
    ("unused_param", im.call_(im.lambda__("x", "y")("x"))("x", "y"), im.ref("x")),
    (
        "composed_addition",
        im.call_(im.lambda__("x")(im.plus_("x", "x")))(im.plus_("y", "y")),
        {
            True: im.call_(im.lambda__("x")(im.plus_("x", "x")))(im.plus_("y", "y")),  # stays as is
            False: im.plus_(im.plus_("y", "y"), im.plus_("y", "y")),
        },
    ),
    (
        "name_collision",
        im.call_(im.lambda__("x")(im.plus_("x", "x")))(im.plus_("x", "y")),
        {
            True: im.call_(im.lambda__("x")(im.plus_("x", "x")))(im.plus_("x", "y")),  # stays as is
            False: im.plus_(im.plus_("x", "y"), im.plus_("x", "y")),
        },
    ),
    (
        "name_shadowing",
        im.call_(
            im.lambda__("x")(im.multiplies_(im.call_(im.lambda__("x")(im.plus_("x", 1)))(2), "x"))
        )(im.plus_("x", "x")),
        im.multiplies_(im.plus_(2, 1), im.plus_("x", "x")),
    ),
]


@pytest.mark.parametrize("opcount_preserving", [True, False])
@pytest.mark.parametrize("name,testee,expected", test_data)
def test(name, opcount_preserving, testee, expected):
    if isinstance(expected, dict):
        expected = expected[opcount_preserving]

    inlined = InlineLambdas.apply(testee, opcount_preserving=opcount_preserving)
    assert inlined == expected
