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

from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.propagate_deref import PropagateDeref


def test_deref_let_propagation():
    testee = im.deref(im.call(im.lambda_("inner_it")(im.lift("stencil")("inner_it")))("outer_it"))
    expected = im.call(im.lambda_("inner_it")(im.deref(im.lift("stencil")("inner_it"))))("outer_it")

    actual = PropagateDeref.apply(testee)
    assert actual == expected


def test_deref_if_propagation():
    testee = im.deref(im.if_("cond", "true_branch", "false_branch"))
    expected = im.if_("cond", im.deref("true_branch"), im.deref("false_branch"))

    actual = PropagateDeref.apply(testee)
    assert actual == expected
