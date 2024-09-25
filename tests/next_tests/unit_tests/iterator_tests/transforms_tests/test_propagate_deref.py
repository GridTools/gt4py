# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

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
