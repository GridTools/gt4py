# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.merge_let import MergeLet


def test_simple():
    testee = im.let("a", "arg1")(im.let("b", "arg2")(im.plus("a", "b")))
    expected = im.call(im.lambda_("a", "b")(im.plus("a", "b")))("arg1", "arg2")
    assert MergeLet().visit(testee) == expected


def test_no_merge_on_param_collision():
    testee = im.let("a", "arg1")(im.let("a", "arg2")(im.plus("a", "a")))
    assert MergeLet().visit(testee) == testee


def test_no_merge_outer_arg_refs_inner_param():
    testee = im.let("a", im.deref("b"))(im.let("b", "arg2")(im.plus("a", "b")))
    assert MergeLet().visit(testee) == testee


@pytest.mark.parametrize("outer_param", ["s", "shift"])
def test_no_merge_inner_arg_refs_outer_param(outer_param):
    # `shift` is the name of a builtin. Renaming the binder is semantics-preserving and
    # must not change whether the merge happens, otherwise `·shift` ends up outside of
    # the lambda that binds `shift`.
    testee = im.let(outer_param, "it")(im.let("b", im.deref(outer_param))("b"))
    assert MergeLet().visit(testee) == testee
