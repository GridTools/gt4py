# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.merge_let import MergeLet


def test_merge_nested_let():
    testee = im.let("a", "arg2")(im.let("b", "arg1")(im.plus("a", "b")))
    expected = im.call(im.lambda_("a", "b")(im.plus("a", "b")))("arg2", "arg1")

    assert MergeLet().visit(testee) == expected


def test_alpha_equivalent_lets_merge_differently():
    """Characterization test: `MergeLet` is sensitive to the spelling of the bound symbols.

    The two testees below are α-equivalent (the inner binder is named ``a`` in one and
    ``b`` in the other, and it is not referenced from anywhere outside the inner lambda).
    The pass merges only the second one, because its collision guard compares parameter
    *names* rather than binding structure. Both results are semantically correct; what
    this test pins down is that the result is not a function of the α-equivalence class,
    i.e. a renaming by an earlier pass changes what `MergeLet` does.
    """
    # (λ(a) → (λ(a) → a)(1))(i) — inner binder shadows the outer one, merge is skipped
    colliding = im.let("a", "i")(im.let("a", 1)("a"))
    assert MergeLet().visit(colliding) == colliding

    # (λ(a) → (λ(b) → b)(1))(i) — same program up to renaming, but merged
    renamed = im.let("a", "i")(im.let("b", 1)("b"))
    assert MergeLet().visit(renamed) == im.call(im.lambda_("a", "b")("b"))("i", 1)
