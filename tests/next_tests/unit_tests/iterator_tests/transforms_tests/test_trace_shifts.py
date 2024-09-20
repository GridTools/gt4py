# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.trace_shifts import Sentinel, TraceShifts


def test_trivial_stencil():
    expected = [{()}]

    actual = TraceShifts.trace_stencil(im.ref("deref"), num_args=1)
    assert actual == expected


def test_shift():
    testee = im.lambda_("inp")(im.deref(im.shift("I", 1)("inp")))
    expected = [{(ir.OffsetLiteral(value="I"), ir.OffsetLiteral(value=1))}]

    actual = TraceShifts.trace_stencil(testee)
    assert actual == expected


def test_lift():
    testee = im.lambda_("inp")(im.deref(im.lift("deref")(im.shift("I", 1)("inp"))))
    expected = [{(ir.OffsetLiteral(value="I"), ir.OffsetLiteral(value=1))}]

    actual = TraceShifts.trace_stencil(testee)
    assert actual == expected


def test_neighbors():
    testee = im.lambda_("inp")(im.neighbors("O", "inp"))
    expected = [{(ir.OffsetLiteral(value="O"), Sentinel.ALL_NEIGHBORS)}]

    actual = TraceShifts.trace_stencil(testee)
    assert actual == expected


def test_reduce():
    # λ(inp) → reduce(plus, 0.)(·inp)
    testee = im.lambda_("inp")(im.call(im.call("reduce")("plus", 0.0))(im.deref("inp")))
    expected = [{()}]

    actual = TraceShifts.trace_stencil(testee)
    assert actual == expected


def test_shifted_literal():
    "Test shifting an applied lift of a stencil returning a constant / literal works."
    testee = im.lambda_("inp")(im.deref(im.shift("I", 1)(im.lift(im.lambda_()(1))())))
    expected = [set()]

    actual = TraceShifts.trace_stencil(testee)
    assert actual == expected


def test_tuple_get():
    # λ(x, y) → ·{x, y}[1]
    testee = im.lambda_("x", "y")(im.deref(im.tuple_get(1, im.make_tuple("x", "y"))))
    expected = [
        set(),  # never derefed
        {()},  # once derefed
    ]

    actual = TraceShifts.trace_stencil(testee)
    assert actual == expected


def test_trace_non_closure_input_arg():
    x, y = im.sym("x"), im.sym("y")
    # λ(x) → (λ(y) → ·⟪Iₒ, 1ₒ⟫(y))(⟪Iₒ, 2ₒ⟫(x))
    testee = im.lambda_(x)(
        im.call(im.lambda_(y)(im.deref(im.shift("I", 1)("y"))))(im.shift("I", 2)("x"))
    )

    actual = TraceShifts.trace_stencil(testee, save_to_annex=True)

    assert x.annex.recorded_shifts == {
        (
            ir.OffsetLiteral(value="I"),
            ir.OffsetLiteral(value=2),
            ir.OffsetLiteral(value="I"),
            ir.OffsetLiteral(value=1),
        )
    }
    assert y.annex.recorded_shifts == {(ir.OffsetLiteral(value="I"), ir.OffsetLiteral(value=1))}


def test_inner_iterator():
    inner_shift = im.shift("I", 1)("x")
    # λ(x) → ·⟪Iₒ, 1ₒ⟫(⟪Iₒ, 1ₒ⟫(x))
    testee = im.lambda_("x")(im.deref(im.shift("I", 1)(inner_shift)))
    expected = {(ir.OffsetLiteral(value="I"), ir.OffsetLiteral(value=1))}

    actual = TraceShifts.trace_stencil(testee, save_to_annex=True)
    assert inner_shift.annex.recorded_shifts == expected


def test_tuple_get_on_closure_input():
    # λ(x) → (·⟪Iₒ, 1ₒ⟫(x))[0]
    testee = im.lambda_("x")(im.tuple_get(0, im.deref(im.shift("I", 1)("x"))))
    expected = [{(ir.OffsetLiteral(value="I"), ir.OffsetLiteral(value=1))}]

    actual = TraceShifts.trace_stencil(testee)
    assert actual == expected


def test_if_tuple_branch_broadcasting():
    # λ(cond, inp) → (if ·cond then ·inp else {1, 2})[1]
    testee = im.lambda_("cond", "inp")(
        im.tuple_get(
            1,
            im.if_(
                im.deref("cond"),
                im.deref("inp"),
                im.make_tuple(im.literal_from_value(1), im.literal_from_value(2)),
            ),
        )
    )
    expected = [
        {()},  # cond
        {()},  # inp
    ]

    actual = TraceShifts.trace_stencil(testee)
    assert actual == expected


def test_if_of_iterators():
    # λ(cond, x) → ·⟪Iₒ, 1ₒ⟫(if ·cond then ⟪Iₒ, 2ₒ⟫(x) else ⟪Iₒ, 3ₒ⟫(x))
    testee = im.lambda_("cond", "x")(
        im.deref(
            im.shift("I", 1)(im.if_(im.deref("cond"), im.shift("I", 2)("x"), im.shift("I", 3)("x")))
        )
    )
    expected = [
        {()},  # cond
        {  # inp
            (
                ir.OffsetLiteral(value="I"),
                ir.OffsetLiteral(value=2),
                ir.OffsetLiteral(value="I"),
                ir.OffsetLiteral(value=1),
            ),
            (
                ir.OffsetLiteral(value="I"),
                ir.OffsetLiteral(value=3),
                ir.OffsetLiteral(value="I"),
                ir.OffsetLiteral(value=1),
            ),
        },
    ]

    actual = TraceShifts.trace_stencil(testee)
    assert actual == expected


def test_if_of_tuples_of_iterators():
    # λ(cond, x) →
    #   ·⟪Iₒ, 1ₒ⟫((if ·cond then {⟪Iₒ, 2ₒ⟫(x), ⟪Iₒ, 3ₒ⟫(x)} else {⟪Iₒ, 4ₒ⟫(x), ⟪Iₒ, 5ₒ⟫(x)})[0])
    testee = im.lambda_("cond", "x")(
        im.deref(
            im.shift("I", 1)(
                im.tuple_get(
                    0,
                    im.if_(
                        im.deref("cond"),
                        im.make_tuple(im.shift("I", 2)("x"), im.shift("I", 3)("x")),
                        im.make_tuple(im.shift("I", 4)("x"), im.shift("I", 5)("x")),
                    ),
                )
            )
        )
    )
    expected = [
        {()},  # cond
        {  # inp
            (
                ir.OffsetLiteral(value="I"),
                ir.OffsetLiteral(value=2),
                ir.OffsetLiteral(value="I"),
                ir.OffsetLiteral(value=1),
            ),
            (
                ir.OffsetLiteral(value="I"),
                ir.OffsetLiteral(value=4),
                ir.OffsetLiteral(value="I"),
                ir.OffsetLiteral(value=1),
            ),
        },
    ]

    actual = TraceShifts.trace_stencil(testee)
    assert actual == expected


def test_non_derefed_iterator():
    """
    Test that even if an iterator is not derefed the resulting dict has an (empty) entry for it.
    """
    non_derefed_it = im.shift("I", 1)("x")
    # λ(x) → (λ(non_derefed_it) → ·x)(⟪Iₒ, 1ₒ⟫(x))
    testee = im.lambda_("x")(im.let("non_derefed_it", non_derefed_it)(im.deref("x")))

    actual = TraceShifts.trace_stencil(testee, save_to_annex=True)
    assert non_derefed_it.annex.recorded_shifts == set()
