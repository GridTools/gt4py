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


def test_trivial():
    testee = ir.StencilClosure(
        stencil=ir.SymRef(id="deref"),
        inputs=[ir.SymRef(id="inp")],
        output=ir.SymRef(id="out"),
        domain=ir.FunCall(fun=ir.SymRef(id="cartesian_domain"), args=[]),
    )
    expected = {"inp": {()}}

    actual = TraceShifts.apply(testee)
    assert actual == expected


def test_shift():
    testee = ir.StencilClosure(
        stencil=ir.Lambda(
            expr=ir.FunCall(
                fun=ir.SymRef(id="deref"),
                args=[
                    ir.FunCall(
                        fun=ir.FunCall(
                            fun=ir.SymRef(id="shift"),
                            args=[ir.OffsetLiteral(value="I"), ir.OffsetLiteral(value=1)],
                        ),
                        args=[ir.SymRef(id="x")],
                    )
                ],
            ),
            params=[ir.Sym(id="x")],
        ),
        inputs=[ir.SymRef(id="inp")],
        output=ir.SymRef(id="out"),
        domain=ir.FunCall(fun=ir.SymRef(id="cartesian_domain"), args=[]),
    )
    expected = {"inp": {(ir.OffsetLiteral(value="I"), ir.OffsetLiteral(value=1))}}

    actual = TraceShifts.apply(testee)
    assert actual == expected


def test_lift():
    testee = ir.StencilClosure(
        stencil=ir.Lambda(
            expr=ir.FunCall(
                fun=ir.SymRef(id="deref"),
                args=[
                    ir.FunCall(
                        fun=ir.FunCall(fun=ir.SymRef(id="lift"), args=[ir.SymRef(id="deref")]),
                        args=[
                            ir.FunCall(
                                fun=ir.FunCall(
                                    fun=ir.SymRef(id="shift"),
                                    args=[ir.OffsetLiteral(value="I"), ir.OffsetLiteral(value=1)],
                                ),
                                args=[ir.SymRef(id="x")],
                            )
                        ],
                    )
                ],
            ),
            params=[ir.Sym(id="x")],
        ),
        inputs=[ir.SymRef(id="inp")],
        output=ir.SymRef(id="out"),
        domain=ir.FunCall(fun=ir.SymRef(id="cartesian_domain"), args=[]),
    )
    expected = {"inp": {(ir.OffsetLiteral(value="I"), ir.OffsetLiteral(value=1))}}

    actual = TraceShifts.apply(testee)
    assert actual == expected


def test_neighbors():
    testee = ir.StencilClosure(
        stencil=ir.Lambda(
            expr=ir.FunCall(
                fun=ir.SymRef(id="neighbors"), args=[ir.OffsetLiteral(value="O"), ir.SymRef(id="x")]
            ),
            params=[ir.Sym(id="x")],
        ),
        inputs=[ir.SymRef(id="inp")],
        output=ir.SymRef(id="out"),
        domain=ir.FunCall(fun=ir.SymRef(id="cartesian_domain"), args=[]),
    )
    expected = {"inp": {(ir.OffsetLiteral(value="O"), Sentinel.ALL_NEIGHBORS)}}

    actual = TraceShifts.apply(testee)
    assert actual == expected


def test_reduce():
    testee = ir.StencilClosure(
        # λ(inp) → reduce(plus, 0.)(·inp)
        stencil=ir.Lambda(
            params=[ir.Sym(id="inp")],
            expr=ir.FunCall(
                fun=ir.FunCall(
                    fun=ir.SymRef(id="reduce"),
                    args=[ir.SymRef(id="plus"), im.literal_from_value(0.0)],
                ),
                args=[ir.FunCall(fun=ir.SymRef(id="deref"), args=[ir.SymRef(id="inp")])],
            ),
        ),
        inputs=[ir.SymRef(id="inp")],
        output=ir.SymRef(id="out"),
        domain=ir.FunCall(fun=ir.SymRef(id="cartesian_domain"), args=[]),
    )
    expected = {"inp": {()}}

    actual = TraceShifts.apply(testee)
    assert actual == expected


def test_shifted_literal():
    "Test shifting an applied lift of a stencil returning a constant / literal works."
    testee = ir.StencilClosure(
        # λ(x) → ·⟪Iₒ, 1ₒ⟫((↑(λ() → 1))())
        stencil=im.lambda_("x")(im.deref(im.shift("I", 1)(im.lift(im.lambda_()(1))()))),
        inputs=[ir.SymRef(id="inp")],
        output=ir.SymRef(id="out"),
        domain=ir.FunCall(fun=ir.SymRef(id="cartesian_domain"), args=[]),
    )
    expected = {"inp": set()}

    actual = TraceShifts.apply(testee)
    assert actual == expected


def test_tuple_get():
    testee = ir.StencilClosure(
        # λ(x, y) → ·{x, y}[1]
        stencil=im.lambda_("x", "y")(im.deref(im.tuple_get(1, im.make_tuple("x", "y")))),
        inputs=[ir.SymRef(id="inp1"), ir.SymRef(id="inp2")],
        output=ir.SymRef(id="out"),
        domain=ir.FunCall(fun=ir.SymRef(id="cartesian_domain"), args=[]),
    )
    expected = {"inp1": set(), "inp2": {()}}  # never derefed  # once derefed

    actual = TraceShifts.apply(testee)
    assert actual == expected


def test_trace_non_closure_input_arg():
    x, y = im.sym("x"), im.sym("y")
    testee = ir.StencilClosure(
        # λ(x) → (λ(y) → ·⟪Iₒ, 1ₒ⟫(y))(⟪Iₒ, 2ₒ⟫(x))
        stencil=im.lambda_(x)(
            im.call(im.lambda_(y)(im.deref(im.shift("I", 1)("y"))))(im.shift("I", 2)("x"))
        ),
        inputs=[ir.SymRef(id="inp")],
        output=ir.SymRef(id="out"),
        domain=ir.FunCall(fun=ir.SymRef(id="cartesian_domain"), args=[]),
    )

    actual = TraceShifts.apply(testee, inputs_only=False)

    assert actual[id(x)] == {
        (
            ir.OffsetLiteral(value="I"),
            ir.OffsetLiteral(value=2),
            ir.OffsetLiteral(value="I"),
            ir.OffsetLiteral(value=1),
        )
    }
    assert actual[id(y)] == {(ir.OffsetLiteral(value="I"), ir.OffsetLiteral(value=1))}


def test_inner_iterator():
    inner_shift = im.shift("I", 1)("x")
    testee = ir.StencilClosure(
        # λ(x) → ·⟪Iₒ, 1ₒ⟫(⟪Iₒ, 1ₒ⟫(x))
        stencil=im.lambda_("x")(im.deref(im.shift("I", 1)(inner_shift))),
        inputs=[ir.SymRef(id="inp")],
        output=ir.SymRef(id="out"),
        domain=ir.FunCall(fun=ir.SymRef(id="cartesian_domain"), args=[]),
    )
    expected = {(ir.OffsetLiteral(value="I"), ir.OffsetLiteral(value=1))}

    actual = TraceShifts.apply(testee, inputs_only=False)
    assert actual[id(inner_shift)] == expected


def test_tuple_get_on_closure_input():
    testee = ir.StencilClosure(
        # λ(x) → (·⟪Iₒ, 1ₒ⟫(x))[0]
        stencil=im.lambda_("x")(im.tuple_get(0, im.deref(im.shift("I", 1)("x")))),
        inputs=[ir.SymRef(id="inp")],
        output=ir.SymRef(id="out"),
        domain=ir.FunCall(fun=ir.SymRef(id="cartesian_domain"), args=[]),
    )
    expected = {"inp": {(ir.OffsetLiteral(value="I"), ir.OffsetLiteral(value=1))}}

    actual = TraceShifts.apply(testee)
    assert actual == expected


def test_if_tuple_branch_broadcasting():
    testee = ir.StencilClosure(
        # λ(cond, inp) → (if ·cond then ·inp else {1, 2})[1]
        stencil=im.lambda_("cond", "inp")(
            im.tuple_get(
                1,
                im.if_(
                    im.deref("cond"),
                    im.deref("inp"),
                    im.make_tuple(im.literal_from_value(1), im.literal_from_value(2)),
                ),
            )
        ),
        inputs=[ir.SymRef(id="cond"), ir.SymRef(id="inp")],
        output=ir.SymRef(id="out"),
        domain=ir.FunCall(fun=ir.SymRef(id="cartesian_domain"), args=[]),
    )
    expected = {"cond": {()}, "inp": {()}}

    actual = TraceShifts.apply(testee)
    assert actual == expected


def test_if_of_iterators():
    testee = ir.StencilClosure(
        # λ(cond, x) → ·⟪Iₒ, 1ₒ⟫(if ·cond then ⟪Iₒ, 2ₒ⟫(x) else ⟪Iₒ, 3ₒ⟫(x))
        stencil=im.lambda_("cond", "x")(
            im.deref(
                im.shift("I", 1)(
                    im.if_(im.deref("cond"), im.shift("I", 2)("x"), im.shift("I", 3)("x"))
                )
            )
        ),
        inputs=[ir.SymRef(id="cond"), ir.SymRef(id="inp")],
        output=ir.SymRef(id="out"),
        domain=ir.FunCall(fun=ir.SymRef(id="cartesian_domain"), args=[]),
    )
    expected = {
        "cond": {()},
        "inp": {
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
    }

    actual = TraceShifts.apply(testee)
    assert actual == expected


def test_if_of_tuples_of_iterators():
    testee = ir.StencilClosure(
        # λ(cond, x) →
        #   ·⟪Iₒ, 1ₒ⟫((if ·cond then {⟪Iₒ, 2ₒ⟫(x), ⟪Iₒ, 3ₒ⟫(x)} else {⟪Iₒ, 4ₒ⟫(x), ⟪Iₒ, 5ₒ⟫(x)})[0])
        stencil=im.lambda_("cond", "x")(
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
        ),
        inputs=[ir.SymRef(id="cond"), ir.SymRef(id="inp")],
        output=ir.SymRef(id="out"),
        domain=ir.FunCall(fun=ir.SymRef(id="cartesian_domain"), args=[]),
    )
    expected = {
        "cond": {()},
        "inp": {
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
    }

    actual = TraceShifts.apply(testee)
    assert actual == expected


def test_non_derefed_iterator():
    """
    Test that even if an iterator is not derefed the resulting dict has an (empty) entry for it.
    """
    non_derefed_it = im.shift("I", 1)("x")
    testee = ir.StencilClosure(
        # λ(x) → (λ(non_derefed_it) → ·x)(⟪Iₒ, 1ₒ⟫(x))
        stencil=im.lambda_("x")(im.let("non_derefed_it", non_derefed_it)(im.deref("x"))),
        inputs=[ir.SymRef(id="inp")],
        output=ir.SymRef(id="out"),
        domain=ir.FunCall(fun=ir.SymRef(id="cartesian_domain"), args=[]),
    )

    actual = TraceShifts.apply(testee, inputs_only=False)
    assert actual[id(non_derefed_it)] == set()
