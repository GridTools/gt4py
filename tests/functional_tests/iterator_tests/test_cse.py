import pytest

from eve.utils import UIDs
from functional.ffront import itir_makers as im
from functional.iterator import ir
from functional.iterator.transforms.cse import CommonSubexpressionElimination as CSE


def test_trivial():
    common = ir.FunCall(fun=ir.SymRef(id="plus"), args=[ir.SymRef(id="x"), ir.SymRef(id="y")])
    testee = ir.FunCall(fun=ir.SymRef(id="plus"), args=[common, common])
    expected = ir.FunCall(
        fun=ir.Lambda(
            params=[ir.Sym(id="_cs_1")],
            expr=ir.FunCall(
                fun=ir.SymRef(id="plus"), args=[ir.SymRef(id="_cs_1"), ir.SymRef(id="_cs_1")]
            ),
        ),
        args=[common],
    )
    actual = CSE().visit(testee)
    assert actual == expected


def test_lambda_capture():
    common = ir.FunCall(fun=ir.SymRef(id="plus"), args=[ir.SymRef(id="x"), ir.SymRef(id="y")])
    testee = ir.FunCall(fun=ir.Lambda(params=[ir.Sym(id="x")], expr=common), args=[common])
    expected = testee
    actual = CSE().visit(testee)
    assert actual == expected


def test_lambda_no_capture():
    common = ir.FunCall(fun=ir.SymRef(id="plus"), args=[ir.SymRef(id="x"), ir.SymRef(id="y")])
    testee = ir.FunCall(fun=ir.Lambda(params=[ir.Sym(id="z")], expr=common), args=[common])
    expected = ir.FunCall(
        fun=ir.Lambda(
            params=[ir.Sym(id="_cs_1")],
            expr=ir.FunCall(
                fun=ir.Lambda(params=[ir.Sym(id="z")], expr=ir.SymRef(id="_cs_1")),
                args=[ir.SymRef(id="_cs_1")],
            ),
        ),
        args=[common],
    )
    actual = CSE().visit(testee)
    assert actual == expected


def test_lambda_nested_capture():
    def common_expr():
        return ir.FunCall(fun=ir.SymRef(id="plus"), args=[ir.SymRef(id="x"), ir.SymRef(id="y")])

    testee = ir.FunCall(
        fun=ir.Lambda(params=[ir.Sym(id="x"), ir.Sym(id="y")], expr=common_expr()),
        args=[common_expr(), common_expr()],
    )
    expected = ir.FunCall(
        fun=ir.Lambda(
            params=[ir.Sym(id="_cs_1")],
            expr=ir.FunCall(
                fun=ir.Lambda(params=[ir.Sym(id="x"), ir.Sym(id="y")], expr=common_expr()),
                args=[ir.SymRef(id="_cs_1"), ir.SymRef(id="_cs_1")],
            ),
        ),
        args=[common_expr()],
    )
    actual = CSE().visit(testee)
    assert actual == expected


def test_lambda_nested_capture_scoped():
    def common_expr():
        return ir.FunCall(fun=ir.SymRef(id="plus"), args=[ir.SymRef(id="x"), ir.SymRef(id="x")])

    # λ(x) → (λ(y) → y + (x + x + (x + x)))(z)
    testee = im.lambda__("x")(
        im.call_(im.lambda__("y")(im.plus_("y", im.plus_(common_expr(), common_expr()))))("z")
    )
    # λ(x) → (λ(_cs_1) → (λ(y) → y + (_cs_1 + _cs_1))(z))(x + x)
    expected = im.lambda__("x")(
        im.call_(
            im.lambda__("_cs_1")(
                im.call_(im.lambda__("y")(im.plus_("y", im.plus_("_cs_1", "_cs_1"))))("z")
            )
        )(common_expr())
    )
    actual = CSE().visit(testee)
    assert actual == expected
