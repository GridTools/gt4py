import pytest

from eve.utils import UIDs
from functional.iterator import ir
from functional.iterator.transforms.cse import CommonSubexpressionElimination as CSE


@pytest.fixture
def fresh_uid_sequence():
    UIDs.reset_sequence()


def test_trivial(fresh_uid_sequence):
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


def test_lambda_capture(fresh_uid_sequence):
    common = ir.FunCall(fun=ir.SymRef(id="plus"), args=[ir.SymRef(id="x"), ir.SymRef(id="y")])
    testee = ir.FunCall(fun=ir.Lambda(params=[ir.Sym(id="x")], expr=common), args=[common])
    expected = testee
    actual = CSE().visit(testee)
    assert actual == expected


def test_lambda_no_capture(fresh_uid_sequence):
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


def test_lambda_nested_capture(fresh_uid_sequence):
    common = ir.FunCall(fun=ir.SymRef(id="plus"), args=[ir.SymRef(id="x"), ir.SymRef(id="y")])
    testee = ir.FunCall(
        fun=ir.Lambda(params=[ir.Sym(id="x"), ir.Sym(id="y")], expr=common), args=[common, common]
    )
    expected = ir.FunCall(
        fun=ir.Lambda(
            params=[ir.Sym(id="_cs_1")],
            expr=ir.FunCall(
                fun=ir.Lambda(params=[ir.Sym(id="x"), ir.Sym(id="y")], expr=common),
                args=[ir.SymRef(id="_cs_1"), ir.SymRef(id="_cs_1")],
            ),
        ),
        args=[common],
    )
    actual = CSE().visit(testee)
    assert actual == expected
