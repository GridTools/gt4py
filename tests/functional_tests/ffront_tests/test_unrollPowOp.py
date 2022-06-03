import pathlib

import pytest

from eve.pattern_matching import ObjectPattern
from functional.ffront import common_types as ct, field_operator_ast as foast
from functional.ffront.foast_passes.unroll_power_op import FieldOperatorPowerError, UnrollPowerOp


def _make_power_testee(pow_n: int) -> foast.BinOp:
    loc = foast.SourceLocation(line=106, column=12, source=str(pathlib.Path(__file__).resolve()))

    testee = foast.BinOp(
        right=foast.Constant(
            dtype=ct.ScalarType(kind=ct.ScalarKind.INT),
            location=loc,
            value=str(pow_n),
            type=ct.ScalarType(kind=ct.ScalarKind.INT),
        ),
        left=foast.Name(
            id="a",
            type=ct.DeferredSymbolType(constraint=None),
            location=loc,
        ),
        op=foast.BinaryOperator.POW,
        location=loc,
        type=ct.DeferredSymbolType(constraint=None),
    )
    return testee


def power_test_cases():
    return [
        # exponent, expected
        (1, ObjectPattern(foast.Name, id=foast.SymbolRef("a"))),
        (
            2,
            ObjectPattern(
                foast.BinOp,
                right=ObjectPattern(foast.Name, id=foast.SymbolRef("a")),
                op=foast.BinaryOperator.MULT,
            ),
        ),
        (
            3,
            ObjectPattern(
                foast.BinOp,
                right=ObjectPattern(foast.Name, id=foast.SymbolRef("a")),
                left=ObjectPattern(
                    foast.BinOp,
                    right=ObjectPattern(foast.Name, id=foast.SymbolRef("a")),
                    left=ObjectPattern(foast.Name, id=foast.SymbolRef("a")),
                    op=foast.BinaryOperator.MULT,
                ),
                op=foast.BinaryOperator.MULT,
            ),
        ),
    ]


@pytest.mark.parametrize("power_n,expected", power_test_cases())
def test_eval(power_n, expected):
    actual = UnrollPowerOp.apply(_make_power_testee(power_n))
    assert expected.match(actual)


def test_power_0():
    with pytest.raises(
        FieldOperatorPowerError,
        match="Only integer values greater than zero allowed in the power operation",
    ):
        _ = UnrollPowerOp.apply(_make_power_testee(0))


def test_power_neg_exponent():
    loc = foast.SourceLocation(line=1, column=1, source="none")

    testee = foast.BinOp(
        right=foast.UnaryOp(
            location=loc,
            op=foast.UnaryOperator.USUB,
            operand=foast.Constant(
                dtype=ct.ScalarType(kind=ct.ScalarKind.INT),
                location=loc,
                value=str(2),
                type=ct.ScalarType(kind=ct.ScalarKind.INT),
            ),
            type=ct.DeferredSymbolType(constraint=None),
        ),
        left=foast.Name(
            id=foast.SymbolRef("a"),
            type=ct.DeferredSymbolType(constraint=None),
            location=loc,
        ),
        op=foast.BinaryOperator.POW,
        location=loc,
        type=ct.DeferredSymbolType(constraint=None),
    )

    with pytest.raises(
        FieldOperatorPowerError,
        match="Exponent must be a constant value.",
    ):
        _ = UnrollPowerOp.apply(testee)


def test_power_float_exponent():

    loc = foast.SourceLocation(line=1, column=1, source="none")

    testee = foast.BinOp(
        right=foast.Constant(
            dtype=ct.ScalarType(kind=ct.ScalarKind.FLOAT64),
            location=loc,
            value=str(6.7),
            type=ct.ScalarType(kind=ct.ScalarKind.FLOAT64),
        ),
        left=foast.Name(
            id=foast.SymbolRef("a"),
            type=ct.DeferredSymbolType(constraint=None),
            location=loc,
        ),
        op=foast.BinaryOperator.POW,
        location=loc,
        type=ct.DeferredSymbolType(constraint=None),
    )

    with pytest.raises(
        FieldOperatorPowerError,
        match="Only integer values greater than zero allowed in the power operation",
    ):
        _ = UnrollPowerOp.apply(testee)


def test_power_arithmetic_op():
    # expr: a + b ** 2
    loc = foast.SourceLocation(line=1, column=1, source="none")

    testee = foast.BinOp(
        right=_make_power_testee(2),
        left=foast.Name(
            id=foast.SymbolRef("b"),
            type=ct.DeferredSymbolType(constraint=None),
            location=loc,
        ),
        op=foast.BinaryOperator.ADD,
        location=loc,
        type=ct.DeferredSymbolType(constraint=None),
    )

    expected = ObjectPattern(
        foast.BinOp,
        right=ObjectPattern(
            foast.BinOp,
            right=ObjectPattern(foast.Name, id=foast.SymbolRef("a")),
            left=ObjectPattern(foast.Name, id=foast.SymbolRef("a")),
            op=foast.BinaryOperator.MULT,
        ),
        left=ObjectPattern(foast.Name, id=foast.SymbolRef("b")),
        op=foast.BinaryOperator.ADD,
    )

    actual = UnrollPowerOp.apply(testee)

    assert expected.match(actual)
