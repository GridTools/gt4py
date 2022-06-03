import pathlib

import pytest

from eve.pattern_matching import ObjectPattern
from functional.ffront import common_types as ct, field_operator_ast as foast
from functional.ffront.foast_passes.type_deduction import FieldOperatorTypeDeductionError
from functional.ffront.foast_passes.unroll_power_op import UnrollPowerOp


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
  return (
    # exponent, expected
    (1, ObjectPattern(foast.Name, id="a")),
    (2, ObjectPattern(
        foast.BinOp, right=ObjectPattern(foast.Name, id="a"), op=foast.BinaryOperator.MULT
    )),
    (3, ObjectPattern(
        foast.BinOp,
        right=ObjectPattern(foast.Name, id="a"),
        left=ObjectPattern(
            foast.BinOp,
            right=ObjectPattern(foast.Name, id="a"),
            left=ObjectPattern(foast.Name, id="a"),
            op=foast.BinaryOperator.MULT,
        ),
        op=foast.BinaryOperator.MULT,
    ))
  )


@pytest.mark.parametrize("power_n,expected", [(1, power_1), (2, power_2), (3, power_3)])
def test_eval(power_n, expected):
    actual = UnrollPowerOp.apply(_make_power_testee(power_n))
    assert expected.match(actual)


def test_power_0():
    with pytest.raises(
        FieldOperatorTypeDeductionError,
        match="Only integer values greater than zero allowed in the power operation",
    ):
        _ = UnrollPowerOp.apply(_make_power_testee(0))


def test_power_neg():
    loc = foast.SourceLocation(
        line=1, column=1, source="none")
    )

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
            id="a",
            type=ct.DeferredSymbolType(constraint=None),
            location=loc,
        ),
        op=foast.BinaryOperator.POW,
        location=loc,
        type=ct.DeferredSymbolType(constraint=None),
    )

    with pytest.raises(
        FieldOperatorTypeDeductionError,
        match="Only integer values greater than zero allowed in the power operation",
    ):
        _ = UnrollPowerOp.apply(pow_neg())


def test_power_float():
    def pow_float():
        loc = foast.SourceLocation(
            line=106, column=12, source=str(pathlib.Path(__file__).resolve())
        )

        testee = foast.BinOp(
            right=foast.Constant(
                dtype=ct.ScalarType(kind=ct.ScalarKind.FLOAT64),
                location=loc,
                value=str(6.7),
                type=ct.ScalarType(kind=ct.ScalarKind.FLOAT64),
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

    with pytest.raises(
        FieldOperatorTypeDeductionError,
        match="Only integer values allowed in the power operation",
    ):
        _ = UnrollPowerOp.apply(pow_float())


def test_power_arithmetic_op():
    # expr: a + b ** 2
    loc = foast.SourceLocation(line=106, column=12, source=str(pathlib.Path(__file__).resolve()))
    pow_n = "2"

    testee = foast.BinOp(
        right=foast.BinOp(
            right=foast.Constant(
                dtype=ct.ScalarType(kind=ct.ScalarKind.INT),
                location=loc,
                value=pow_n,
                type=ct.ScalarType(kind=ct.ScalarKind.INT),
            ),
            left=foast.Name(id="a", type=ct.DeferredSymbolType(constraint=None), location=loc),
            location=loc,
            type=ct.DeferredSymbolType(constraint=None),
            op=foast.BinaryOperator.POW,
        ),
        left=foast.Name(
            id="b",
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
            right=ObjectPattern(foast.Name, id="a"),
            left=ObjectPattern(foast.Name, id="a"),
            op=foast.BinaryOperator.MULT,
        ),
        left=ObjectPattern(foast.Name, id="b"),
        op=foast.BinaryOperator.ADD,
    )

    actual = UnrollPowerOp.apply(testee)

    assert expected.match(actual)
