import pathlib

from eve.pattern_matching import ObjectPattern
from functional.ffront import common_types as ct, field_operator_ast as foast
from functional.ffront.func_to_foast import UnrollPowerOp


def _make_power_testee(pow_n: str):
    loc = foast.SourceLocation(
        line=0, column=0, source="none")
    )

    testee = foast.BinOp(
        right=foast.Constant(
            dtype=ct.ScalarType(kind=ct.ScalarKind.INT),
            location=_set_source_location,
            value=pow_n,
            type=ct.ScalarType(kind=ct.ScalarKind.INT),
        ),
        left=foast.Name(
            id="a",
            type=ct.DeferredSymbolType(constraint=None),
            location=_set_source_location,
        ),
        op=foast.BinaryOperator.POW,
        location=_set_source_location,
        type=ct.DeferredSymbolType(constraint=None),
    )
    return testee


def test_power_1():

    actual = UnrollPowerOp().visit(_make_power_testee("1"))

    expected = ObjectPattern(foast.Name, id="a")

    assert expected.match(actual)


def test_power_2():

    actual = UnrollPowerOp().visit(_make_power_testee("2"))

    expected = ObjectPattern(
        foast.BinOp,
        right=ObjectPattern(foast.Name, id="a"),
        op=foast.BinaryOperator.MULT,
    )

    assert expected.match(actual)


def test_power_3():

    actual = UnrollPowerOp().visit(_make_power_testee("3"))

    expected = ObjectPattern(
        foast.BinOp,
        right=ObjectPattern(foast.Name, id="a"),
        left=ObjectPattern(
            foast.BinOp,
            right=ObjectPattern(foast.Name, id="a"),
            left=ObjectPattern(foast.Name, id="a"),
            op=foast.BinaryOperator.MULT,
        ),
        op=foast.BinaryOperator.MULT,
    )
    assert expected.match(actual)


def test_power_0():
    try:
        UnrollPowerOp().visit(_make_power_testee("0"))
    except ValueError:
        True


def test_power_neg():
    try:
        UnrollPowerOp().visit(_make_power_testee("-2"))
    except ValueError:
        True


def test_power_float():
    try:
        UnrollPowerOp().visit(_make_power_testee("6.7"))
    except ValueError:
        True


def test_power_reg_op():
    # expr: a + b ** 2
    _set_source_location = foast.SourceLocation(
        line=106, column=12, source=str(pathlib.Path(__file__).resolve())
    )
    pow_n = "2"

    testee = foast.BinOp(
        right=foast.BinOp(
            right=foast.Constant(
                dtype=ct.ScalarType(kind=ct.ScalarKind.INT),
                location=_set_source_location,
                value=pow_n,
                type=ct.ScalarType(kind=ct.ScalarKind.INT),
            ),
            left=foast.Name(
                id="a", type=ct.DeferredSymbolType(constraint=None), location=_set_source_location
            ),
            location=_set_source_location,
            type=ct.DeferredSymbolType(constraint=None),
            op=foast.BinaryOperator.POW,
        ),
        left=foast.Name(
            id="b",
            type=ct.DeferredSymbolType(constraint=None),
            location=_set_source_location,
        ),
        op=foast.BinaryOperator.ADD,
        location=_set_source_location,
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

    actual = UnrollPowerOp().visit(testee)

    assert expected.match(actual)
