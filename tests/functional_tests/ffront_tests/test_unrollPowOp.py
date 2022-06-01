import pathlib

from eve.pattern_matching import ObjectPattern
from functional.ffront import common_types as ct, field_operator_ast as foast
from functional.ffront.func_to_foast import UnrollPowerOp


def test_power_0():
    testee = foast.BinOp(
        right=foast.Constant(
            dtype=ct.ScalarType(kind=ct.ScalarKind.INT),
            location=foast.SourceLocation(
                line=106, column=18, source=str(pathlib.Path(__file__).resolve())
            ),
            value="0",
            type=ct.ScalarType(kind=ct.ScalarKind.INT),
        ),
        left=foast.Name(
            id="a",
            type=ct.DeferredSymbolType(constraint=None),
            location=foast.SourceLocation(
                line=106, column=12, source=str(pathlib.Path(__file__).resolve())
            ),
        ),
        op=foast.BinaryOperator.POW,
        location=foast.SourceLocation(
            line=106, column=12, source=str(pathlib.Path(__file__).resolve())
        ),
        type=ct.DeferredSymbolType(constraint=None),
    )

    actual = UnrollPowerOp().visit(testee)

    expected = ObjectPattern(
        foast.BinOp,
        right=foast.Name(
            id="a",
            type=ct.DeferredSymbolType(constraint=None),
            location=foast.SourceLocation(
                line=106, column=12, source=str(pathlib.Path(__file__).resolve())
            ),
        ),
        op=foast.BinaryOperator.DIV,
    )
    assert expected.match(actual)


def test_power_1():
    testee = foast.BinOp(
        right=foast.Constant(
            dtype=ct.ScalarType(kind=ct.ScalarKind.INT),
            location=foast.SourceLocation(
                line=106, column=18, source=str(pathlib.Path(__file__).resolve())
            ),
            value="1",
            type=ct.ScalarType(kind=ct.ScalarKind.INT),
        ),
        left=foast.Name(
            id="a",
            type=ct.DeferredSymbolType(constraint=None),
            location=foast.SourceLocation(
                line=106, column=12, source=str(pathlib.Path(__file__).resolve())
            ),
        ),
        op=foast.BinaryOperator.POW,
        location=foast.SourceLocation(
            line=106, column=12, source=str(pathlib.Path(__file__).resolve())
        ),
        type=ct.DeferredSymbolType(constraint=None),
    )

    actual = UnrollPowerOp().visit(testee)

    expected = ObjectPattern(foast.Name(id="a"))

    assert expected.match(actual)


def test_power_2():

    testee = foast.BinOp(
        right=foast.Constant(
            dtype=ct.ScalarType(kind=ct.ScalarKind.INT),
            location=foast.SourceLocation(
                line=106, column=18, source=str(pathlib.Path(__file__).resolve())
            ),
            value="2",
            type=ct.ScalarType(kind=ct.ScalarKind.INT),
        ),
        left=foast.Name(
            id="a",
            type=ct.DeferredSymbolType(constraint=None),
            location=foast.SourceLocation(
                line=106, column=12, source=str(pathlib.Path(__file__).resolve())
            ),
        ),
        op=foast.BinaryOperator.POW,
        location=foast.SourceLocation(
            line=106, column=12, source=str(pathlib.Path(__file__).resolve())
        ),
        type=ct.DeferredSymbolType(constraint=None),
    )

    actual = UnrollPowerOp().visit(testee)

    expected = ObjectPattern(
        foast.BinOp,
        right=foast.Name(
            id="a",
            type=ct.DeferredSymbolType(constraint=None),
            location=foast.SourceLocation(
                line=106, column=12, source=str(pathlib.Path(__file__).resolve())
            ),
        ),
        op=foast.BinaryOperator.MULT,
    )

    assert expected.match(actual)


def test_power_3():
    testee = foast.BinOp(
        right=foast.Constant(
            dtype=ct.ScalarType(kind=ct.ScalarKind.INT),
            location=foast.SourceLocation(
                line=106, column=18, source=str(pathlib.Path(__file__).resolve())
            ),
            value="3",
            type=ct.ScalarType(kind=ct.ScalarKind.INT),
        ),
        left=foast.Name(
            id="a",
            type=ct.DeferredSymbolType(constraint=None),
            location=foast.SourceLocation(
                line=106, column=12, source=str(pathlib.Path(__file__).resolve())
            ),
        ),
        op=foast.BinaryOperator.POW,
        location=foast.SourceLocation(
            line=106, column=12, source=str(pathlib.Path(__file__).resolve())
        ),
        type=ct.DeferredSymbolType(constraint=None),
    )

    actual = UnrollPowerOp().visit(testee)

    expected = ObjectPattern(
        foast.BinOp,
        right=foast.Name(
            id="a",
            type=ct.DeferredSymbolType(constraint=None),
            location=foast.SourceLocation(
                line=106, column=12, source=str(pathlib.Path(__file__).resolve())
            ),
        ),
        left=foast.BinOp(
            right=foast.Name(
                id="a",
                type=ct.DeferredSymbolType(constraint=None),
                location=foast.SourceLocation(
                    line=106, column=12, source=str(pathlib.Path(__file__).resolve())
                ),
            ),
            left=foast.Name(
                id="a",
                type=ct.DeferredSymbolType(constraint=None),
                location=foast.SourceLocation(
                    line=106, column=12, source=str(pathlib.Path(__file__).resolve())
                ),
            ),
            location=foast.SourceLocation(
                line=106, column=12, source=str(pathlib.Path(__file__).resolve())
            ),
            type=ct.DeferredSymbolType(constraint=None),
            op=foast.BinaryOperator.MULT,
        ),
        op=foast.BinaryOperator.MULT,
    )
    assert expected.match(actual)
