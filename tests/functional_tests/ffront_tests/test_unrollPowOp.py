import eve
from eve.pattern_matching import ObjectPattern
from functional.ffront import field_operator_ast as foast
from functional.ffront.func_to_foast import UnrollPowerOp
from functional.iterator.ir import Literal, Sym, SymRef


def test_power_0():
    print(3)


def test_power_1():
    print(3)


# a**2
def test_power_2():

    testee = foast.BinOp(
        left=Sym(id="a"), right=Literal(value="2.0", type=str), op=foast.BinaryOperator.POW
    )
    expected = foast.BinOp(
        left=SymRef(id="a"),
        right=SymRef(id="a"),
        op=foast.BinaryOperator.MULT,
        location=SymRef(id="b"),
    )

    # a*a

    actual = UnrollPowerOp(eve.NodeTranslator).visit(testee)
    # testee.match(expected, raise_exception=True)
    # assert ObjectPattern(foast.BinOp, left=SymRef(id="a"), right=Literal(value=2, dtype=float), op=foast.BinaryOperator.POW).match(foast.BinOp(left=SymRef(id="a"), right=SymRef(id="a"), op=foast.BinaryOperator.MULT))
    assert actual == expected


def test_power_3():
    print(3)
