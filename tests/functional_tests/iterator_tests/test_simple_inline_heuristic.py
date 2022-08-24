import pytest

from functional.iterator import ir
from functional.iterator.transforms.simple_inline_heuristic import heuristic


@pytest.fixture
def scan():
    return ir.FunCall(
        fun=ir.SymRef(id="scan"),
        args=[
            ir.Lambda(
                params=[ir.Sym(id="acc"), ir.Sym(id="x")],
                expr=ir.FunCall(
                    fun=ir.SymRef(id="plus"),
                    args=[
                        ir.SymRef(id="acc"),
                        ir.FunCall(fun=ir.SymRef(id="deref"), args=[ir.SymRef(id="x")]),
                    ],
                ),
            ),
            ir.Literal(value="True", type="bool"),
            ir.Literal(value="0.0", type="float"),
        ],
    )


def test_trivial():
    testee = ir.FunCall(
        fun=ir.SymRef(id="lift"),
        args=[ir.SymRef(id="deref")],
    )
    predicate = heuristic(testee)
    assert predicate(testee)


def test_scan(scan):
    testee = ir.FunCall(fun=ir.SymRef(id="lift"), args=[scan])
    predicate = heuristic(testee)
    assert not predicate(testee)


def test_scan_with_lifted_arg(scan):
    testee = ir.FunCall(
        fun=ir.FunCall(fun=ir.SymRef(id="lift"), args=[scan]),
        args=[
            ir.FunCall(
                fun=ir.FunCall(
                    fun=ir.SymRef(id="lift"),
                    args=[ir.SymRef(id="deref")],
                ),
                args=[ir.SymRef(id="x")],
            )
        ],
    )
    predicate = heuristic(testee)
    assert not predicate(testee.fun)
    assert not predicate(testee.args[0].fun)
