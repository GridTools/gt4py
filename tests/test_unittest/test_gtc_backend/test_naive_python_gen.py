import ast

import pytest

from gt4py.backend.gtc_backend.common import BinaryOperator
from gt4py.backend.gtc_backend.gtir import AssignStmt, BinaryOp, CartesianOffset, FieldAccess
from gt4py.backend.gtc_backend.python_naive_codegen import PythonNaiveCodegen


GTIROP_TO_ASTOP = {
    BinaryOperator.ADD: ast.Add,
    BinaryOperator.SUB: ast.Sub,
    BinaryOperator.MUL: ast.Mult,
    BinaryOperator.DIV: ast.Div,
}


@pytest.fixture
def naive_codegen():
    yield PythonNaiveCodegen()


@pytest.fixture(params=BinaryOperator.__members__.values())
def binary_operator(request):
    yield request.param


def test_field_ref(naive_codegen):
    field_ref = FieldAccess(name="a", offset=CartesianOffset(i=1, j=2, k=3))
    source_tree = ast.parse(naive_codegen.apply(field_ref))
    toplevel = source_tree.body[0]
    assert isinstance(toplevel, ast.Expr)
    assert isinstance(toplevel.value, ast.Subscript)


def test_assign(naive_codegen):
    assign = AssignStmt(left=FieldAccess.centered(name="a"), right=FieldAccess.centered(name="b"))
    source_tree = ast.parse(naive_codegen.apply(assign))
    assert isinstance(source_tree.body[0], ast.Assign)


def test_binary_op(naive_codegen, binary_operator):
    bin_op = BinaryOp(
        op=binary_operator,
        left=FieldAccess.centered(name="a"),
        right=FieldAccess.centered(name="b"),
    )
    source_tree = ast.parse(naive_codegen.apply(bin_op))
    toplevel = source_tree.body[0]
    assert isinstance(source_tree.body[0], ast.Expr)
    assert isinstance(toplevel.value, ast.BinOp)
    assert isinstance(toplevel.value.op, GTIROP_TO_ASTOP[binary_operator])
