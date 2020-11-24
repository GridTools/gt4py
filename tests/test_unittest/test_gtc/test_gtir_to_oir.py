from typing import Type

from eve import Node
from gt4py.gtc import gtir, oir, gtir_to_oir
from gt4py.gtc.common import DataType, ExprKind
from gt4py.gtc.gtir_to_oir import GTIRToOIR
from .gtir_utils import FieldAccessBuilder, FieldIfStmtBuilder

import pytest

A_ARITHMETIC_TYPE = DataType.FLOAT32


class DummyExpr(oir.Expr):
    dtype = A_ARITHMETIC_TYPE
    kind = ExprKind.FIELD


def isinstance_and_return(node: Node, expected_type: Type[Node]):
    assert isinstance(node, expected_type)
    return node


def test_visit_ParAssignStmt():
    out_name = "out"
    in_name = "in"
    testee = gtir.ParAssignStmt(
        left=FieldAccessBuilder(out_name).build(), right=FieldAccessBuilder(in_name).build()
    )

    result_decls, result_horizontal_executions = GTIRToOIR().visit(testee)

    assert len(result_decls) == 1
    assert isinstance(result_decls[0], oir.Temporary)
    tmp_name = result_decls[0].name

    assert len(result_horizontal_executions) == 2
    first_assign = isinstance_and_return(result_horizontal_executions[0].body[0], oir.AssignStmt)
    second_assign = isinstance_and_return(result_horizontal_executions[1].body[0], oir.AssignStmt)

    first_left = isinstance_and_return(first_assign.left, oir.FieldAccess)
    first_right = isinstance_and_return(first_assign.right, oir.FieldAccess)
    assert first_left.name == tmp_name
    assert first_right.name == in_name

    second_left = isinstance_and_return(second_assign.left, oir.FieldAccess)
    second_right = isinstance_and_return(second_assign.right, oir.FieldAccess)
    assert second_left.name == out_name
    assert second_right.name == tmp_name


def test_create_mask():
    mask_name = "mask"
    cond = DummyExpr(dtype=DataType.BOOL)
    result_decl, result_assign = gtir_to_oir._create_mask(mask_name, cond)

    assert isinstance(result_decl, oir.Temporary)
    assert result_decl.name == mask_name

    horizontal_exec = isinstance_and_return(result_assign, oir.HorizontalExecution)
    assign = isinstance_and_return(horizontal_exec.body[0], oir.AssignStmt)

    left = isinstance_and_return(assign.left, oir.FieldAccess)
    right = isinstance_and_return(assign.right, DummyExpr)

    assert left.name == mask_name
    assert right == cond


@pytest.mark.parametrize(
    "field_if_stmt",
    [
        # No else
        FieldIfStmtBuilder().cond(FieldAccessBuilder("cond").dtype(DataType.BOOL).build()).build(),
        # If and else
        FieldIfStmtBuilder()
        .cond(FieldAccessBuilder("cond").dtype(DataType.BOOL).build())
        .false_branch([])
        .build(),
        # Nested ifs
        FieldIfStmtBuilder()
        .cond(FieldAccessBuilder("cond").dtype(DataType.BOOL).build())
        .add_true_stmt(
            FieldIfStmtBuilder()
            .cond(FieldAccessBuilder("cond2").dtype(DataType.BOOL).build())
            .build()
        )
        .build(),
    ],
)
def test_visit_FieldIfStmt(field_if_stmt):
    # Testing only that lowering doesn't error.
    # I see no good testing strategy which is robust against changes of the lowering.
    GTIRToOIR().visit(field_if_stmt)
