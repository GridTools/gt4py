# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Type

import pytest

from gt4py.cartesian.gtc import oir
from gt4py.cartesian.gtc.gtir_to_oir import GTIRToOIR
from gt4py.eve import Node

from .gtir_utils import (
    FieldDeclFactory,
    FieldIfStmtFactory,
    HorizontalMaskFactory,
    HorizontalRestrictionFactory,
    ParAssignStmtFactory,
    ScalarIfStmtFactory,
    StencilFactory,
    VariableKOffsetFactory,
    VerticalLoopFactory,
)


def isinstance_and_return(node: Node, expected_type: Type[Node]):
    assert isinstance(node, expected_type)
    return node


def test_visit_ParAssignStmt():
    out_name = "out"
    in_name = "in"

    testee = ParAssignStmtFactory(left__name=out_name, right__name=in_name)
    assign = GTIRToOIR().visit(testee)
    left = isinstance_and_return(assign.left, oir.FieldAccess)
    right = isinstance_and_return(assign.right, oir.FieldAccess)
    assert left.name == out_name
    assert right.name == in_name


def test_visit_gtir_Stencil():
    out_name = "out"
    in_name = "in"

    testee = StencilFactory(
        vertical_loops__0__body__0=ParAssignStmtFactory(left__name=out_name, right__name=in_name)
    )
    oir_stencil = GTIRToOIR().visit(testee)
    hexecs = oir_stencil.vertical_loops[0].sections[0].horizontal_executions
    assert len(hexecs) == 1
    assert len(hexecs[0].body) == 1

    assign = hexecs[0].body[0]
    left = isinstance_and_return(assign.left, oir.FieldAccess)
    right = isinstance_and_return(assign.right, oir.FieldAccess)
    assert left.name == out_name
    assert right.name == in_name


def test_visit_FieldIfStmt():
    testee = FieldIfStmtFactory(true_branch__body__0=ParAssignStmtFactory())
    mask_stmts = GTIRToOIR().visit(testee, ctx=GTIRToOIR.Context())

    assert len(mask_stmts) == 2
    assert "mask" in mask_stmts[0].left.name
    assert testee.cond.name == mask_stmts[0].right.name
    assert mask_stmts[1].body[0].left.name == testee.true_branch.body[0].left.name


def test_visit_FieldIfStmt_nesting():
    testee = FieldIfStmtFactory(true_branch__body__0=FieldIfStmtFactory())
    GTIRToOIR().visit(testee, ctx=GTIRToOIR.Context())


def test_visit_ScalarIfStmt():
    testee = ScalarIfStmtFactory()
    GTIRToOIR().visit(testee, ctx=GTIRToOIR.Context())


def test_visit_HorizontalRestriction_HorizontalMask():
    testee = HorizontalRestrictionFactory(mask=HorizontalMaskFactory())
    GTIRToOIR().visit(testee, ctx=GTIRToOIR.Context())


def test_visit_Assign_VariableKOffset():
    testee = ParAssignStmtFactory(right__offset=VariableKOffsetFactory())
    assign_stmt = GTIRToOIR().visit(testee)
    assert assign_stmt.walk_values().if_isinstance(oir.VariableKOffset).to_list()


def test_indirect_read_with_offset_and_write():
    testee = StencilFactory(
        vertical_loops=[
            VerticalLoopFactory(
                temporaries=[FieldDeclFactory(name="tmp")],
                body=[
                    ParAssignStmtFactory(left__name="tmp", right__name="foo", right__offset__i=1),
                    ParAssignStmtFactory(right__name="tmp"),
                ],
            ),
            VerticalLoopFactory(body=[ParAssignStmtFactory(left__name="foo")]),
        ]
    )

    with pytest.raises(ValueError, match="non-zero read extent on written fields:.*foo"):
        GTIRToOIR().visit(testee)
