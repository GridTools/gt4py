# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""`pparse` is meant to invert `pformat`; this asserts that it does.

`test_pretty_printer.py` and `test_pretty_parser.py` each check one direction
against a hand-written expectation and neither imports the other's module, so a
token added to the printer alone goes unnoticed by both.  The cases here are the
union of the terms those two files use, plus the ones that exposed a gap.
"""

import pytest

from gt4py.next.iterator import builtins, ir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.pretty_parser import pparse
from gt4py.next.iterator.pretty_printer import pformat
from gt4py.next.type_system import type_specifications as ts


_XFAIL_LITERAL_TYPE = pytest.mark.xfail(
    reason="`Literal.type` is not printed; the parser re-types the lexeme", strict=True
)


_SET_AT = ir.SetAt(expr=im.ref("x"), domain=im.call("cartesian_domain")(), target=im.ref("y"))


_LONG_NAME = (
    "very_loooooooooooooooooooong_condition_to_force_a_line_break_and_test_alignment_of_branches"
)


ROUNDTRIP_CASES = [
    pytest.param(im.ref("x"), id="symref"),
    pytest.param(im.lambda_("x")(im.ref("x")), id="lambda"),
    pytest.param(ir.OffsetLiteral(value="I"), id="offset_literal"),
    pytest.param(ir.OffsetLiteral(value=1), id="offset_literal_int"),
    pytest.param(
        im.divides_(
            im.multiplies_(
                im.plus(im.literal("1", "int64"), im.literal("2", "int64")),
                im.literal("3", "int64"),
            ),
            im.literal("4", "int64"),
        ),
        id="arithmetic",
        marks=_XFAIL_LITERAL_TYPE,
    ),
    pytest.param(
        im.plus(
            im.plus(im.literal("1", "int64"), im.literal("2", "int64")),
            im.plus(im.literal("3", "int64"), im.literal("4", "int64")),
        ),
        id="associativity",
        marks=_XFAIL_LITERAL_TYPE,
    ),
    pytest.param(im.deref("x"), id="deref"),
    pytest.param(im.call("lift")("x"), id="lift"),
    pytest.param(im.call("as_fieldop")("x"), id="as_fieldop"),
    pytest.param(
        im.not_(im.or_(im.not_("a"), im.and_("b", im.or_("c", "d")))),
        id="bool_arithmetic",
    ),
    pytest.param(
        im.call("shift")(im.ensure_offset("I"), im.ensure_offset(1)),
        id="shift",
    ),
    pytest.param(
        ir.CartesianOffset(domain=ir.AxisLiteral(value="I"), codomain=ir.AxisLiteral(value="J")),
        id="cartesian_offset",
    ),
    pytest.param(
        im.tuple_get(im.literal("42", builtins.INTEGER_INDEX_BUILTIN), "x"), id="tuple_get"
    ),
    pytest.param(im.make_tuple("x", "y"), id="make_tuple"),
    pytest.param(
        ir.AxisLiteral(value="I", kind=ir.DimensionKind.HORIZONTAL), id="axis_literal_horizontal"
    ),
    pytest.param(
        ir.AxisLiteral(value="I", kind=ir.DimensionKind.VERTICAL), id="axis_literal_vertical"
    ),
    pytest.param(
        im.named_range(ir.AxisLiteral(value="IDim"), "x", "y"), id="named_range_horizontal"
    ),
    pytest.param(
        im.named_range(ir.AxisLiteral(value="IDim", kind=ir.DimensionKind.VERTICAL), "x", "y"),
        id="named_range_vertical",
    ),
    pytest.param(im.call("cartesian_domain")("x"), id="cartesian_domain"),
    pytest.param(im.call("unstructured_domain")("x"), id="unstructured_domain"),
    pytest.param(im.if_("x", "y", "z"), id="if_short"),
    pytest.param(im.if_(_LONG_NAME, "y", "z"), id="if_long"),
    pytest.param(im.call("f")("x"), id="fun_call"),
    pytest.param(im.call(im.lambda_("x")(im.ref("x")))("x"), id="lambda_call"),
    pytest.param(
        ir.FunctionDefinition(id="f", params=[im.sym("x")], expr=im.ref("x")),
        id="function_definition",
    ),
    pytest.param(
        ir.Temporary(
            id="t", domain=im.ref("domain"), dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
        ),
        id="temporary",
    ),
    pytest.param(_SET_AT, id="set_at"),
    pytest.param(
        ir.IfStmt(
            cond=im.ref("cond"),
            true_branch=[
                _SET_AT,
                ir.IfStmt(cond=im.ref("cond"), true_branch=[_SET_AT], false_branch=[]),
            ],
            false_branch=[_SET_AT],
        ),
        id="if_stmt",
    ),
    pytest.param(
        ir.Program(
            id="f",
            function_definitions=[
                ir.FunctionDefinition(id="g", params=[im.sym("x")], expr=im.ref("x"))
            ],
            params=[im.sym("d"), im.sym("x"), im.sym("y")],
            declarations=[
                ir.Temporary(
                    id="tmp",
                    domain=im.call("cartesian_domain")(),
                    dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64),
                )
            ],
            body=[_SET_AT],
        ),
        id="program",
    ),
    pytest.param(ir.InfinityLiteral.POSITIVE, id="infinity_positive"),
    pytest.param(ir.InfinityLiteral.NEGATIVE, id="infinity_negative"),
    pytest.param(
        im.named_range(
            ir.AxisLiteral(value="KDim", kind=ir.DimensionKind.VERTICAL),
            ir.InfinityLiteral.NEGATIVE,
            im.literal("5", "int32"),
        ),
        id="named_range_unbounded",
    ),
    pytest.param(im.less_equal("a", "b"), id="less_equal"),
    pytest.param(im.greater_equal("a", "b"), id="greater_equal"),
]


@pytest.mark.parametrize("testee", ROUNDTRIP_CASES)
def test_pformat_pparse_roundtrip(testee):
    assert pparse(pformat(testee)) == testee
