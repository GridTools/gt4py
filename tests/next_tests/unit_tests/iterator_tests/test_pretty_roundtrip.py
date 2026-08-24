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


_SET_AT = ir.SetAt(
    expr=ir.SymRef(id="x"),
    domain=ir.FunCall(fun=ir.SymRef(id="cartesian_domain"), args=[]),
    target=ir.SymRef(id="y"),
)


ROUNDTRIP_CASES = [
    pytest.param(ir.SymRef(id="x"), id="symref"),
    pytest.param(ir.Lambda(params=[ir.Sym(id="x")], expr=ir.SymRef(id="x")), id="lambda"),
    pytest.param(ir.OffsetLiteral(value="I"), id="offset_literal"),
    pytest.param(ir.OffsetLiteral(value=1), id="offset_literal_int"),
    pytest.param(
        ir.FunCall(
            fun=ir.SymRef(id="divides"),
            args=[
                ir.FunCall(
                    fun=ir.SymRef(id="multiplies"),
                    args=[
                        ir.FunCall(
                            fun=ir.SymRef(id="plus"),
                            args=[im.literal("1", "int64"), im.literal("2", "int64")],
                        ),
                        im.literal("3", "int64"),
                    ],
                ),
                im.literal("4", "int64"),
            ],
        ),
        id="arithmetic",
        marks=_XFAIL_LITERAL_TYPE,
    ),
    pytest.param(
        ir.FunCall(
            fun=ir.SymRef(id="plus"),
            args=[
                ir.FunCall(
                    fun=ir.SymRef(id="plus"),
                    args=[im.literal("1", "int64"), im.literal("2", "int64")],
                ),
                ir.FunCall(
                    fun=ir.SymRef(id="plus"),
                    args=[im.literal("3", "int64"), im.literal("4", "int64")],
                ),
            ],
        ),
        id="associativity",
        marks=_XFAIL_LITERAL_TYPE,
    ),
    pytest.param(ir.FunCall(fun=ir.SymRef(id="deref"), args=[ir.SymRef(id="x")]), id="deref"),
    pytest.param(ir.FunCall(fun=ir.SymRef(id="lift"), args=[ir.SymRef(id="x")]), id="lift"),
    pytest.param(
        ir.FunCall(fun=ir.SymRef(id="as_fieldop"), args=[ir.SymRef(id="x")]), id="as_fieldop"
    ),
    pytest.param(
        ir.FunCall(
            fun=ir.SymRef(id="not_"),
            args=[
                ir.FunCall(
                    fun=ir.SymRef(id="or_"),
                    args=[
                        ir.FunCall(fun=ir.SymRef(id="not_"), args=[ir.SymRef(id="a")]),
                        ir.FunCall(
                            fun=ir.SymRef(id="and_"),
                            args=[
                                ir.SymRef(id="b"),
                                ir.FunCall(
                                    fun=ir.SymRef(id="or_"),
                                    args=[ir.SymRef(id="c"), ir.SymRef(id="d")],
                                ),
                            ],
                        ),
                    ],
                )
            ],
        ),
        id="bool_arithmetic",
    ),
    pytest.param(
        ir.FunCall(
            fun=ir.SymRef(id="shift"),
            args=[ir.OffsetLiteral(value="I"), ir.OffsetLiteral(value=1)],
        ),
        id="shift",
    ),
    pytest.param(
        ir.CartesianOffset(domain=ir.AxisLiteral(value="I"), codomain=ir.AxisLiteral(value="J")),
        id="cartesian_offset",
    ),
    pytest.param(
        ir.FunCall(
            fun=ir.SymRef(id="tuple_get"),
            args=[im.literal("42", builtins.INTEGER_INDEX_BUILTIN), ir.SymRef(id="x")],
        ),
        id="tuple_get",
    ),
    pytest.param(
        ir.FunCall(fun=ir.SymRef(id="make_tuple"), args=[ir.SymRef(id="x"), ir.SymRef(id="y")]),
        id="make_tuple",
    ),
    pytest.param(
        ir.AxisLiteral(value="I", kind=ir.DimensionKind.HORIZONTAL), id="axis_literal_horizontal"
    ),
    pytest.param(
        ir.AxisLiteral(value="I", kind=ir.DimensionKind.VERTICAL), id="axis_literal_vertical"
    ),
    pytest.param(
        ir.FunCall(
            fun=ir.SymRef(id="named_range"),
            args=[ir.AxisLiteral(value="IDim"), ir.SymRef(id="x"), ir.SymRef(id="y")],
        ),
        id="named_range_horizontal",
    ),
    pytest.param(
        ir.FunCall(
            fun=ir.SymRef(id="named_range"),
            args=[
                ir.AxisLiteral(value="IDim", kind=ir.DimensionKind.VERTICAL),
                ir.SymRef(id="x"),
                ir.SymRef(id="y"),
            ],
        ),
        id="named_range_vertical",
    ),
    pytest.param(
        ir.FunCall(fun=ir.SymRef(id="cartesian_domain"), args=[ir.SymRef(id="x")]),
        id="cartesian_domain",
    ),
    pytest.param(
        ir.FunCall(fun=ir.SymRef(id="unstructured_domain"), args=[ir.SymRef(id="x")]),
        id="unstructured_domain",
    ),
    pytest.param(
        ir.FunCall(
            fun=ir.SymRef(id="if_"),
            args=[ir.SymRef(id="x"), ir.SymRef(id="y"), ir.SymRef(id="z")],
        ),
        id="if_short",
    ),
    pytest.param(
        ir.FunCall(
            fun=ir.SymRef(id="if_"),
            args=[
                ir.SymRef(
                    id="very_loooooooooooooooooooong_condition_to_force_a_line_break_and_test_alignment_of_branches"
                ),
                ir.SymRef(id="y"),
                ir.SymRef(id="z"),
            ],
        ),
        id="if_long",
    ),
    pytest.param(ir.FunCall(fun=ir.SymRef(id="f"), args=[ir.SymRef(id="x")]), id="fun_call"),
    pytest.param(
        ir.FunCall(
            fun=ir.Lambda(params=[ir.Sym(id="x")], expr=ir.SymRef(id="x")), args=[ir.SymRef(id="x")]
        ),
        id="lambda_call",
    ),
    pytest.param(
        ir.FunctionDefinition(id="f", params=[ir.Sym(id="x")], expr=ir.SymRef(id="x")),
        id="function_definition",
    ),
    pytest.param(
        ir.Temporary(
            id="t", domain=ir.SymRef(id="domain"), dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
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
                ir.FunctionDefinition(id="g", params=[ir.Sym(id="x")], expr=ir.SymRef(id="x"))
            ],
            params=[ir.Sym(id="d"), ir.Sym(id="x"), ir.Sym(id="y")],
            declarations=[
                ir.Temporary(
                    id="tmp",
                    domain=ir.FunCall(fun=ir.SymRef(id="cartesian_domain"), args=[]),
                    dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64),
                )
            ],
            body=[_SET_AT],
        ),
        id="program",
    ),
    pytest.param(
        ir.Temporary(
            id="t",
            domain=ir.SymRef(id="domain"),
            dtype=ts.TupleType(
                types=[
                    ts.ScalarType(kind=ts.ScalarKind.BOOL),
                    ts.ScalarType(kind=ts.ScalarKind.INT16),
                ]
            ),
        ),
        id="temporary_tuple_dtype",
    ),
    pytest.param(
        ir.Temporary(
            id="t",
            domain=ir.SymRef(id="domain"),
            dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64, shape=[3]),
        ),
        id="temporary_shaped_dtype",
    ),
    pytest.param(ir.InfinityLiteral.POSITIVE, id="infinity_positive"),
    pytest.param(ir.InfinityLiteral.NEGATIVE, id="infinity_negative"),
    pytest.param(
        ir.FunCall(
            fun=ir.SymRef(id="named_range"),
            args=[
                ir.AxisLiteral(value="KDim", kind=ir.DimensionKind.VERTICAL),
                ir.InfinityLiteral.NEGATIVE,
                im.literal("5", "int32"),
            ],
        ),
        id="named_range_unbounded",
    ),
    pytest.param(
        ir.FunCall(fun=ir.SymRef(id="less_equal"), args=[ir.SymRef(id="a"), ir.SymRef(id="b")]),
        id="less_equal",
    ),
    pytest.param(
        ir.FunCall(fun=ir.SymRef(id="greater_equal"), args=[ir.SymRef(id="a"), ir.SymRef(id="b")]),
        id="greater_equal",
    ),
]


@pytest.mark.parametrize("testee", ROUNDTRIP_CASES)
def test_pformat_pparse_roundtrip(testee):
    assert pparse(pformat(testee)) == testee
