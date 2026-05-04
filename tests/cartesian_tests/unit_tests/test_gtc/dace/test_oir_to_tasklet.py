# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from gt4py.cartesian.gtc.dace import oir_to_tasklet
from gt4py.cartesian.gtc import oir, common

# Because "dace tests" filter by `requires_dace`, we still need to add the marker.
# This global variable adds the marker to all test functions in this module.
pytestmark = pytest.mark.requires_dace


@pytest.mark.parametrize(
    "node,expected",
    [
        (
            oir.FieldAccess(
                name="A",
                offset=oir.VariableKOffset(k=oir.Literal(value="1", dtype=common.DataType.AUTO)),
            ),
            "var_k",
        ),
        (oir.FieldAccess(name="A", offset=common.CartesianOffset(i=1, j=-1, k=0)), "ip1_jm1"),
    ],
)
def test__field_offset_postfix(node: oir.FieldAccess, expected: str) -> None:
    assert oir_to_tasklet._field_offset_postfix(node) == expected


@pytest.mark.parametrize(
    "node,is_target,postfix,expected",
    [
        (oir.ScalarAccess(name="A"), False, "", "gtIN__A"),
        (oir.ScalarAccess(name="A"), True, "", "gtOUT__A"),
        (oir.ScalarAccess(name="A"), False, "im1", "gtIN__A_im1"),
        (
            oir.FieldAccess(name="A", offset=common.CartesianOffset(i=1, j=-1, k=0)),
            True,
            "",
            "gtOUT__A",
        ),
    ],
)
def test__tasklet_name(
    node: oir.FieldAccess | oir.ScalarAccess, is_target: bool, postfix: str, expected: str
) -> None:
    assert oir_to_tasklet._tasklet_name(node, is_target, postfix) == expected


@pytest.mark.parametrize(
    "literal,expected",
    [
        (oir.Literal(value=common.BuiltInLiteral.TRUE, dtype=common.DataType.BOOL), "True"),
        (oir.Literal(value=common.BuiltInLiteral.FALSE, dtype=common.DataType.BOOL), "False"),
        (oir.Literal(value="42.0", dtype=common.DataType.FLOAT32), "float(42.0)"),
        (oir.Literal(value="42.0", dtype=common.DataType.FLOAT64), "double(42.0)"),
        (oir.Literal(value="42", dtype=common.DataType.INT32), "int(42)"),
        (oir.Literal(value="42", dtype=common.DataType.INT64), "int64_t(42)"),
    ],
)
def test_visit_literal(literal: oir.Literal, expected: str):
    visitor = oir_to_tasklet.OIRToTasklet()

    assert visitor.visit_Literal(literal) == expected


@pytest.mark.parametrize(
    "exponent,expected_ipow_usage",
    [
        (oir.Literal(value="0", dtype=common.DataType.INT32), False),
        (oir.Literal(value="1", dtype=common.DataType.INT32), True),
        (oir.Literal(value="2", dtype=common.DataType.INT32), True),
        (oir.Literal(value="3", dtype=common.DataType.INT32), True),
        (oir.Literal(value="3", dtype=common.DataType.INT16), True),
        (oir.Literal(value="3", dtype=common.DataType.INT8), True),
        (oir.Literal(value="3", dtype=common.DataType.INT64), True),
        (oir.Literal(value="4", dtype=common.DataType.INT32), False),
        (oir.Literal(value="2.0", dtype=common.DataType.FLOAT32), False),
        (
            oir.BinaryOp(
                op=common.ArithmeticOperator.ADD,
                left=oir.Literal(value="1", dtype=common.DataType.INT32),
                right=oir.Literal(value="0", dtype=common.DataType.INT32),
            ),
            False,
        ),
        (
            oir.UnaryOp(
                op=common.UnaryOperator.NEG,
                expr=oir.Literal(value="2", dtype=common.DataType.INT32),
            ),
            False,
        ),
    ],
)
def test_integer_power_function(exponent: oir.Expr, expected_ipow_usage: bool) -> None:
    base = oir.ScalarAccess(name="A", dtype=common.DataType.FLOAT32)
    pow_call = oir.NativeFuncCall(func=common.NativeFunction.POW, args=[base, exponent])

    visitor = oir_to_tasklet.OIRToTasklet()
    fake_context = oir_to_tasklet.Context(
        code="asdf", targets=set(), inputs={}, outputs={}, tree=None, scope=None
    )
    tasklet_code = visitor.visit_NativeFuncCall(pow_call, ctx=fake_context, is_target=False)

    uses_ipow = "ipow" in tasklet_code
    assert uses_ipow == expected_ipow_usage


def test_integer_power_zero() -> None:
    base = oir.ScalarAccess(name="A", dtype=common.DataType.FLOAT32)
    exponent = oir.Literal(value="0", dtype=common.DataType.INT32)
    pow_call = oir.NativeFuncCall(func=common.NativeFunction.POW, args=[base, exponent])

    visitor = oir_to_tasklet.OIRToTasklet()
    fake_context = oir_to_tasklet.Context(
        code="asdf", targets=set(), inputs={}, outputs={}, tree=None, scope=None
    )
    tasklet_code = visitor.visit_NativeFuncCall(pow_call, ctx=fake_context, is_target=False)

    assert tasklet_code == "float(1)"


def test_integer_power_of_integer() -> None:
    base = oir.ScalarAccess(name="A", dtype=common.DataType.INT32)
    exponent = oir.Literal(value="2", dtype=common.DataType.INT32)
    pow_call = oir.NativeFuncCall(func=common.NativeFunction.POW, args=[base, exponent])

    visitor = oir_to_tasklet.OIRToTasklet()
    fake_context = oir_to_tasklet.Context(
        code="asdf", targets=set(), inputs={}, outputs={}, tree=None, scope=None
    )
    tasklet_code = visitor.visit_NativeFuncCall(pow_call, ctx=fake_context, is_target=False)

    assert "ipow" not in tasklet_code
