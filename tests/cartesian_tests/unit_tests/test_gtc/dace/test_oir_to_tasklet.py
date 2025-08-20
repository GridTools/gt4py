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
