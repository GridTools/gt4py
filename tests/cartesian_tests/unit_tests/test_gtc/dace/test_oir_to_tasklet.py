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
# This global variable add the marker to all test functions in this module.
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
        (oir.ScalarAccess(name="A"), False, "", "gtIN___A"),
        (oir.ScalarAccess(name="A"), True, "", "gtOUT___A"),
        (oir.ScalarAccess(name="A"), False, "im1", "gtIN___A_im1"),
        (
            oir.FieldAccess(name="A", offset=common.CartesianOffset(i=1, j=-1, k=0)),
            True,
            "",
            "gtOUT___A",
        ),
    ],
)
def test__tasklet_name(
    node: oir.FieldAccess | oir.ScalarAccess, is_target: bool, postfix: str, expected: str
) -> None:
    assert oir_to_tasklet._tasklet_name(node, is_target, postfix) == expected
