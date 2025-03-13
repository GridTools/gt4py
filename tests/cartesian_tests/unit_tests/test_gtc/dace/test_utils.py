# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from typing import Optional

from gt4py.cartesian.gtc.common import DataType, CartesianOffset
from gt4py.cartesian.gtc.dace import daceir as dcir
from gt4py.cartesian.gtc.dace import prefix
from gt4py.cartesian.gtc.dace import utils

# Because "dace tests" filter by `requires_dace`, we still need to add the marker.
# This global variable add the marker to all test functions in this module.
pytestmark = pytest.mark.requires_dace


@pytest.mark.parametrize(
    "name,is_target,offset,expected",
    [
        ("A", False, None, f"{prefix.TASKLET_IN}A"),
        ("A", True, None, f"{prefix.TASKLET_OUT}A"),
        ("A", True, CartesianOffset(i=0, j=0, k=-1), f"{prefix.TASKLET_OUT}Akm1"),
        ("A", False, CartesianOffset(i=1, j=-2, k=3), f"{prefix.TASKLET_IN}Aip1_jm2_kp3"),
        (
            "A",
            True,
            dcir.VariableKOffset(k=dcir.Literal(value="3", dtype=DataType.INT32)),
            f"{prefix.TASKLET_OUT}A",
        ),
    ],
)
def test_get_tasklet_symbol(
    name: str,
    is_target: bool,
    offset: Optional[CartesianOffset | dcir.VariableKOffset],
    expected: str,
) -> None:
    assert utils.get_tasklet_symbol(name, is_target=is_target, offset=offset) == expected
